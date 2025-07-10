import os
import time
import shutil
import torch
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from enhance import enhance
from evaluate import evaluate
from utils import bold, copy_state, pull_metric, swap_state, LogProgress
from criteria import CompositeLoss, batch_pesq

class Solver(object):
    def __init__(
        self, 
        data, 
        model,
        discriminator,
        optim, 
        optim_disc,
        scheduler,
        scheduler_disc,
        args, 
        logger, 
        rank=0,     
        world_size=1, 
        device=None
    ):
        # Dataloaders and samplers
        self.tr_loader = data['tr_loader']      # Training DataLoader
        self.va_loader = data['va_loader']      # Validation DataLoader
        self.tt_loader = data['tt_loader']      # Test DataLoader for checking result samples
        self.ev_loader = data['ev_loader']      # Evaluation DataLoader
        self.tr_sampler = data['tr_sampler']    # Distributed sampler for training
        
        self.model = model
        self.discriminator = discriminator
        self.optim = optim
        self.optim_disc = optim_disc
        self.scheduler = scheduler
        self.scheduler_disc = scheduler_disc
        self.logger = logger

        # Basic config
        self.device = device or torch.device(args.device)
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        
        self.epochs = args.epochs
        self.loss = args.loss
        self.clip_grad_norm = args.clip_grad_norm
        
        self.loss = CompositeLoss(args.loss, self.discriminator).to(self.device)

        self.eval_every = args.eval_every   # interval for evaluation
        self.validate_with_pesq = args.validate_with_pesq
            
        # Checkpoint settings
        self.checkpoint = args.checkpoint
        
        self.continue_from = args.continue_from
        
        if self.checkpoint and self.rank == 0:
            self.checkpoint_file = Path(args.checkpoint_file)
            self.logger.info("Checkpoint will be saved to %s", self.checkpoint_file.resolve())

        self.writer = None
        self.best_state = None
        self.history = []
        self.log_dir = args.log_dir
        self.samples_dir = args.samples_dir
        self.num_prints = args.num_prints
        self.args = args
        
        # Initialize or resume (checkpoint loading)
        self._reset()

    def _serialize(self):
        """ Save checkpoint and best model (only rank=0).
        
            - We save both a 'checkpoint.th' file and a 'best.th' file.
            - The 'checkpoint.th' contains model state, optimizer state, and history.
            - The 'best.th' contains the best performing model state so far.
        """
        if self.rank != 0:  # save only on rank 0
            return
        
        # Create a package dict
        package = {}
        if self.is_distributed:
            # If using DDP, we need to copy the state dict from the module
            package['model'] = copy_state(self.model.module.state_dict())
            package['discriminator'] = copy_state(self.discriminator.module.state_dict()) if self.discriminator is not None else None
        else:
            # In non-distributed mode, just use the model's state dict directly
            package['model'] = copy_state(self.model.state_dict())
            package['discriminator'] = copy_state(self.discriminator.state_dict()) if self.discriminator is not None else None
        package['optimizer'] = self.optim.state_dict()
        package['optimizer_disc'] = self.optim_disc.state_dict() if self.optim_disc is not None else None
        package['scheduler'] = self.scheduler.state_dict() if self.scheduler is not None else None
        package['scheduler_disc'] = self.scheduler_disc.state_dict() if self.scheduler_disc is not None else None
        package['history'] = self.history
        package['args'] = self.args
        
        # Write to a temporary file first
        tmp_path = str(self.checkpoint_file) + ".tmp"
        torch.save(package, tmp_path)
        os.rename(tmp_path, self.checkpoint_file)

        # Save the best model separately to 'best.th'
        best_path = Path("best.th")
        tmp_path = str(best_path) + ".tmp"
        torch.save(self.best_state, tmp_path)
        os.rename(tmp_path, best_path)

    def _reset(self):
        """Load checkpoint if 'continue_from' is specified, or create a fresh writer if not."""
        if self.continue_from is not None:
            if self.rank == 0:
                self.logger.info(f'Loading checkpoint model: {self.continue_from}')
                if not os.path.exists(self.continue_from):
                    raise FileNotFoundError(f"Checkpoint directory {self.continue_from} not found.")
                
                # Attempt to copy the 'tensorbd' directory (TensorBoard logs) if it exists
                src_tb_dir = os.path.join(self.continue_from, 'tensorbd')
                dst_tb_dir = self.log_dir
                
                if os.path.exists(src_tb_dir):
                    # If the previous tensorboard logs exist, we either copy them
                    # to the new log dir or skip if it already exists.
                    if not os.path.exists(dst_tb_dir):
                        shutil.copytree(src_tb_dir, dst_tb_dir)
                    else:
                        # If the new log dir already exists, just issue a warning and do not overwrite
                        self.logger.warning(f"TensorBoard log dir {dst_tb_dir} already exists. Skipping copy.")
                    # Initialize the SummaryWriter to continue logging in the (possibly copied) directory
                    self.writer = SummaryWriter(log_dir=dst_tb_dir)
            
            package = None  # Initialize package to None for non-rank 0 processes
            
            # Rank 0 loads the checkpoint file from disk
            if self.rank == 0:
                ckpt_path = os.path.join(self.continue_from, 'checkpoint.th')
                if not os.path.exists(ckpt_path):
                    raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found.")
                self.logger.info(f"Loading checkpoint from {ckpt_path}")
                package = torch.load(ckpt_path, map_location='cpu')
                
            if self.is_distributed:
                # Wait until rank 0 finishes loading the checkpoint
                dist.barrier()
                
                # Broadcast the loaded checkpoint object to all ranks
                obj_list = [package]
                dist.broadcast_object_list(obj_list, src=0)
                package = obj_list[0]  # Extract the broadcasted package
            
            model_state = package['model']
            model_disc_state = package.get('discriminator', None)
            optim_state = package['optimizer']
            optim_disc_state = package.get('optimizer_disc', None)
            scheduler_state = package.get('scheduler', None)
            scheduler_disc_state = package.get('scheduler_disc', None)
            
            target_model = self.model.module if self.is_distributed else self.model
            target_model.load_state_dict(model_state)
            self.optim.load_state_dict(optim_state)
            
            if self.discriminator is not None and model_disc_state is not None:
                target_disc = self.discriminator.module if self.is_distributed else self.discriminator
                target_disc.load_state_dict(model_disc_state)
            
            if self.optim_disc is not None and optim_disc_state is not None:
                self.optim_disc.load_state_dict(optim_disc_state)
            
            if self.scheduler is not None and scheduler_state is not None:
                self.scheduler.load_state_dict(scheduler_state)
            
            if self.scheduler_disc is not None and scheduler_disc_state is not None:
                self.scheduler_disc.load_state_dict(scheduler_disc_state)
            
            # Now attempt to load the best checkpoint if it exists
            best_path = os.path.join(self.continue_from, 'best.th')
            if os.path.exists(best_path):
                self.logger.info(f"Loading best model from {best_path}")
                self.best_state = torch.load(best_path, 'cpu')
                
            else:
                # If best.th does not exist, create a fallback best_state from the current model
                self.best_state = {
                    'model': copy_state(
                        self.model.module.state_dict() if self.is_distributed 
                        else self.model.state_dict()
                    )
                }
            
            # If there's any historical training metrics in the checkpoint, restore them
            if 'history' in package:
                self.history = package['history']
        else:
            # If there's no checkpoint to resume from, just create a fresh SummaryWriter
            self.writer = SummaryWriter(log_dir=self.log_dir)

    def train(self):

        # If there's a history from the checkpoint, replay metrics
        if self.history and self.rank == 0:  
            self.logger.info("Replaying metrics from previous run")
            for epoch, metrics in enumerate(self.history):
                info = " ".join(f"{k.capitalize()}={v:.5f}" for k, v in metrics.items())
                self.logger.info(f"Epoch {epoch + 1}: {info}")
        
        if self.rank == 0:
            self.logger.info(f"Training for {self.epochs} epochs")
        
        # Main epoch loop
        for epoch in range(len(self.history), self.epochs):
            # Switch to training mode
            self.model.train()
            
            # If using a distributed sampler, set its epoch to ensure different random ordering each epoch
            if self.tr_sampler is not None:
                self.tr_sampler.set_epoch(epoch)
            
            start = time.time()
            if self.rank == 0:
                self.logger.info('-' * 70)
                self.logger.info("Training...")
            
            # Run one epoch of training
            train_loss = self._run_one_epoch(epoch)
            
            if self.rank == 0:
                self.logger.info(
                    bold(f'Train Summary | End of Epoch {epoch + 1} | '
                         f'Time {time.time() - start:.2f}s | Train Loss {train_loss:.5f}'))
            
            if self.is_distributed:
                dist.barrier()
                
            # Optionally run validation if va_loader is present
            self.model.eval()
            

            # Running validation with PESQ can cause errors in a DDP environment, 
            # so validation is not performed using DDP.
            if self.rank == 0:
                self.logger.info('-' * 70)
                self.logger.info('Validation...')
                with torch.no_grad():
                    if self.validate_with_pesq:
                        valid_loss = self._validate_with_pesq(epoch)
                    else:
                        valid_loss = self._run_one_epoch(epoch, valid=True)
            
            if self.rank == 0:
                self.logger.info(
                    bold(f'Valid Summary | End of Epoch {epoch + 1} | '
                        f'Time {time.time() - start:.2f}s | Valid Loss {valid_loss:.5f}'))
                
                if self.scheduler is not None:
                    self.logger.info(f"Learning rate: {self.scheduler.get_last_lr()[0]:.6f}")
                    self.writer.add_scalar("train/Learning_Rate", self.scheduler.get_last_lr()[0], epoch)
                
            # rank=0 handles model saving and test set evaluation
            if self.rank == 0:
                best_loss = min(pull_metric(self.history, 'valid') + [valid_loss])
                metrics = {'train': train_loss, 'valid': valid_loss, 'best': best_loss}
                
                # Update best_state if we got a new best validation loss
                if valid_loss == best_loss:
                    if self.validate_with_pesq:
                        self.logger.info(bold('New best valid PESQ score %.4f'), (-valid_loss))
                    else:
                        self.logger.info(bold('New best valid loss %.4f'), valid_loss)
                    self.best_state = {'model': copy_state(self.model.module.state_dict() if self.is_distributed else self.model.state_dict())}

                # Evaluate on ev_loader (test set) every eval_every epochs (or last epoch)
                if self.eval_every is not None:
                    if ((epoch + 1) % self.eval_every == 0 or epoch == self.epochs - 1):
                        self.logger.info('-' * 70)
                        self.logger.info('Evaluating on the test set...')
                        
                        # Temporarily swap model weights with best_state for evaluation
                        with swap_state(self.model.module if self.is_distributed else self.model, self.best_state['model']):
                            ev_metric = evaluate(
                                args= self.args, 
                                model= self.model, 
                                data_loader=self.ev_loader,
                                logger=self.logger, 
                                epoch=epoch
                            )

                            self.logger.info('Enhance and save samples...')
                            enhance(
                                args= self.args, 
                                model= self.model, 
                                data_loader= self.tt_loader, 
                                logger= self.logger, 
                                epoch= epoch, 
                                local_out_dir=self.samples_dir
                            )
                    
                        for k, v in ev_metric.items():
                            self.writer.add_scalar(f"test/{k}", v, epoch)
                
                # Append metrics to history and print summary
                self.history.append(metrics)
                info = " | ".join(f"{k.capitalize()} {v:.5f}" for k, v in metrics.items())
                self.logger.info('-' * 70)
                self.logger.info(bold(f"Overall Summary | Epoch {epoch + 1} | {info}"))
                
                # Checkpoint serialization
                if self.checkpoint:
                    self._serialize()
                    self.logger.debug("Checkpoint saved to %s", self.checkpoint_file.resolve())
                    

    def _run_one_epoch(self, epoch, valid=False):
        """Run one epoch of training or validation."""
        total_loss = 0.0
        data_loader = self.tr_loader if not valid else self.va_loader
        
         # If the sampler has a set_epoch method, call it to shuffle data consistently across ranks
        if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
            data_loader.sampler.set_epoch(epoch)

        label = ["Train", "Valid"][valid]
        name = label + f" | Epoch {epoch + 1}"
        
        # For rank=0, use a LogProgress, else just iterate the data
        if self.rank == 0:
            logprog = LogProgress(self.logger, data_loader, updates=self.num_prints, name=name)
        else:
            logprog = data_loader
        
        for i, data in enumerate(logprog):
            # Unpack data
            noisy, clean = data
            
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            clean_hat = self.model(noisy)
            
            # Compute loss
            loss_all, loss_dict = self.loss(clean_hat, clean)
            
            # For distributed training, we do all_reduce
            # Here, we only do all_reduce for training, not for validation
            if self.is_distributed and not valid:
                dist.all_reduce(loss_all)
                loss_all = loss_all / self.world_size
            
            total_loss += loss_all.item()

            if not valid:
                # Training step
                self.optim.zero_grad()
                
                if self.rank == 0:
                    # Log current losses in the progress bar
                    for j, (key, value) in enumerate(loss_dict.items()):
                        if j == 0:
                            logprog.update(**{f"{key}_Loss": format(value, "4.5f")})
                        else:
                            logprog.append(**{f"{key}_Loss": format(value, "4.5f")})
                        self.writer.add_scalar(f"train/{key}_Loss", value, epoch * len(data_loader) + i)
                    self.writer.add_scalar("train/Loss", loss_all.item(), epoch * len(data_loader) + i)
                
                # Backpropagation
                loss_all.backward()
                
                # Gradient clipping
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
                    
                # Optimizer step
                self.optim.step()
                
                if self.discriminator is not None:
                    disc_loss = self.loss.forward_disc_loss(clean_hat.detach(), clean)
                    if disc_loss is not None:
                        self.optim_disc.zero_grad()
                        disc_loss.backward()
                        self.optim_disc.step()
                        if self.rank == 0:
                            logprog.append(**{f'Discriminator_Loss': format(disc_loss, "4.5f")})
                            self.writer.add_scalar(f"train/Discriminator_Loss", disc_loss, epoch * len(data_loader) + i)
                
            else:
                # Validation step (rank=0 logs)
                if self.rank == 0:
                    for j, (key, value) in enumerate(loss_dict.items()):
                        if j == 0:
                            logprog.update(**{f"{key}_Loss": format(value, "4.5f")})
                        else:
                            logprog.append(**{f"{key}_Loss": format(value, "4.5f")})
                    self.writer.add_scalar("valid/Loss", loss_all.item(), epoch * len(data_loader) + i)
        
        if self.scheduler is not None and not valid:
            self.scheduler.step()
        if self.scheduler_disc is not None and not valid:
            self.scheduler_disc.step()
        
        # Return the average loss over the entire epoch
        return total_loss / len(data_loader)
    

    def _validate_with_pesq(self, epoch):
        """Run validation and compute PESQ."""
        data_loader = self.va_loader
        label = "Valid with PESQ"
        name = label + f" | Epoch {epoch + 1}"
        
        logprog = LogProgress(self.logger, data_loader, updates=self.num_prints, name=name)
        
        clean_list = []
        clean_hat_list = []
        with torch.no_grad():
            for data in logprog:
                noisy, clean = data
                
                clean_hat = self.model(noisy.to(self.device))
                clean_list.append(clean.squeeze().detach().cpu().numpy())
                clean_hat_list.append(clean_hat.squeeze().detach().cpu().numpy())
                
        pesq_score = batch_pesq(
            clean_list, 
            clean_hat_list, 
            workers=8,
            normalize=False
        )

        pesq_score = -pesq_score.mean().item()

        self.logger.info(f"PESQ Score: {pesq_score:.5f}")
        self.writer.add_scalar("valid/PESQ", pesq_score, epoch * len(data_loader))
        
        return pesq_score
        
        
                
        
        
                
        
        
        
        
        