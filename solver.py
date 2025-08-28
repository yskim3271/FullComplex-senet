import os
import time
import shutil
import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from evaluate import evaluate
from data import mag_pha_istft, mag_pha_stft, segment_sample
from utils import copy_state, swap_state, anti_wrapping_function, batch_pesq

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
        device=None
    ):
        # Dataloaders and samplers
        self.tr_loader = data['tr_loader']      # Training DataLoader
        self.va_loader = data['va_loader']      # Validation DataLoader
        self.ev_loader = data['ev_loader']      # Evaluation DataLoader
        
        self.model = model
        self.discriminator = discriminator
        self.optim = optim
        self.optim_disc = optim_disc
        self.scheduler = scheduler
        self.scheduler_disc = scheduler_disc
        self.logger = logger

        # dataset
        self.segment = args.segment
        self.n_fft = args.n_fft
        self.hop_size = args.hop_size
        self.win_size = args.win_size
        self.compress_factor = args.compress_factor
        self.stft_args = {
            "n_fft": args.n_fft,
            "hop_size": args.hop_size,
            "win_size": args.win_size,
            "compress_factor": args.compress_factor
        }

        # Basic config
        self.device = device or torch.device(args.device)
        
        self.epochs = args.epochs
        
        self.validation_interval = args.validation_interval   # interval for validation
        self.evaluation_interval = args.evaluation_interval   # interval for evaluation
        self.summary_interval = args.summary_interval   # interval for summary
        self.best_models_num = args.best_models_num
        self.log_interval = args.log_interval
            
        # Checkpoint settings
        self.continue_from = args.continue_from
        
        self.writer = None
        self.best_models = []
        self.log_dir = args.log_dir
        self.samples_dir = args.samples_dir
        self.args = args

        self.epoch_start = 0
        
        # Initialize or resume (checkpoint loading)
        self._reset()

    def _save_model_checkpoint(self, steps, state_dict):
        """ Save model checkpoint. """        
        # Create a package dict
        package_model = {}
        package_model['model'] = copy_state(state_dict)
        
        # Write to a temporary file first
        tmp_path = "model.tmp"
        torch.save(package_model, tmp_path)
        os.rename(tmp_path, f"model_{steps}.th")
    
    def _save_states_checkpoint(self, epoch):
        """ Save states checkpoint. """
        package = {}
        package['model'] = copy_state(self.model.state_dict())
        package['best_models'] = self.best_models
        package['discriminator'] = copy_state(self.discriminator.state_dict())
        package['optimizer'] = self.optim.state_dict()
        package['optimizer_disc'] = self.optim_disc.state_dict()
        package['scheduler'] = self.scheduler.state_dict() if self.scheduler is not None else None
        package['scheduler_disc'] = self.scheduler_disc.state_dict() if self.scheduler_disc is not None else None
        package['args'] = self.args
        package['epoch'] = epoch
        # Write to a temporary file first
        tmp_path = "states.tmp"
        torch.save(package, tmp_path)
        os.rename(tmp_path, f"states.th")

    def _update_best_models(self, steps, valid_pesq_value):
        """Maintain top-k models by validation PESQ. """
        entry = {
            "steps": steps,
            "valid_pesq_value": valid_pesq_value,
            "model": copy_state(self.model.state_dict())
        }
        self.best_models.append(entry)
        # Keep only top k by PESQ in descending order
        self.best_models.sort(key=lambda x: x["valid_pesq_value"], reverse=True)
        if len(self.best_models) > self.best_models_num:
            self.best_models = self.best_models[:self.best_models_num]

    def _reset(self):
        """Load checkpoint if 'continue_from' is specified, or create a fresh writer if not."""
        if self.continue_from is not None:
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
                        
            # loads the checkpoint file from disk
            ckpt_path = os.path.join(self.continue_from, 'states.th')
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found.")
            self.logger.info(f"Loading checkpoint from {ckpt_path}")
            package = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                            
            model_state = package['model']
            model_disc_state = package.get('discriminator', None)
            optim_state = package['optimizer']
            optim_disc_state = package.get('optimizer_disc', None)
            scheduler_state = package.get('scheduler', None)
            scheduler_disc_state = package.get('scheduler_disc', None)
            self.best_models = package.get('best_models', [])
            self.epoch_start = package.get('epoch', 0)
                    
            self.model.load_state_dict(model_state)
            self.optim.load_state_dict(optim_state)
            
            self.discriminator.load_state_dict(model_disc_state)
            self.optim_disc.load_state_dict(optim_disc_state)
            
            if self.scheduler is not None and scheduler_state is not None:
                self.scheduler.load_state_dict(scheduler_state)
            if self.scheduler_disc is not None and scheduler_disc_state is not None:
                self.scheduler_disc.load_state_dict(scheduler_disc_state)
            
        else:
            # If there's no checkpoint to resume from, just create a fresh SummaryWriter
            self.writer = SummaryWriter(log_dir=self.log_dir)


    def train(self):
        self.logger.info("Training for %d epochs", self.epochs)

        if self.epoch_start != 0:
            self.logger.info("Resuming training from epoch %d", self.epoch_start + 1)
        
        for epoch in range(self.epoch_start, self.epochs):
            self.logger.info(f"Train | Epoch {epoch + 1} | Learning Rate {self.optim.param_groups[0]['lr']:.6f}")
            for i, (noisy, clean, _) in enumerate(self.tr_loader):
                steps = epoch * len(self.tr_loader) + i + 1
                start = time.time()
                
                loss_dict = self._run_one_step(noisy, clean)

                if steps % self.log_interval == 0:
                    info = " | ".join(f"{k.capitalize()} {v:.4f}" for k, v in loss_dict.items())
                    self.logger.info(f"Train | Epoch {epoch + 1} | Steps {steps} | {1/(time.time() - start):.1f} iters/s | {info}")
                
                if steps % self.summary_interval == 0:
                    for key, value in loss_dict.items():
                        self.writer.add_scalar(f"Train/{key}", value, steps)
                    
                if steps % self.validation_interval == 0:
                    val_pesq_score = self._run_validation(epoch, steps)
                    self._update_best_models(steps, val_pesq_score)

            self._save_states_checkpoint(epoch)
            
            if self.scheduler is not None:
                self.scheduler.step()
            if self.scheduler_disc is not None:
                self.scheduler_disc.step()
            
            if (epoch % self.evaluation_interval == 0 or epoch == self.epochs - 1) and epoch > 0:
                self.logger.info("-" * 70)
                self.logger.info("Evaluation")
                for model in self.best_models:
                    if model.get('metrics') is None:
                        with swap_state(self.model, model['model']):
                            metrics = evaluate(self.model, self.va_loader, self.device, model['steps'], self.stft_args)
                        model['metrics'] = metrics

                        self._save_model_checkpoint(model['steps'], model['model'])
                        for key, value in metrics.items():
                            self.writer.add_scalar(f"Evaluation/{key}", value, model['steps'])
                    else:
                        metrics = model['metrics']
                    
                    info = f"Model Step: {model['steps']} | Valid PESQ: {model['valid_pesq_value']:.4f}"
                    for key, value in metrics.items():
                        info += f" | {key.title()} {value:.4f}"
                    self.logger.info(info)
                    
                self.logger.info("-" * 70)

        self.logger.info("-" * 70)
        self.logger.info("Training Completed")
        self.logger.info(f"Best Model | Steps: {self.best_models[0]['steps']}, PESQ: {self.best_models[0]['metrics']['pesq']:.4f}")
        self.logger.info("-" * 70)
        self.writer.close()

    def _run_one_step(self, noisy, clean):

        self.model.train()
        self.discriminator.train()

        noisy = noisy.to(self.device)
        clean = clean.to(self.device)
        one_labels = torch.ones(noisy.shape[0]).to(self.device)

        noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy, **self.stft_args)
        clean_mag, clean_pha, clean_com = mag_pha_stft(clean, **self.stft_args)

        clean_mag_hat, clean_pha_hat, clean_com_hat = self.model(noisy_com)

        clean_hat = mag_pha_istft(clean_mag_hat, clean_pha_hat, **self.stft_args)
        clean_mag_hat_con, clean_pha_hat_con, clean_com_hat_con = mag_pha_stft(clean_hat, **self.stft_args)

        clean_list, clean_list_hat = list(clean.cpu().numpy()), list(clean_hat.detach().cpu().numpy())
        batch_pesq_score = batch_pesq(clean_list, clean_list_hat)

        self.optim_disc.zero_grad()

        metric_r = self.discriminator(clean_mag.unsqueeze(1), clean_mag.unsqueeze(1))
        metric_g = self.discriminator(clean_mag.unsqueeze(1), clean_mag_hat_con.detach().unsqueeze(1))
        

        loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())

        if batch_pesq_score is not None:
            loss_disc_g = F.mse_loss(batch_pesq_score.to(self.device), metric_g.flatten())
        else:
            loss_disc_g = 0

        loss_disc = loss_disc_r + loss_disc_g

        loss_disc.backward()
        self.optim_disc.step()

        self.optim.zero_grad()

        loss_complex = F.mse_loss(clean_com, clean_com_hat) * 2
        loss_consistency = F.mse_loss(clean_com_hat, clean_com_hat_con)

        metric_g = self.discriminator(clean_mag.unsqueeze(1), clean_mag_hat_con.unsqueeze(1))
        loss_metric = F.mse_loss(metric_g.flatten(), one_labels)

        loss_gen = loss_metric * 0.05 + loss_complex * 2 + loss_consistency * 0.1

        loss_gen.backward()

        self.optim.step()

        loss_dict = {
            "Metric_Loss": loss_metric,
            "Complex_Loss": loss_complex,
            "Consistency_Loss": loss_consistency,
            "Disc_Loss": loss_disc,
            "Gen_Loss": loss_gen
        }

        return loss_dict


    def _run_validation(self, epoch, steps):
        self.model.eval()
        self.discriminator.eval()

        val_err_complex = 0
        val_err_mag = 0
        val_err_phase = 0
        clean_list, clean_hat_list = [], []
        
        with torch.no_grad():
            for data in self.va_loader:
                noisy, clean, _ = data
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                
                length = noisy.shape[-1]
                noisy, clean_seg = segment_sample(noisy, clean, self.segment)

                clean_mag, clean_pha, clean_com = mag_pha_stft(clean_seg, **self.stft_args)
                noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy, **self.stft_args)  
                
                clean_mag_hat, clean_pha_hat, clean_com_hat = self.model(noisy_com)
                
                clean_hat = mag_pha_istft(clean_mag_hat, clean_pha_hat, **self.stft_args)
                
                num_segments, segment_size = clean_hat.shape
                if length <= segment_size:
                    clean_hat = clean_hat[:, :length]
                else:
                    clean_hat = clean_hat.view(1, -1)
                    clean_hat = torch.cat([
                        clean_hat[:, :(num_segments - 2) * segment_size + length % segment_size],
                        clean_hat[:, -segment_size:]
                    ], dim=1)
                
                
                clean_list.append(clean.squeeze().detach().cpu().numpy())
                clean_hat_list.append(clean_hat.squeeze().detach().cpu().numpy())

                val_err_complex += F.l1_loss(clean_com, clean_com_hat)
                val_err_mag += F.l1_loss(clean_mag, clean_mag_hat)
                val_err_phase += torch.mean(anti_wrapping_function(clean_pha - clean_pha_hat))

    
        val_err_complex /= len(self.va_loader)
        val_err_mag /= len(self.va_loader)
        val_err_phase /= len(self.va_loader)
        val_pesq_score = batch_pesq(clean_list, clean_hat_list, workers=15, normalize=False).mean().item()
        if val_pesq_score is None:
            val_pesq_score = 0

        self.logger.info("-" * 70)
        self.logger.info(
            f"Validation | Epoch {epoch + 1} | Steps {steps} | Complex Diff {val_err_complex:.5f} | Magnitude Diff {val_err_mag:.5f} | Phase Diff {val_err_phase:.5f} | Valid PESQ {val_pesq_score:.5f}"
        )
        self.logger.info("-" * 70)
        self.writer.add_scalar("Validation/Complex_Loss", val_err_complex, steps)
        self.writer.add_scalar("Validation/Magnitude_Loss", val_err_mag, steps)    
        self.writer.add_scalar("Validation/Phase_Loss", val_err_phase, steps)
        self.writer.add_scalar("Validation/Validation_PESQ_Score", val_pesq_score, steps)

        return val_pesq_score
            


                
        
        
                
        
        
        
        
        