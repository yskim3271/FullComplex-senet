import os
import sys
import logging
import psutil
import importlib
import hydra
import random
import torch
import torch.distributed as dist
import numpy as np
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from datasets import load_dataset

from data import VoiceBankDataset, StepSampler, validation_collate_fn
from solver import Solver

def kill_child_processes():
    """kill child processes"""
    current_process = psutil.Process(os.getpid())
    children = current_process.children(recursive=True)
    for child in children:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

def setup_logger(name, rank=None):
    """Set up logger"""
    if rank == 0:
        hydra_conf = OmegaConf.load(".hydra/hydra.yaml")
        logging.config.dictConfig(OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True))
    else:
        logging.basicConfig(level=logging.ERROR)
        
    return logging.getLogger(name)

def setup_distributed(rank, world_size, args):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = str(args.ddp.master_addr)
    os.environ['MASTER_PORT'] = str(args.ddp.master_port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up distributed training environment"""
    if dist.is_initialized():
        dist.destroy_process_group()

def parse_list(dir, file):
    return [os.path.join(dir, line.strip()) for line in open(file, "r")]

def run(rank, world_size, args):
        
    # Create and initialize logger
    logger = setup_logger("train", rank)

    # Set up distributed training environment
    if world_size > 1:
        setup_distributed(rank, world_size, args)
    
    if rank == 0:
        logger.info(f"Training with {world_size} GPUs")
    
    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device(f'cuda:{rank}')
    
    model_args = args.model
    model_name = model_args.model_name
    
    # import model library
    module = importlib.import_module("models." + model_name)
    model_class = getattr(module, model_name)
    
    model = model_class(**model_args.param)
    model = model.to(args.device)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    # Load dataset
    if rank == 0:
        dataset = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
    if world_size > 1:
        dist.barrier()
    if rank != 0:
        dataset = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
    
    trainset = dataset['train']
    testset = dataset['test']
                
    # Set up dataset and dataloader
    tr_dataset = VoiceBankDataset(
        datapair_list=trainset,
        sampling_rate=args.sampling_rate,
        segment=args.segment, 
        stride=args.stride, 
    )
    
    # Set up distributed sampler
    tr_sampler = DistributedSampler(tr_dataset) if world_size > 1 else None
    tr_loader = DataLoader(
        dataset=tr_dataset,
        batch_size=args.batch_size,
        sampler=tr_sampler,
        shuffle=(tr_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True
    )
        
    # Set up validation and test dataset and dataloader
    va_dataset = VoiceBankDataset(
        datapair_list=testset,
        sampling_rate=args.sampling_rate
    )
    va_sampler = DistributedSampler(va_dataset, shuffle=False) if world_size > 1 else None
    va_loader = DataLoader(
        dataset=va_dataset, 
        batch_size=1,
        sampler=va_sampler,
        num_workers=args.num_workers,
        collate_fn=validation_collate_fn,
        pin_memory=True
    )
    
    ev_dataset = VoiceBankDataset(
        datapair_list=testset,
        sampling_rate=args.sampling_rate,
        with_id=True
    )
    
    ev_loader = DataLoader(
        dataset=ev_dataset, 
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    tt_loader = DataLoader(
        dataset=ev_dataset, 
        batch_size=1,
        sampler=StepSampler(len(ev_dataset), 100),
        num_workers=args.num_workers,
        pin_memory=True
    )
        
    
    dataloader = {
        "tr_loader": tr_loader,
        "va_loader": va_loader,
        "ev_loader": ev_loader,
        "tt_loader": tt_loader,
        "tr_sampler": tr_sampler,
    }
    
    # optimizer
    if args.optim == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.betas)
    elif args.optim == "adamW" or args.optim == "adamw":
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=args.betas)
    
    # Solver
    solver = Solver(
        data=dataloader,
        model=model,
        optim=optim,
        args=args,
        logger=logger,
        rank=rank,
        world_size=world_size,
        device=device
    )
    solver.train()
    
    cleanup()
        

def _main(args):
    global __file__

    logger = setup_logger("main")
            
    __file__ = hydra.utils.to_absolute_path(__file__)
    
    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)
    
    world_size = torch.cuda.device_count()
    
    if world_size > 1:
        import torch.multiprocessing as mp
        try:
            mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
        except KeyboardInterrupt:
            logger.info("Training stopped by user")
            kill_child_processes()
        except Exception as e:
            logger.exception(f"Error occurred in spawn: {str(e)}")
            kill_child_processes()
    else:
        run(0, 1, args)

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(args):
    logger = setup_logger("main")
    try:
        _main(args)
    except KeyboardInterrupt:
        logger.info("Training stopped by user")
        kill_child_processes()
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Error occurred in main: {str(e)}")
        kill_child_processes()
        sys.exit(1)

if __name__ == "__main__":
    main()