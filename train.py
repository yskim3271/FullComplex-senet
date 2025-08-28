import os
import sys
import logging
import psutil
import importlib
import hydra
import random
import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import load_dataset
from models.discriminator import MetricGAN_Discriminator
import shutil
from data import VoiceBankDataset, StepSampler
from solver import Solver

torch.backends.cudnn.benchmark = True

def kill_child_processes():
    """kill child processes"""
    current_process = psutil.Process(os.getpid())
    children = current_process.children(recursive=True)
    for child in children:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

def setup_logger(name):
    """Set up logger"""
    hydra_conf = OmegaConf.load(".hydra/hydra.yaml")
    logging.config.dictConfig(OmegaConf.to_container(hydra_conf.hydra.job_logging, resolve=True))
    return logging.getLogger(name)


def run(args):
        
    # Create and initialize logger
    logger = setup_logger("train")

    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model_args = args.model
    model_lib = model_args.model_lib
    model_class = model_args.model_class
    
    # import model library
    module = importlib.import_module("models." + model_lib)
    model_class = getattr(module, model_class)
    
    model = model_class(**model_args.param)
    model = model.to(device)

    # Calculate and log the total number of parameters and model size
    logger.info(f"Selected model: {model_lib}.{model_class}")
    total_params = sum(p.numel() for p in model.parameters())
    model_params = (total_params) / 1000000
    logger.info(f"Model's parameters: {model_params:.2f} M")

    if args.save_code:
        # Use hydra.utils.to_absolute_path to get the correct path
        project_root = os.path.dirname(hydra.utils.to_absolute_path(__file__))
        src = os.path.join(project_root, "models", f"{model_lib}.py")
        dest = f"./{model_lib}.py"
        
        if os.path.exists(src):
            shutil.copy2(src, dest)
            logger.info(f"Copied {src} to {dest}")
        else:
            logger.warning(f"Model file not found: {src}")

    if args.optim == "adam":
        optim_class = torch.optim.Adam
    elif args.optim == "adamW" or args.optim == "adamw":
        optim_class = torch.optim.AdamW

    discriminator = MetricGAN_Discriminator().to(device)

    # optimizer
    optim = optim_class(model.parameters(), lr=args.lr, betas=args.betas)
    optim_disc = optim_class(discriminator.parameters(), lr=args.lr, betas=args.betas)
    
    # scheduler
    scheduler = None
    scheduler_disc = None

    if args.lr_decay is not None:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.lr_decay, last_epoch=-1)
        scheduler_disc = torch.optim.lr_scheduler.ExponentialLR(optim_disc, gamma=args.lr_decay, last_epoch=-1)

    # Load dataset from Huggingface
    if args.use_huggingface:
        dataset = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
        trainset = dataset['train']
        validset = dataset['test']
    else:
        # Convert relative paths to absolute paths for train and valid file lists
        project_root = os.path.dirname(hydra.utils.to_absolute_path(__file__))
        trainset = os.path.join(project_root, args.input_train_file_list)
        validset = os.path.join(project_root, args.input_valid_file_list)
        if trainset is None or validset is None:
            logger.error("input_train_file_list and input_valid_file_list are not set")
            sys.exit(1)
    
    
    # Set up dataset and dataloader
    tr_dataset = VoiceBankDataset(trainset, use_pcs400=args.use_pcs400, use_huggingface=args.use_huggingface, segment=args.segment)
    
    # Sampler (not used in single GPU mode)
    tr_loader = DataLoader(
        dataset=tr_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Set up validation and test dataset and dataloader
    va_dataset = VoiceBankDataset(
        validset, use_pcs400=False, use_huggingface=args.use_huggingface
    )
    
    va_loader = DataLoader(
        dataset=va_dataset, 
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    ev_loader = DataLoader(
        dataset=va_dataset, 
        batch_size=1,
        sampler=StepSampler(len(va_dataset), 100),
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    dataloader = {
        "tr_loader": tr_loader,
        "va_loader": va_loader,
        "ev_loader": ev_loader,
    }

    # Solver
    solver = Solver(
        data=dataloader,
        model=model,
        discriminator=discriminator,
        optim=optim,
        optim_disc=optim_disc,
        scheduler=scheduler,
        scheduler_disc=scheduler_disc,
        args=args,
        logger=logger,
        device=device
    )
    solver.train()
    sys.exit(0)

def _main(args):
    global __file__

    logger = setup_logger("main")
            
    __file__ = hydra.utils.to_absolute_path(__file__)
    
    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)

    run(args)

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