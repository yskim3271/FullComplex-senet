# Copyright (c) POSTECH, and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: yunsik kim
import torch
import numpy as np
from compute_metrics import compute_metrics
from data import mag_pha_istft
from utils import bold, LogProgress
    

def evaluate(args, model, data_loader, logger, epoch=None):
    
    prefix = f"Epoch {epoch+1}, " if epoch is not None else ""
    metrics = {}
    model.eval()
    iterator = LogProgress(logger, data_loader, name=f"Evaluate")

    results = []
    with torch.no_grad():
        for data in iterator:
            noisy, clean, id = data
            noisy = {key: value.to(args.device) for key, value in noisy.items()}

            clean_hat = model(noisy)
            clean_hat = mag_pha_istft(clean_hat["magnitude"], clean_hat["phase"], args.fft_size, args.hop_size, args.win_length, args.compress_factor)
            clean_hat = clean_hat.squeeze().detach().cpu().numpy()
            clean = clean["wav"].squeeze().detach().cpu().numpy()
            results.append(compute_metrics(clean, clean_hat))
    
    pesq, csig, cbak, covl, segSNR, stoi = np.mean(results, axis=0)
    metrics = {
        "pesq": pesq,
        "stoi": stoi,
        "csig": csig,
        "cbak": cbak,
        "covl": covl,
        "segSNR": segSNR
    }
    logger.info(bold(f"{prefix}Performance: PESQ={pesq:.4f}, STOI={stoi:.4f}, CSIG={csig:.4f}, CBAK={cbak:.4f}, COVL={covl:.4f}, SEGSNR={segSNR:.4f}"))
   
    return metrics