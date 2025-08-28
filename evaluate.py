# Copyright (c) POSTECH, and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: yunsik kim
import torch
import numpy as np
from compute_metrics import compute_metrics
from data import mag_pha_stft, mag_pha_istft    

def evaluate(model, data_loader, device, steps=None, stft_args=None):
    
    model.eval()
    prefix = f"Steps {steps}, " if steps is not None else ""

    results = []
    with torch.no_grad():
        for data in data_loader:
            noisy, clean, _ = data
            noisy = noisy.to(device)
            clean = clean.to(device)
            noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy, **stft_args)

            clean_mag_hat, clean_pha_hat, clean_com_hat = model(noisy_com)

            clean_hat = mag_pha_istft(clean_mag_hat, clean_pha_hat, **stft_args)

            clean = clean.squeeze().detach().cpu().numpy()
            clean_hat = clean_hat.squeeze().detach().cpu().numpy()
            if len(clean) != len(clean_hat):
                length = min(len(clean), len(clean_hat))
                clean = clean[0: length]
                clean_hat = clean_hat[0: length]
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
    return metrics