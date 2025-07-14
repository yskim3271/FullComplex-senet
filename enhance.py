

import os
import torch
import torchaudio

from matplotlib import pyplot as plt
from utils import LogProgress, mel_spectrogram
from data import mag_pha_istft

def save_wavs(wavs_dict, filepath, sr=16_000):
    for i, (key, wav) in enumerate(wavs_dict.items()):
        torchaudio.save(filepath + f"_{key}.wav", wav, sr)
        
def save_mels(wavs_dict, filepath):
    num_mels = len(wavs_dict)
    figure, axes = plt.subplots(num_mels, 1, figsize=(10, 10))
    figure.set_tight_layout(True)
    figure.suptitle(filepath)
    
    for i, (key, wav) in enumerate(wavs_dict.items()):
        mel = mel_spectrogram(wav, device='cpu', sampling_rate=16_000)
        axes[i].imshow(mel.squeeze().numpy(), aspect='auto', origin='lower')
        axes[i].set_title(key)
    
    figure.savefig(filepath)
    plt.close(figure)

def write(wav, filename, sr=16_000):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr)

def enhance(args, model, data_loader, logger, epoch=None, local_out_dir="samples"):
    model.eval()
        
    suffix = f"_epoch{epoch+1}" if epoch is not None else ""
    
    iterator = LogProgress(logger, data_loader, name=f"Enhance")
    outdir_mels= os.path.join(local_out_dir, f"mels" + suffix)
    outdir_wavs= os.path.join(local_out_dir, f"wavs" + suffix)
    os.makedirs(outdir_mels, exist_ok=True)
    os.makedirs(outdir_wavs, exist_ok=True)
    
    with torch.no_grad():
        for data in iterator:
            noisy, clean, id = data
            noisy = {key: value.to(args.device) for key, value in noisy.items()}

            clean_hat = model(noisy)
            
            clean = clean["wav"].squeeze(1).detach().cpu()
            noisy = noisy["wav"].squeeze(1).detach().cpu()
            clean_hat = mag_pha_istft(clean_hat["magnitude"], clean_hat["phase"], args.fft_size, args.hop_size, args.win_length, args.compress_factor)
            clean_hat = clean_hat.squeeze(1).detach().cpu()
                        
            wavs_dict = {
                "noisy": noisy,
                "clean": clean,
                "clean_hat": clean_hat,
            }
            
            save_wavs(wavs_dict, os.path.join(outdir_wavs, id[0]))
            save_mels(wavs_dict, os.path.join(outdir_mels, id[0]))
