import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import random
from pcs400 import cal_pcs

def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True, stack_dim=-1):

    hann_window = torch.hann_window(win_size).to(y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    stft_spec = torch.view_as_real(stft_spec)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1)+(1e-9))
    pha = torch.atan2(stft_spec[:, :, :, 1]+(1e-10), stft_spec[:, :, :, 0]+(1e-5))
    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag*torch.cos(pha), mag*torch.sin(pha)), dim=stack_dim)

    return mag, pha, com

def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    # Magnitude Decompression
    mag = torch.pow(mag, (1.0/compress_factor))
    com = torch.complex(mag*torch.cos(pha), mag*torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    wav = torch.istft(com, n_fft=n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=center)

    return wav

def complex_to_mag_pha(com, stack_dim=-1):
    real, imag = com.chunk(2, dim=stack_dim)
    mag = torch.sqrt(real**2 + imag**2).squeeze(stack_dim)
    pha = torch.atan2(imag, real).squeeze(stack_dim)
    return mag, pha

def mag_pha_to_complex(mag, pha, stack_dim=-1):
    real = mag * torch.cos(pha)
    imag = mag * torch.sin(pha)
    com = torch.stack((real, imag), dim=stack_dim)
    return com

class VoiceBankDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 datapair_list,
                 sampling_rate=16_000,
                 segment=32000,
                 fft_len=400,
                 hop_size=100,
                 win_size=400,
                 compress_factor=1.0,
                 with_id=False,
                 type="train",
                 use_pcs400=False
                 ):
        # Initialize variables with constructor arguments
        self.datapair_list = datapair_list
        self.sampling_rate = sampling_rate
        self.segment = segment
        self.fft_len = fft_len
        self.hop_size = hop_size
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.with_id = with_id
        assert type in ["train", "valid", "test"]
        self.type = type
        self.use_pcs400 = use_pcs400
        # Prepare lists for noisy and clean audio arrays
        self.audio_pairs = []
        for item in self.datapair_list:
            noisy = item["noisy"]['array'].astype('float32')
            clean = item["clean"]['array'].astype('float32')
            id = item["id"]
            if self.use_pcs400:
                clean = cal_pcs(clean)
            norm_factor = np.sqrt(noisy.shape[-1] / np.sum(noisy ** 2.0))
            noisy = noisy * norm_factor
            clean = clean * norm_factor
            
            self.audio_pairs.append((noisy, clean, id))
                
    def __len__(self):
        return len(self.audio_pairs)

    def __getitem__(self, index):
        # Output shape: [B, 1, F, T]
        
        noisy, clean, id = self.audio_pairs[index]
        
        noisy = torch.FloatTensor(noisy).unsqueeze(0)
        clean = torch.FloatTensor(clean).unsqueeze(0)
        
        length = noisy.size(1)
        assert length == clean.size(1)
        
        if self.type == "train":
            if clean.size(1) >= self.segment:
                max_audio_start = clean.size(1) - self.segment
                rand_num = random.random()
                
                if rand_num < 0.01:
                    audio_start = 0
                elif rand_num < 0.02:
                    audio_start = max_audio_start
                else:
                    audio_start = random.randint(0, max_audio_start)
                    
                clean = clean[:, audio_start:audio_start+self.segment]
                noisy = noisy[:, audio_start:audio_start+self.segment]
                
            else:
                clean = torch.nn.functional.pad(clean, (0, self.segment - clean.size(1)), mode='constant')
                noisy = torch.nn.functional.pad(noisy, (0, self.segment - noisy.size(1)), mode='constant')
            
        elif self.type == "valid":
            if length >= self.segment:
                num_segments = length // self.segment
                last_segment_size = length % self.segment
                segments = [noisy[:, i*self.segment:(i+1)*self.segment] for i in range(num_segments)]
                if last_segment_size > 0:
                    segments.append(noisy[:, -self.segment:])
                segments = torch.cat(segments, dim=0)
            else:
                segments = torch.FloatTensor(F.pad(noisy, (0, self.segment - noisy.size(1)), mode='constant', value=0)) 

        # elif the type is test, we don't need to do anything         
        clean_mag, clean_pha, clean_com = mag_pha_stft(clean,
                                                        n_fft=self.fft_len,
                                                        hop_size=self.hop_size,
                                                        win_size=self.win_size,
                                                        compress_factor=self.compress_factor,
                                                        center=True)
        noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy if not self.type == "valid" else segments, 
                                                        n_fft=self.fft_len,
                                                        hop_size=self.hop_size,
                                                        win_size=self.win_size,
                                                        compress_factor=self.compress_factor,
                                                        center=True)
        
        if self.type in ["train", "test"]:
            clean, clean_mag, clean_pha, clean_com = clean.squeeze(), clean_mag.squeeze(), clean_pha.squeeze(), clean_com.squeeze()
            noisy, noisy_mag, noisy_pha, noisy_com = noisy.squeeze(), noisy_mag.squeeze(), noisy_pha.squeeze(), noisy_com.squeeze()

        input_clean = {
            "wav": clean,
            "magnitude": clean_mag,
            "phase": clean_pha,
            "complex": clean_com,
            "length": length,
        }
        input_noisy = {
            "wav": noisy,
            "magnitude": noisy_mag,
            "phase": noisy_pha,
            "complex": noisy_com,
            "length": length,
        }

        if self.with_id:
            return input_noisy, input_clean, id
        else:
            return input_noisy, input_clean 


class StepSampler(torch.utils.data.Sampler):
    def __init__(self, length, step):
        # Save the total length and sampling step
        self.step = step
        self.length = length
        
    def __iter__(self):
        # Return indices at intervals of step
        return iter(range(0, self.length, self.step))
    
    def __len__(self):
        # Length is how many indices we can produce based on the step
        return self.length // self.step