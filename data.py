import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import random
from pcs400 import cal_pcs
import librosa

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

def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def random_sample(noisy, clean, segment):
    length = noisy.shape[-1]
    if length >= segment:
        max_audio_start = length - segment
        rand_num = random.random()
        if rand_num < 0.01:
            audio_start = 0
        elif rand_num < 0.02:
            audio_start = max_audio_start
        else:
            audio_start = random.randint(0, max_audio_start)
        noisy = noisy[audio_start: audio_start + segment]
        clean = clean[audio_start: audio_start + segment]
    else:
        noisy = F.pad(noisy, (0, segment - length), mode='constant')
        clean = F.pad(clean, (0, segment - length), mode='constant')
    return noisy, clean

def segment_sample(noisy, clean, segment):
    length = noisy.shape[-1]
    if length >= segment:
        num_segments = length // segment
        last_segment_size = length % segment
        noisy_segments = [noisy[i*segment: (i+1)*segment] for i in range(num_segments)]
        clean_segments = [clean[i*segment: (i+1)*segment] for i in range(num_segments)]
        if last_segment_size > 0:
            noisy_segments.append(noisy[num_segments*segment:])
            clean_segments.append(clean[num_segments*segment:])
        noisy = torch.cat(noisy_segments, dim=0)
        clean = torch.cat(clean_segments, dim=0)

    else:
        noisy = F.pad(noisy, (0, segment - length), mode='constant')
        clean = F.pad(clean, (0, segment - length), mode='constant')
    return noisy, clean


class VoiceBankDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 dataset,
                 use_pcs400=False,
                 use_huggingface=False,
                 segment=None
                 ):
        self.use_pcs400 = use_pcs400
        self.sampling_rate = 16000
        self.segment = segment
        self.audio_pairs = []

        if use_huggingface:
            for item in dataset:
                noisy = item["noisy"]['array'].astype('float32')
                clean = item["clean"]['array'].astype('float32')
                id = item["id"]
                if self.use_pcs400:
                    clean = cal_pcs(clean)
                norm_factor = np.sqrt(noisy.shape[-1] / np.sum(noisy ** 2.0))
                noisy = noisy * norm_factor
                clean = clean * norm_factor
            
                self.audio_pairs.append((noisy, clean, id))

        else:
            with open(dataset, 'r', encoding='utf-8') as fi:
                file_list = [x.split('|') for x in fi.read().split('\n') if len(x) > 0]
            for item in file_list:
                noisy, _ = librosa.load(item[1], sr=self.sampling_rate)
                clean, _ = librosa.load(item[2], sr=self.sampling_rate)
                id = item[0]
                if self.use_pcs400:
                    clean = cal_pcs(clean)
                norm_factor = np.sqrt(noisy.shape[-1] / np.sum(noisy ** 2.0))
                noisy = noisy * norm_factor
                clean = clean * norm_factor

                self.audio_pairs.append((noisy, clean, id))
            
    def __len__(self):
        return len(self.audio_pairs)

    def __getitem__(self, index):        
        noisy, clean, id = self.audio_pairs[index]

        noisy = torch.FloatTensor(noisy)
        clean = torch.FloatTensor(clean)

        if self.segment is not None:
            noisy, clean = random_sample(noisy, clean, self.segment)

        return noisy, clean, id
        


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