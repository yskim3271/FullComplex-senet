import random
import torch
import torch.utils.data
import torchaudio
import math
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from scipy import signal

class VoiceBankDataset:
    def __init__(self, 
                 datapair_list,
                 sampling_rate=16_000,
                 segment=None, 
                 stride=None, 
                 shift=None, 
                 with_id=False,
                 ):
        # Initialize variables with constructor arguments
        self.datapair_list = datapair_list
        self.sampling_rate = sampling_rate
        self.segment = segment
        self.stride = stride
        self.shift = shift
        self.with_id = with_id
        
        # Prepare lists for noisy and clean audio arrays
        noisy_list, clean_list = [], []
        for item in self.datapair_list:
            noisy = item["noisy"]['array'].astype('float32')
            clean = item["clean"]['array'].astype('float32')
            id = item["id"]
            length = noisy.shape[-1]
            noisy_list.append((noisy, id, length))
            clean_list.append((clean, id, length))
        
        # Create Audioset objects for noisy and clean
        self.noisy_set = Audioset(wavs=noisy_list, segment=segment, stride=stride, with_id=with_id)
        self.clean_set = Audioset(wavs=clean_list, segment=segment, stride=stride, with_id=with_id)

    def __len__(self):
        # The length of the dataset is the number of noisy_set samples
        return len(self.noisy_set)

    def __getitem__(self, index):
        eps = 1e-6
        
        if self.with_id:
            noisy, id = self.noisy_set[index]
            clean, id = self.clean_set[index]
        else:
            noisy = self.noisy_set[index]
            clean = self.clean_set[index]

        # If shift is specified, randomly pick an offset for noisy and clean
        if self.shift:
            t = noisy.shape[-1] - self.shift
            # Ensure shift is even and enough frames remain
            assert self.shift % 2 == 0 and t > 0
            offset = random.randint(0, self.shift)
            # Cut both noisy and clean with the chosen offset
            noisy = noisy[..., offset:offset+t]
            clean = clean[..., offset:offset+t]
        
        noisy = torch.tensor(noisy, dtype=torch.float32)
        clean = torch.tensor(clean, dtype=torch.float32)
        
        if self.with_id:
            return noisy, clean, id
        else:
            return noisy, clean


class Audioset:
    def __init__(self, wavs=None, segment=None, stride=None, with_id=False):
        # Store the file list and hyperparameters
        self.wavs = wavs
        self.num_examples = []
        self.segment = segment
        self.stride = stride or segment
        self.with_id = with_id
        
        # Calculate how many segments (examples) each file can produce
        for _, _, wav_length in self.wavs:
            # If no fixed segment length is provided or the file is shorter, only 1 example
            if segment is None or wav_length < segment:
                examples = 1
            else:
                # Otherwise, calculate how many segments fit given stride
                examples = int(math.ceil((wav_length - self.segment) / (self.stride)) + 1)
            self.num_examples.append(examples)

    def __len__(self):
        # The total length is the sum of all examples across files
        return sum(self.num_examples)

    def __getitem__(self, index):
        # Iterate through files and find which file/segment corresponds to 'index'
        for (wav, id, _), examples in zip(self.wavs, self.num_examples):
            # If index is larger than current file's examples, skip to the next file
            if index >= examples:
                index -= examples
                continue
                        
            # Otherwise, compute the offset based on stride and index
            offset = self.stride * index if self.segment else 0
            # Decide how many frames to load (full file if segment is None)
            num_frames = self.segment if self.segment else len(wav)
            # Slice the waveform
            wav = wav[offset:offset+num_frames]
            # If the loaded waveform is shorter than the segment length, pad it
            if self.segment:
                wav = np.pad(wav, (0, num_frames - wav.shape[-1]), 'constant')
                
            # Add channel dimension
            wav = np.expand_dims(wav, axis=0)
                        
            if self.with_id:
                return wav, id
            else:
                return wav

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

def validation_collate_fn(batch):
    noisy, clean = zip(*batch)
        
    noisy = [inp.clone().detach().squeeze() for inp in noisy]
    clean = [inp.clone().detach().squeeze() for inp in clean]
           
    padded_noisy = pad_sequence(noisy, batch_first=True, padding_value=0.0).unsqueeze(1)
    padded_clean = pad_sequence(clean, batch_first=True, padding_value=0.0).unsqueeze(1)
    
    mask = torch.zeros(padded_noisy.shape, dtype=torch.float32)
    for i, length in enumerate([inp.size(0) for inp in noisy]):
        mask[i, :, :length] = 1
    
    return padded_noisy, padded_clean, mask