import torch
import torch.utils.data
import numpy as np
import random

class VoiceBankDataset:
    def __init__(self, 
                 datapair_list,
                 sampling_rate=16_000,
                 segment=None, 
                 with_id=False,
                 ):
        # Initialize variables with constructor arguments
        self.datapair_list = datapair_list
        self.sampling_rate = sampling_rate
        self.segment = segment
        self.with_id = with_id
        
        # Prepare lists for noisy and clean audio arrays
        self.audio_pairs = []
        for item in self.datapair_list:
            noisy = item["noisy"]['array'].astype('float32')
            clean = item["clean"]['array'].astype('float32')
            id = item["id"]

            norm_factor = np.sqrt(noisy.shape[-1] / np.sum(noisy ** 2.0))
            noisy = noisy * norm_factor
            clean = clean * norm_factor
            
            self.audio_pairs.append((noisy, clean, id))
                
    def __len__(self):
        return len(self.audio_pairs)

    def __getitem__(self, index):
        noisy, clean, id = self.audio_pairs[index]
        
        noisy = torch.FloatTensor(noisy).unsqueeze(0)
        clean = torch.FloatTensor(clean).unsqueeze(0)
        
        assert noisy.size(1) == clean.size(1)
        
        if self.segment:
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

        if self.with_id:
            return noisy, clean, id
        else:
            return noisy, clean

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