import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from pesq import pesq
from data import mag_pha_stft, mag_pha_istft
from torch_pesq import PesqLoss

def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

def phase_losses(phase_r, phase_g):
    ip_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)))
    iaf_loss = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))

    return ip_loss + gd_loss + iaf_loss

def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, "wb")
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score


def batch_pesq(clean, noisy, workers=10, normalize=True):
    pesq_score = Parallel(n_jobs=workers)(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    if normalize:
        pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score)


class MetricGAN_Loss(torch.nn.Module):
    def __init__(self, 
                 fft_size, 
                 hop_size, 
                 win_length,
                 compress_factor, 
                 discriminator, 
                 weight_gen=1.0,
                 weight_disc=1.0, 
                 pesq_workers=10,
                 use_torch_pesq=False
                 ):
        super().__init__()
        self.name = "MetricGAN_Loss"
        self.discriminator = discriminator
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.compress_factor = compress_factor
        self.weight_gen = weight_gen
        self.weight_disc = weight_disc
        self.pesq_workers = pesq_workers
        self.use_torch_pesq = use_torch_pesq
        self.torch_pesq = None
        if self.use_torch_pesq:
            self.torch_pesq = PesqLoss(factor=1.0, sample_rate=16000)

    def calculate_disc_loss(self, x, y):
        batch_size = x["wav"].shape[0]

        if self.use_torch_pesq:
            self.torch_pesq.to(x["wav"].device)
            pesq_score = self.torch_pesq.mos(ref=y["wav"].squeeze(1), deg=x["wav"].squeeze(1).detach())
            pesq_score = (pesq_score - 1.08) / 4.999
            pesq_score = pesq_score.type(torch.FloatTensor)
        else:
            x_list = list(x["wav"].squeeze(1).detach().cpu().numpy())
            y_list = list(y["wav"].squeeze(1).detach().cpu().numpy())
            pesq_score = batch_pesq(y_list, x_list, workers=self.pesq_workers)
        
        if pesq_score is not None:
            x_mag = x["magnitude"].unsqueeze(1).detach()
            y_mag = y["magnitude"].unsqueeze(1)

            predict_enhance_metric = self.discriminator(y_mag, x_mag)
            predict_max_metric = self.discriminator(y_mag, y_mag.detach())
            discriminator_loss = F.mse_loss(
                predict_max_metric.flatten(), torch.ones(batch_size).to(x_mag.device)
            ) + F.mse_loss(
                predict_enhance_metric.flatten(), pesq_score.to(x_mag.device))
            discriminator_loss = discriminator_loss * self.weight_disc
        else:
            discriminator_loss = None

        return discriminator_loss

    def forward(self, x, y):
        batch_size = x["magnitude"].shape[0]

        predict_fake_metric = self.discriminator(y["magnitude"].unsqueeze(1), x["magnitude"].unsqueeze(1))

        generator_loss = F.mse_loss(
            predict_fake_metric.flatten(), torch.ones(batch_size).to(x["magnitude"].device)
        ).float()

        return generator_loss * self.weight_gen


class TimeLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(TimeLoss, self).__init__()
        self.name = "Time_Loss" 
        self.weight = weight

    def forward(self, x, y):
        x_wav = x["wav"]
        y_wav = y["wav"]
        return F.l1_loss(x_wav, y_wav) * self.weight
    
class PhaseLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(PhaseLoss, self).__init__()
        self.name = "Phase_Loss"
        self.weight = weight

    def forward(self, x, y):
        return phase_losses(x["phase"], y["phase"]) * self.weight

class MagnitudeLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(MagnitudeLoss, self).__init__()
        self.name = "Magnitude_Loss"
        self.weight = weight

    def forward(self, x, y):
        return F.mse_loss(x["magnitude"], y["magnitude"]) * self.weight
    
class ComplexLoss(torch.nn.Module):
    def __init__(self, weight=1.0):
        super(ComplexLoss, self).__init__()
        self.name = "Complex_Loss"
        self.weight = weight

    def forward(self, x, y):
        return F.mse_loss(x["complex"], y["complex"]) * 2 * self.weight   
    
class ConsistencyLoss(torch.nn.Module):
    def __init__(self, fft_size, hop_size, win_length, compress_factor, weight=1.0):
        super(ConsistencyLoss, self).__init__()
        self.name = "Consistency_Loss"
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.compress_factor = compress_factor
        self.weight = weight

    def forward(self, x, y):
        mag_con, pha_con, com_con = mag_pha_stft(x["wav"], 
                                        self.fft_size, 
                                        self.hop_size, 
                                        self.win_length, 
                                        compress_factor=self.compress_factor, 
                                        center=True)

        return F.mse_loss(mag_con, y["magnitude"]) * self.weight

class CompositeLoss(torch.nn.Module):
    def __init__(self, args, fft_size=400, hop_size=100, win_length=400, compress_factor=1.0, discriminator=None):
        super(CompositeLoss, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.compress_factor = compress_factor
        self.loss_dict = {}
        self.discriminator = discriminator

        allowed_keys = {"time_loss", "phase_loss", "magnitude_loss", "complex_loss", "consistency_loss", "metricgan_loss"}
        unknown_keys = set(args.keys()) - allowed_keys
        if unknown_keys:
            raise ValueError(f"Unknown loss keys in args: {unknown_keys}.")
        
        weight_time_loss = args.get("time_loss")
        if weight_time_loss is not None:
            self.loss_dict['Time'] = TimeLoss(weight=weight_time_loss)

        weight_phase_loss = args.get("phase_loss")
        if weight_phase_loss is not None:
            self.loss_dict['Phase'] = PhaseLoss(weight_phase_loss)

        weight_magnitude_loss = args.get("magnitude_loss")
        if weight_magnitude_loss is not None:
            self.loss_dict['Magnitude'] = MagnitudeLoss(weight_magnitude_loss)
        
        weight_complex_loss = args.get("complex_loss")
        if weight_complex_loss is not None:
            self.loss_dict['Complex'] = ComplexLoss(weight_complex_loss)
        
        weight_consistency_loss = args.get("consistency_loss")
        if weight_consistency_loss is not None:
            self.loss_dict['Consistency'] = ConsistencyLoss(
                fft_size=self.fft_size,
                hop_size=self.hop_size,
                win_length=self.win_length,
                compress_factor=self.compress_factor,
                weight=weight_consistency_loss
            )
        
        metricgan_loss_cfg = args.get("metricgan_loss")
        if metricgan_loss_cfg is not None:
            self.loss_dict['MetricGAN'] = MetricGAN_Loss(
                fft_size=self.fft_size,
                hop_size=self.hop_size,
                win_length=self.win_length,
                compress_factor=self.compress_factor,
                discriminator=self.discriminator, 
                **metricgan_loss_cfg
            )
            
    def forward(self, x, y):
        loss_all = 0
        loss_dict = {}
        
        x["wav"] = mag_pha_istft(x["magnitude"], x["phase"], self.fft_size, self.hop_size, self.win_length, self.compress_factor)
        
        for loss_name, loss_fn in self.loss_dict.items():
            loss = loss_fn(x, y)
            loss_all += loss
            loss_dict[loss_name] = loss

        return loss_all, loss_dict
    
    def forward_disc_loss(self, x, y):
        return self.loss_dict['MetricGAN'].calculate_disc_loss(x, y)
