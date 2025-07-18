import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from data import mag_pha_to_complex

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (int((kernel_size[0]*dilation[0] - dilation[0])/2), int((kernel_size[1]*dilation[1] - dilation[1])/2))

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = nn.Sigmoid()
        reduced_chs = math.ceil(in_chs * se_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x    

class LearnableSigmoid_2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class DS_DDB(nn.Module):
    def __init__(self, dense_channel, kernel_size=(3, 3), depth=4):
        super(DS_DDB, self).__init__()
        self.dense_channel = dense_channel
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(dense_channel*(i+1), dense_channel*(i+1), kernel_size, dilation=(dil, 1),
                          padding=get_padding_2d(kernel_size, dilation=(dil, 1)), groups=dense_channel*(i+1), bias=True),
                nn.Conv2d(in_channels=dense_channel*(i+1), out_channels=dense_channel, kernel_size=1, padding=0, stride=1, groups=1,
                          bias=True),
                nn.InstanceNorm2d(dense_channel, affine=True),
                nn.PReLU(dense_channel)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x

class DenseEncoder(nn.Module):
    def __init__(self, dense_channel, in_channel):
        super(DenseEncoder, self).__init__()
        self.dense_channel = dense_channel
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, dense_channel, (1, 1)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel))

        self.dense_block = DS_DDB(dense_channel, depth=4) # [b, h.dense_channel, ndim_time, h.n_fft//2+1]

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel))

    def forward(self, x):
        x = self.dense_conv_1(x)  # [b, 64, T, F]
        x = self.dense_block(x)   # [b, 64, T, F]
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x

class MaskDecoder(nn.Module):
    def __init__(self, 
                 dense_channel,
                 n_fft,
                 sigmoid_beta,
                 out_channel=1):
        super(MaskDecoder, self).__init__()
        self.n_fft = n_fft
        self.dense_block = DS_DDB(dense_channel, depth=4)
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.Conv2d(dense_channel, out_channel, (1, 1)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1))
        )
        self.lsigmoid = LearnableSigmoid_2d(n_fft//2+1, beta=sigmoid_beta)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        x = self.lsigmoid(x).permute(0, 2, 1).unsqueeze(1)
        return x

class PhaseDecoder(nn.Module):
    def __init__(self, 
                 dense_channel, 
                 out_channel=1):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DS_DDB(dense_channel, depth=4)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel)
        )
        self.phase_conv_r = nn.Conv2d(dense_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(dense_channel, out_channel, (1, 1))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x

class PrimeFFN(nn.Module):
    def __init__(self, in_channel, kernel_size_list=[3, 11, 23, 31], mode="time"):
        super(PrimeFFN, self).__init__()

        self.in_channel = in_channel
        self.expand_ratio = len(kernel_size_list)
        self.mid_channel =  in_channel * self.expand_ratio
        self.kernel_size_list = kernel_size_list
        self.mode = mode

        self.norm = nn.InstanceNorm2d(in_channel)
        for ksize in kernel_size_list:
            setattr(self, f"attn_{ksize}", nn.Sequential(
                nn.Conv1d(in_channel, in_channel, kernel_size=ksize, padding=get_padding(ksize), groups=in_channel),
                nn.Conv1d(in_channel, in_channel, kernel_size=1),
            ))
            setattr(self, f"conv_{ksize}", nn.Conv1d(in_channel, in_channel, kernel_size=ksize, padding=get_padding(ksize), groups=in_channel))

        self.expand = nn.Conv1d(in_channel, self.mid_channel, kernel_size=1)
        self.reduce = nn.Conv1d(self.mid_channel, in_channel, kernel_size=1)
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1), requires_grad=True)

    def forward(self, x):
        b, c, t, f = x.size()
        if self.mode == "time":
            x = x.permute(0, 3, 1, 2).contiguous().view(b*f, c, t)
        elif self.mode == "freq":
            x = x.permute(0, 2, 1, 3).contiguous().view(b*t, c, f)

        shortcut = x.clone()
        x = self.norm(x)
        x = self.expand(x)
        x_list = list(torch.chunk(x, self.expand_ratio, dim=1))
        for i, ksize in enumerate(self.kernel_size_list):
            x_list[i] = getattr(self, f"attn_{ksize}")(x_list[i]) * getattr(self, f"conv_{ksize}")(x_list[i])
        x = torch.cat(x_list, dim=1)
        x = self.reduce(x)
        x = x * self.scale + shortcut
        if self.mode == "time":
            x = x.view(b, f, c, t).permute(0, 2, 3, 1).contiguous()
        elif self.mode == "freq":
            x = x.view(b, t, c, f).permute(0, 2, 1, 3).contiguous()
        return x


class SPConvTranspose2d(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 r=1
                 ):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out

class DFC_AttentionModule(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(DFC_AttentionModule, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=1, padding=get_padding(kernel_size), bias=True),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel)
        )
        self.attn_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), stride=1, padding=(0, 2), bias=False),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), stride=1, padding=(2, 0), bias=False),
            nn.InstanceNorm2d(out_channel, affine=True),
        )
        self.gate_fn = nn.Sigmoid()

    def forward(self, x):
        res = self.attn_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
        x = self.conv(x)

        return x[:,:self.out_channel,:,:]*F.interpolate(self.gate_fn(res), size=(x.shape[-2], x.shape[-1]), mode='nearest')

class PrimeAttentionEncoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, dw_size=3):
        super(PrimeAttentionEncoder, self).__init__()
        
        self.dfc_attention = DFC_AttentionModule(in_channel, out_channel, kernel_size=kernel_size)
        self.se = SqueezeExcite(out_channel, se_ratio=0.25)
        self.primeFFN_time = PrimeFFN(out_channel, mode="time", kernel_size_list=[3, 11, 23, 31])
        self.primeFFN_freq = PrimeFFN(out_channel, mode="freq", kernel_size_list=[3, 5, 7])
        self.conv_dw = nn.Conv2d(out_channel, out_channel, kernel_size=dw_size, stride=(1, 2), padding=1, groups=out_channel, bias=False)
        self.norm_dw = nn.InstanceNorm2d(out_channel, affine=True)

    def forward(self, x):
        shortcut = x.clone()
        x = self.dfc_attention(x)
        x = self.se(x)
        x = self.primeFFN_time(x)
        x = self.primeFFN_freq(x)
        x = self.conv_dw(x)
        x = self.norm_dw(x)
        return x

class PrimeAttentionBlock(nn.Module):
    def __init__(self, kernel_size=3, dw_size=3):
        super(PrimeAttentionBlock, self).__init__()
        
        self.encoder1 = PrimeAttentionEncoder(64, 96, kernel_size=kernel_size, dw_size=dw_size)
        self.encoder2 = PrimeAttentionEncoder(96, 128, kernel_size=kernel_size, dw_size=dw_size)
        self.decoder1 = nn.Sequential(
            SPConvTranspose2d(128, 96, kernel_size=(1, 3), r=2),
            nn.InstanceNorm2d(96, affine=True),
            nn.PReLU(96)
        )
        self.decoder2 = nn.Sequential(
            SPConvTranspose2d(96, 64, kernel_size=(1, 3), r=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        self.beta = nn.Parameter(torch.ones(1, 96, 1, 1), requires_grad=True)

    def forward(self, x):
        x = self.encoder1(x)
        skip = x 
        x = self.encoder2(x)
        x = self.decoder1(x)
        x = x + skip * self.beta
        x = self.decoder2(x)         
        return x


class PrimeKnetv6(nn.Module):
    def __init__(self, 
                 fft_len=400,
                 dense_channel=64,
                 sigmoid_beta=2.0,
                 ):
        super(PrimeKnetv6, self).__init__()
        self.fft_len = fft_len
        self.dense_channel = dense_channel
        self.dense_encoder = DenseEncoder(dense_channel, in_channel=2)
        self.prime_attention_block = PrimeAttentionBlock(kernel_size=3, dw_size=3)
        self.mask_decoder = MaskDecoder(dense_channel, fft_len, sigmoid_beta, out_channel=1)
        self.phase_decoder = PhaseDecoder(dense_channel, out_channel=1)

    def forward(self, inputs):
        # Input shape: [B, F, T]
        mag = inputs["magnitude"]
        pha = inputs["phase"]

        mag = mag.unsqueeze(1).permute(0, 1, 3, 2) # [B, 1, T, F]
        pha = pha.unsqueeze(1).permute(0, 1, 3, 2) # [B, 1, T, F]

        x = torch.cat((mag, pha), dim=1) # [B, 2, T, F]

        x = self.dense_encoder(x)

        x = self.prime_attention_block(x)
        
        denoised_mag = (mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)
        
        denoised_com = mag_pha_to_complex(denoised_mag, denoised_pha)
        
        outputs = {
            "magnitude": denoised_mag,
            "phase": denoised_pha,
            "complex": denoised_com,
        }

        return outputs