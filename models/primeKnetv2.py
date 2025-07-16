import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from data import mag_pha_to_complex

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (int((kernel_size[0]*dilation[0] - dilation[0])/2), int((kernel_size[1]*dilation[1] - dilation[1])/2))

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        B, C, T = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1) * y + bias.view(1, C, 1)
        return y
    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        B, C, T = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=2).sum(dim=0), grad_output.sum(dim=2).sum(
            dim=0), None


class LayerNorm1d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm1d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class GCGFN(nn.Module):
    def __init__(self, n_feats, fnn_expend=4):
        super().__init__()
        i_feats = fnn_expend * n_feats

        self.n_feats = n_feats
        self.i_feats = i_feats

        self.norm = LayerNorm1d(n_feats)
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1)), requires_grad=True)

        # Multiscale Large Kernel Attention (replaced with 1D convolutions)
        self.LKA9 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=31, padding=get_padding(31), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA7 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=23, padding=get_padding(23), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA5 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=11, padding=get_padding(11), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA3 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=3, padding=get_padding(3), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.X3 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=3, padding=get_padding(3), groups=i_feats // 4)
        self.X5 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=11, padding=get_padding(11), groups=i_feats // 4)
        self.X7 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=23, padding=get_padding(23), groups=i_feats // 4)
        self.X9 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=31, padding=get_padding(31), groups=i_feats // 4)

        self.proj_first = nn.Sequential(
            nn.Conv1d(n_feats, i_feats, kernel_size=1))

        self.proj_last = nn.Sequential(
            nn.Conv1d(i_feats, n_feats, kernel_size=1))


    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        a_1, a_2, a_3, a_4 = torch.chunk(x, 4, dim=1)
        x = torch.cat([self.LKA3(a_1) * self.X3(a_1), self.LKA5(a_2) * self.X5(a_2), self.LKA7(a_3) * self.X7(a_3),
                       self.LKA9(a_4) * self.X9(a_4)], dim=1)
        x = self.proj_last(x) * self.scale + shortcut
        return x

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

class GhostAttentionModule(nn.Module):
    def __init__(self, dense_channel):
        super(GhostAttentionModule, self).__init__()
        self.dense_channel = dense_channel
        self.ratio = 2
        self.layer_norm = nn.GroupNorm(num_groups=1, num_channels=dense_channel)
        self.init_channel = math.ceil(dense_channel / self.ratio)
        self.cheap_channel = (self.ratio - 1) * self.init_channel
        self.primary_conv = nn.Sequential(
            nn.Conv2d(dense_channel, self.init_channel, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False),
            nn.InstanceNorm2d(self.init_channel, affine=True),
            nn.PReLU(self.init_channel)
        )
        self.cheap_conv = nn.Sequential(
            nn.Conv2d(self.init_channel, self.cheap_channel, kernel_size=(3, 3), stride=1, padding=(1, 1), groups=self.init_channel, bias=False),
            nn.InstanceNorm2d(self.cheap_channel, affine=True),
            nn.PReLU(self.cheap_channel)
        )
        self.short_conv = nn.Sequential(
            nn.Conv2d(dense_channel, dense_channel, kernel_size=(1, 1), stride=1, padding=(0, 0), bias=False),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.Conv2d(dense_channel, dense_channel, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=dense_channel, bias=False),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.Conv2d(dense_channel, dense_channel, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=dense_channel, bias=False),
            nn.InstanceNorm2d(dense_channel, affine=True)
        )
        self.sigmoid = LearnableSigmoid_2d(dense_channel, beta=1)
        self.se = SqueezeExcite(dense_channel, se_ratio=0.25)
        self.alpha = nn.Parameter(torch.zeros((1, 1, 1, dense_channel)), requires_grad=True)
    
    def forward(self, x):
        x_shortcut = x.clone()
        x_attn = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
        x = self.layer_norm(x)
        x1 = self.primary_conv(x)
        x2 = self.cheap_conv(x1)
        x = torch.cat([x1, x2], dim=1)
        x = x[:, :self.dense_channel, :, :]*F.interpolate(self.sigmoid(x_attn), size=(x.shape[-2], x.shape[-1]), mode='nearest')

        x = self.se(x) * self.alpha + x_shortcut

        return x

class GGGFN(nn.Module):
    def __init__(self, dense_channel):
        super(GGGFN, self).__init__()
        
        self.ghost_attention1 = GhostAttentionModule(dense_channel)
        self.GCGFN_T = GCGFN(dense_channel)
        self.ghost_attention2 = GhostAttentionModule(dense_channel)
        self.GCGFN_F = GCGFN(dense_channel)
        self.beta = nn.Parameter(torch.zeros((1, 1, 1, dense_channel)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, 1, 1, dense_channel)), requires_grad=True)
    
    def forward(self, x):
        
        x = self.ghost_attention1(x)
        b, c, t, f = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b*f, c, t)
        x = self.GCGFN_T(x) + x * self.beta

        x = x.view(b, f, c, t).permute(0, 2, 3, 1).contiguous()

        x = self.ghost_attention2(x)

        x = x.permute(0, 2, 1, 3).contiguous().view(b*t, c, f)
        x = self.GCGFN_F(x) + x * self.gamma
        
        x = x.view(b, t, c, f).permute(0, 2, 1, 3).contiguous()

        return x

class TS_BLOCK(nn.Module):
    def __init__(self, dense_channel, num_blocks=4):
        super(TS_BLOCK, self).__init__()
        self.dense_channel = dense_channel
        self.num_blocks = num_blocks
        self.GGGFN_blocks = nn.ModuleList([GGGFN(dense_channel) for _ in range(num_blocks)])

    def forward(self, x):
        for i in range(self.num_blocks):
            x = self.GGGFN_blocks[i](x)
        return x


class PrimeKnetv2(nn.Module):
    def __init__(self, 
                 fft_len=400,
                 dense_channel=64,
                 sigmoid_beta=2.0,
                 num_tsblock=4
                 ):
        super(PrimeKnetv2, self).__init__()
        self.fft_len = fft_len
        self.dense_channel = dense_channel
        self.num_tsblock = num_tsblock
        self.dense_encoder = DenseEncoder(dense_channel, in_channel=2)
        self.TS_blocks = nn.ModuleList([])
        for i in range(num_tsblock):
            self.TS_blocks.append(TS_BLOCK(dense_channel))
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

        for i in range(self.num_tsblock):
            x = self.TS_blocks[i](x)
        
        denoised_mag = (mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)
        
        denoised_com = mag_pha_to_complex(denoised_mag, denoised_pha)
        
        outputs = {
            "magnitude": denoised_mag,
            "phase": denoised_pha,
            "complex": denoised_com,
        }

        return outputs