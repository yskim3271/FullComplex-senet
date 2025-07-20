import torch
import torch.nn as nn

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
    def __init__(self, in_channel, kernel_list=[3, 11, 23, 31]):
        super().__init__()
        self.in_channel = in_channel
        self.expand_ratio = len(kernel_list)
        self.mid_channel = self.in_channel * self.expand_ratio
        self.kernel_size_list = kernel_list

        self.proj_first = nn.Sequential(
            nn.Conv1d(self.in_channel, self.mid_channel, kernel_size=1))
        self.proj_last = nn.Sequential(
            nn.Conv1d(self.mid_channel, self.in_channel, kernel_size=1))
        self.norm = LayerNorm1d(self.in_channel)
        self.scale = nn.Parameter(torch.zeros((1, self.in_channel, 1)), requires_grad=True)

        for ksize in kernel_list:
            setattr(self, f"attn_{ksize}", nn.Sequential(
                nn.Conv1d(self.in_channel, self.in_channel, kernel_size=ksize, padding=get_padding(ksize), groups=self.in_channel),
                nn.Conv1d(self.in_channel, self.in_channel, kernel_size=1)
            ))
            setattr(self, f"conv_{ksize}", nn.Conv1d(
                self.in_channel, self.in_channel, kernel_size=ksize, padding=get_padding(ksize), groups=self.in_channel))

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        x_list = list(torch.chunk(x, self.expand_ratio, dim=1))
        for i, ksize in enumerate(self.kernel_size_list):
            x_list[i] = getattr(self, f"attn_{ksize}")(x_list[i]) * getattr(self, f"conv_{ksize}")(x_list[i])
        x = torch.cat(x_list, dim=1)
        x = self.proj_last(x) * self.scale + shortcut
        return x

class LKFCA_Block(nn.Module):
    def __init__(self, in_channels, DW_Expand=2):
        super().__init__()

        dw_channel = in_channels * DW_Expand

        self.pwconv1 = nn.Conv1d(in_channels=in_channels, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.dwconv = nn.Conv1d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.pwconv2 = nn.Conv1d(in_channels=dw_channel // 2, out_channels=in_channels, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        # SimpleGate
        self.sg = SimpleGate()
        self.norm1 = LayerNorm1d(in_channels)
        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1)), requires_grad=True)
        self.GCGFN = GCGFN(in_channels)


    def forward(self, x):

        inp2 = x
        x = self.norm1(inp2)
        x = self.pwconv1(x)
        x = self.dwconv(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.pwconv2(x)

        inp3 = inp2 + x * self.beta
        x = self.GCGFN(inp3)

        return x

class TS_BLOCK(nn.Module):
    def __init__(self, dense_channel):
        super(TS_BLOCK, self).__init__()
        self.dense_channel = dense_channel
        self.time = nn.Sequential(
            LKFCA_Block(dense_channel),
            LKFCA_Block(dense_channel),
        )
        self.freq = nn.Sequential(
            LKFCA_Block(dense_channel),
            LKFCA_Block(dense_channel),
        )
        self.beta = nn.Parameter(torch.zeros((1, 1, dense_channel, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, 1, dense_channel, 1)), requires_grad=True)
    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b*f, c, t) 

        x = self.time(x) + x * self.beta
        x = x.view(b, f, c, t).permute(0, 3, 2, 1).contiguous().view(b*t, c, f)

        x = self.freq(x) + x * self.gamma
        x = x.view(b, t, c, f).permute(0, 2, 1, 3)
        return x


class LearnableSigmoid_2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)

class repDDB(nn.Module):
    def __init__(self, dense_channel, depth=4):
        super(repDDB, self).__init__()
        self.dense_channel = dense_channel
        self.depth = depth
        self.dense_block1 = nn.ModuleList([])
        self.dense_block2 = nn.ModuleList([])
        self.dense_block3 = nn.ModuleList([])
        for i in range(depth):
            dil = 2 ** i

            dense_conv5 = nn.Sequential(
                nn.Conv2d(dense_channel*(i+1), dense_channel*(i+1), kernel_size=(5, 5), dilation=(dil, 1),
                          padding=get_padding_2d((5, 5), dilation=(dil, 1)), bias=True),
                nn.BatchNorm2d(dense_channel),
            )
            dense_conv3 = nn.Sequential(
                nn.Conv2d(dense_channel*(i+1), dense_channel*(i+1), kernel_size=(3, 3), dilation=(dil, 1),
                          padding=get_padding_2d((3, 3), dilation=(dil, 1)), bias=True),
                nn.BatchNorm2d(dense_channel),
            )
            dense_conv1 = nn.Sequential(
                nn.Conv2d(dense_channel*(i+1), dense_channel*(i+1), kernel_size=(1, 1), dilation=(dil, 1),
                          padding=get_padding_2d((1, 1), dilation=(dil, 1)), bias=True),
                nn.BatchNorm2d(dense_channel),
            )

            self.dense_block1.append(dense_conv5)
            self.dense_block2.append(dense_conv3)
            self.dense_block3.append(dense_conv1)
            self.nonlinearity = nn.PReLU(dense_channel)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x1 = self.dense_block1[i](skip)
            x2= self.dense_block2[i](skip)
            x3 = self.dense_block3[i](skip)
            skip = torch.cat([self.nonlinearity(x1+x2+x3), skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, dense_channel, in_channel):
        super(DenseEncoder, self).__init__()
        self.dense_channel = dense_channel
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, dense_channel, (1, 1)),
            nn.BatchNorm2d(dense_channel),
            nn.PReLU(dense_channel))

        self.dense_block = repDDB(dense_channel, depth=4) # [b, h.dense_channel, ndim_time, h.n_fft//2+1]

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.BatchNorm2d(dense_channel),
            nn.PReLU(dense_channel))

    def forward(self, x):
        x = self.dense_conv_1(x)  # [b, 64, T, F]
        x = self.dense_block(x)   # [b, 64, T, F]
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x

class SPConvTranspose2d(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 r=1,
                 fft_len=400
                 ):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 2, 0, 0), value=0.)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r
        self.fft_len = fft_len

    def forward(self, x):
        # [B, C, T, F//2]
        x = self.pad1(x)
        out = self.conv(x) # [B, C*r, T, F]
        B, C, T, F = out.shape
        out = out.view((B, self.r, C // self.r, T, F))
        out = out.permute(0, 2, 3, 4, 1) # [B, C//r, T, F, r]
        out = out.contiguous().view((B, C // self.r, T, -1)) # [B, C//r, T, F*r]
        out = out[:, :, :, :self.fft_len//2+1]
        return out

class MaskDecoder(nn.Module):
    def __init__(self, 
                 dense_channel,
                 n_fft,
                 sigmoid_beta,
                 out_channel=1):
        super(MaskDecoder, self).__init__()
        self.n_fft = n_fft
        self.dense_block = repDDB(dense_channel, depth=4)
        self.mask_conv = nn.Sequential(
            nn.Conv2d(dense_channel, dense_channel*2, kernel_size=1),
            nn.GLU(dim=1),
            SPConvTranspose2d(dense_channel, dense_channel, (1, 3), r=2),
            nn.Conv2d(dense_channel, out_channel, (1, 1)),
            nn.InstanceNorm2d(out_channel)
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
        self.dense_block = repDDB(dense_channel, depth=4)
        self.phase_conv = nn.Sequential(
            nn.Conv2d(dense_channel, dense_channel*2, kernel_size=1),
            nn.GLU(dim=1),
            SPConvTranspose2d(dense_channel, dense_channel, (1, 3), r=2),
            nn.InstanceNorm2d(dense_channel),
            nn.SiLU(),
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

class PrimeKnetv11(nn.Module):
    def __init__(self, 
                 fft_len=400,
                 dense_channel=64,
                 sigmoid_beta=2.0,
                 num_tsblock=4
                 ):
        super(PrimeKnetv11, self).__init__()
        self.fft_len = fft_len
        self.dense_channel = dense_channel
        self.num_tsblock = num_tsblock
        self.dense_encoder = DenseEncoder(dense_channel, in_channel=2)
        self.LKFCAnet = nn.ModuleList([])
        for i in range(num_tsblock):
            self.LKFCAnet.append(TS_BLOCK(dense_channel))
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
            x = self.LKFCAnet[i](x)
        
        denoised_mag = (mag * self.mask_decoder(x)).permute(0, 3, 2, 1).squeeze(-1)
        denoised_pha = self.phase_decoder(x).permute(0, 3, 2, 1).squeeze(-1)
        
        denoised_com = mag_pha_to_complex(denoised_mag, denoised_pha)
        
        outputs = {
            "magnitude": denoised_mag,
            "phase": denoised_pha,
            "complex": denoised_com,
        }

        return outputs