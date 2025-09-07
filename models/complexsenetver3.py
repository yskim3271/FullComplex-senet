import torch
import torch.nn.functional as F
import torch.nn as nn
from models.complexnn import ComplexConv2d, ComplexConvTranspose2d, ComplexIN2d, ComplexLeakyModReLU2d, ComplexLayerNorm1d, ComplexConv1d, ComplexModPReLU2d, \
    ComplexModPReLU2d, zReLU, CVCardiod, ComplexGubermanModLeakyReLU2d

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
    def __init__(self, in_channel=64):
        super().__init__()
        self.in_channel = in_channel
        self.expand_ratio = 4
        self.mid_channel = self.in_channel * self.expand_ratio
        self.kernel_size_list = [3, 11, 23, 31]

        self.proj_first = nn.Sequential(
            nn.Conv1d(self.in_channel, self.mid_channel, kernel_size=1))
        self.proj_last = nn.Sequential(
            nn.Conv1d(self.mid_channel, self.in_channel, kernel_size=1))
        self.norm = LayerNorm1d(self.in_channel)
        self.scale = nn.Parameter(torch.zeros((1, self.in_channel, 1)), requires_grad=True)

        self.attn_3 = nn.Sequential(
            nn.Conv1d(self.in_channel, self.in_channel, kernel_size=3, padding=get_padding(3), groups=self.in_channel),
            nn.Conv1d(self.in_channel, self.in_channel, kernel_size=1)
        )
        self.attn_11 = nn.Sequential(
            nn.Conv1d(self.in_channel, self.in_channel, kernel_size=11, padding=get_padding(11), groups=self.in_channel),
            nn.Conv1d(self.in_channel, self.in_channel, kernel_size=1)
        )
        self.attn_23 = nn.Sequential(
            nn.Conv1d(self.in_channel, self.in_channel, kernel_size=23, padding=get_padding(23), groups=self.in_channel),
            nn.Conv1d(self.in_channel, self.in_channel, kernel_size=1)
        )
        self.attn_31 = nn.Sequential(
            nn.Conv1d(self.in_channel, self.in_channel, kernel_size=31, padding=get_padding(31), groups=self.in_channel),
            nn.Conv1d(self.in_channel, self.in_channel, kernel_size=1)
        )
        self.conv_3 = nn.Conv1d(self.in_channel, self.in_channel, kernel_size=3, padding=get_padding(3), groups=self.in_channel)
        self.conv_11 = nn.Conv1d(self.in_channel, self.in_channel, kernel_size=11, padding=get_padding(11), groups=self.in_channel)
        self.conv_23 = nn.Conv1d(self.in_channel, self.in_channel, kernel_size=23, padding=get_padding(23), groups=self.in_channel)
        self.conv_31 = nn.Conv1d(self.in_channel, self.in_channel, kernel_size=31, padding=get_padding(31), groups=self.in_channel)

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        x1 = self.attn_3(x1) * self.conv_3(x1)
        x2 = self.attn_11(x2) * self.conv_11(x2)
        x3 = self.attn_23(x3) * self.conv_23(x3)
        x4 = self.attn_31(x4) * self.conv_31(x4)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.proj_last(x) * self.scale + shortcut
        return x

class LKFCA_Block(nn.Module):
    def __init__(self, in_channels=64, DW_Expand=2):
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
    def __init__(self, dense_channel=64):
        super(TS_BLOCK, self).__init__()
        self.dense_channel = dense_channel
        self.time = nn.Sequential(
            LKFCA_Block(dense_channel*2),
            LKFCA_Block(dense_channel*2),
        )
        self.freq = nn.Sequential(
            LKFCA_Block(dense_channel*2),
            LKFCA_Block(dense_channel*2),
        )
        self.beta = nn.Parameter(torch.zeros((1, 1, dense_channel*2, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, 1, dense_channel*2, 1)), requires_grad=True)
    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b*f, c, t) 

        x = self.time(x) + x * self.beta
        x = x.view(b, f, c, t).permute(0, 3, 2, 1).contiguous().view(b*t, c, f)

        x = self.freq(x) + x * self.gamma
        x = x.view(b, t, c, f).permute(0, 2, 1, 3)
        return x

class ComplexDSDDB(nn.Module):
    def __init__(self, 
                 dense_channel=64, 
                 kernel_size=(3, 3), 
                 depth=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            dil = 2 ** i
            padding = get_padding_2d(kernel_size, dilation=(dil, 1))
            self.layers.append(
                nn.ModuleDict({
                    "dw": ComplexConv2d(dense_channel * (i + 1), dense_channel * (i + 1), kernel_size, 
                                        dilation=(dil, 1), padding=padding, 
                                        groups=dense_channel * (i + 1), bias=True),
                    "pw": ComplexConv2d(dense_channel * (i + 1), dense_channel, (1, 1), bias=True),
                    "norm": ComplexIN2d(dense_channel),
                    "act": ComplexGubermanModLeakyReLU2d(dense_channel, negative_slope=0.2, bias=True, init_bias=-0.05, mask_type='const')
                })
            )

    def forward(self, xr, xi):
        skip_r, skip_i = xr, xi
        for m in self.layers:
            yr, yi = m["dw"](skip_r, skip_i)
            yr, yi = m["pw"](yr, yi)
            yr, yi = m["norm"](yr, yi)
            yr, yi = m["act"](yr, yi)
            # dense concat along channel dimension
            skip_r = torch.cat([yr, skip_r], dim=1)
            skip_i = torch.cat([yi, skip_i], dim=1)
        return yr, yi


class ComplexDenseEncoder(nn.Module):
    def __init__(self, 
                 dense_channel=64,
                 in_channel=1):
        super().__init__()
        self.pre_conv = ComplexConv2d(in_channel, dense_channel, (1,1), bias=True)
        self.pre_norm = ComplexIN2d(dense_channel)
        self.pre_act = ComplexGubermanModLeakyReLU2d(dense_channel, negative_slope=0.2, bias=True, init_bias=-0.05, mask_type='const')
        self.block = ComplexDSDDB(dense_channel, depth=4)
        self.post_conv = ComplexConv2d(dense_channel, dense_channel, (1, 3), stride=(1, 2), bias=True)
        self.post_norm = ComplexIN2d(dense_channel)
        self.post_act = ComplexGubermanModLeakyReLU2d(dense_channel, negative_slope=0.2, bias=True, init_bias=-0.05, mask_type='const')

        self.proj_conv = ComplexConv2d(dense_channel*2, dense_channel*2, (1, 3), stride=(1, 2), bias=True, groups=dense_channel)
        self.proj_norm = nn.InstanceNorm2d(dense_channel*2)
        self.proj_act = nn.PReLU(dense_channel*2)

    def forward(self, xr, xi):
        xr, xi = self.pre_conv(xr, xi)
        xr, xi = self.pre_norm(xr, xi)
        xr, xi = self.pre_act(xr, xi)
        xr, xi = self.block(xr, xi)
        xr, xi = self.post_conv(xr, xi)
        xr, xi = self.post_norm(xr, xi)
        xr, xi = self.post_act(xr, xi)

        xm = xr*xr + xi*xi
        xp = torch.atan2(xi, xr)
        x = torch.cat([xm, xp], dim=1)
        x = self.proj_conv(x)
        x = self.proj_norm(x)
        x = self.proj_act(x)

        return x


class ComplexMaskDecoder(nn.Module):
    def __init__(self, 
                 dense_channel=64,
                 beta=2.0):
        super().__init__()
        self.dense_channel = dense_channel
        self.proj_conv = ComplexConv2d(dense_channel*2, dense_channel*2, (1, 3), stride=(1, 2), bias=True, groups=dense_channel)
        self.proj_norm = nn.InstanceNorm2d(dense_channel*2)
        self.proj_act = nn.PReLU(dense_channel*2)
        self.block = ComplexDSDDB(dense_channel, depth=4)
        self.mask_convtr = ComplexConvTranspose2d(dense_channel, dense_channel, kernel_size=(1, 3), stride=(1, 2))
        self.mask_convpw1 = ComplexConv2d(dense_channel, 1, (1, 1), bias=True)
        self.beta = beta

    def forward(self, x):
        x = self.proj_conv(x)
        x = self.proj_norm(x)
        x = self.proj_act(x)
        xr, xi = x[:, :self.dense_channel], x[:, self.dense_channel:]

        xr, xi = self.block(xr, xi)
        xr, xi = self.mask_convtr(xr, xi)
        mr, mi = self.mask_convpw1(xr, xi)

        rho = torch.sqrt(mr*mr + mi*mi + 1e-8)
        scale = self.beta * torch.tanh(rho) / (rho + 1e-8)
        mr = scale * mr
        mi = scale * mi
        return mr, mi


class ComplexSENet(nn.Module):
    def __init__(self,
                 dense_channel=64,
                 beta=2.0,
                 num_tsblock=4):
        super().__init__()
        self.dense_channel = dense_channel
        # Complex encoder expects 2‑channel (R,I)
        self.encoder = ComplexDenseEncoder(dense_channel, in_channel=1)
        self.ts_blocks = nn.ModuleList([TS_BLOCK(self.dense_channel) for _ in range(num_tsblock)])
        self.decoder = ComplexMaskDecoder(self.dense_channel, beta=beta)

    def forward(self, noisy_com): # [B, F, T]
        
        xr = noisy_com[:, :, :, 0]
        xi = noisy_com[:, :, :, 1]
        xr = xr.unsqueeze(1).permute(0, 1, 3, 2)
        xi = xi.unsqueeze(1).permute(0, 1, 3, 2)

        feat = self.encoder(xr, xi)
        for blk in self.ts_blocks:
            feat = blk(feat)
        mr, mi = self.decoder(feat)

        mr = mr.squeeze(1).permute(0, 2, 1)
        mi = mi.squeeze(1).permute(0, 2, 1)

        deno_r = mr * xr.squeeze(1).permute(0, 2, 1) - mi * xi.squeeze(1).permute(0, 2, 1)
        deno_i = mr * xi.squeeze(1).permute(0, 2, 1) + mi * xr.squeeze(1).permute(0, 2, 1)

        deno_mag = torch.sqrt(deno_r ** 2 + deno_i ** 2 + 1e-9)
        deno_pha = torch.atan2(deno_i, deno_r)
    
        deno_com = torch.stack((deno_mag*torch.cos(deno_pha),
                                deno_mag*torch.sin(deno_pha)), dim=-1)
        
        return deno_mag, deno_pha, deno_com


if __name__ == "__main__":
    # Example usage
    model = ComplexSENet(dense_channel=64, beta=2.0)

    noisy_com = torch.randn(2, 201, 200, 2)

    deno_mag, deno_pha, deno_com = model(noisy_com)
