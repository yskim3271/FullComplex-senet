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
    def forward(self, xr, xi):
        x1r, x2r = xr.chunk(2, dim=1)
        x1i, x2i = xi.chunk(2, dim=1)
        x1 = x1r * x2r - x1i * x2i
        x2 = x1i * x2r + x1r * x2i
        return x1, x2

class ComplexSwiGLU1d(nn.Module):
    def __init__(self, in_channel, kernel_size, padding):
        super().__init__()
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.padding = padding
        self.attn_dw = ComplexConv1d(in_channel, in_channel, kernel_size=kernel_size, padding=padding, groups=in_channel)
        self.attn_pw = ComplexConv1d(in_channel, in_channel, kernel_size=1)
        self.attn_conv = ComplexConv1d(in_channel, in_channel, kernel_size=kernel_size, padding=padding, groups=in_channel)
    def forward(self, xr, xi):
        ar, ai = self.attn_dw(xr, xi)
        ar, ai = self.attn_pw(ar, ai)
        xr, xi = self.attn_conv(xr, xi)
        yr = xr * ar - xi * ai
        yi = xi * ar + xr * ai
        return yr, yi

class ComplexSqueezeExcitation(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel

        self.pool_r = nn.AdaptiveAvgPool1d(1)
        self.pool_i = nn.AdaptiveAvgPool1d(1)
        self.conv = ComplexConv1d(in_channel, in_channel, kernel_size=1, padding=0, groups=1, bias=True)

    def forward(self, xr, xi):
        yr = self.pool_r(xr)
        yi = self.pool_i(xi)
        mr, mi = self.conv(yr, yi)

        yr = xr*mr - xi*mi
        yi = xi*mr + xr*mi

        return yr, yi


class GCGFN(nn.Module):
    def __init__(self, in_channel=64):
        super().__init__()
        self.in_channel = in_channel
        self.expand_ratio = 4
        self.mid_channel = self.in_channel * self.expand_ratio
        self.kernel_size_list = [3, 11, 23, 31]

        self.proj_first = nn.Conv1d(self.in_channel, self.mid_channel, kernel_size=1, bias=False)
        self.proj_last = nn.Conv1d(self.mid_channel, self.in_channel, kernel_size=1, bias=False)
        self.norm = ComplexLayerNorm1d(self.in_channel, elementwise_affine=False)
        self.scale = nn.Parameter(torch.zeros((1, self.in_channel, 1)), requires_grad=True)

        self.attn_3 = ComplexSwiGLU1d(self.in_channel, 3, get_padding(3))
        self.attn_11 = ComplexSwiGLU1d(self.in_channel, 11, get_padding(11))
        self.attn_23 = ComplexSwiGLU1d(self.in_channel, 23, get_padding(23))
        self.attn_31 = ComplexSwiGLU1d(self.in_channel, 31, get_padding(31))

    def forward(self, xr, xi):
        skip_r, skip_i = xr, xi
        xr, xi = self.norm(xr, xi)

        xr = self.proj_first(xr)
        xi = self.proj_first(xi)
        
        x1r, x2r, x3r, x4r = torch.chunk(xr, 4, dim=1)
        x1i, x2i, x3i, x4i = torch.chunk(xi, 4, dim=1)

        x1r, x1i = self.attn_3(x1r, x1i)
        x2r, x2i = self.attn_11(x2r, x2i)
        x3r, x3i = self.attn_23(x3r, x3i)
        x4r, x4i = self.attn_31(x4r, x4i)

        xr = torch.cat((x1r, x2r, x3r, x4r), dim=1)
        xi = torch.cat((x1i, x2i, x3i, x4i), dim=1)

        xr = self.proj_last(xr)
        xi = self.proj_last(xi)
        
        xr = xr * self.scale + skip_r
        xi = xi * self.scale + skip_i
        return xr, xi

class LKFCA_Block(nn.Module):
    def __init__(self, in_channels=64, DW_Expand=2):
        super().__init__()

        dw_channel = in_channels * DW_Expand
        self.norm1 = ComplexLayerNorm1d(in_channels, elementwise_affine=False)

        self.pwconv1 = ComplexConv1d(in_channels=in_channels, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.dwconv = ComplexConv1d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        
        self.sg = SimpleGate()

        self.cse = ComplexSqueezeExcitation(in_channels)

        self.pwconv2 = ComplexConv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        # Complex affine parameters: y = alpha * x + bias
        self.alpha_r = nn.Parameter(torch.zeros((1, in_channels, 1)), requires_grad=True)
        self.alpha_i = nn.Parameter(torch.zeros((1, in_channels, 1)), requires_grad=True)
        self.GCGFN = GCGFN(in_channels)


    def forward(self, xr, xi):

        skip1_r, skip1_i = xr, xi
        xr, xi = self.norm1(xr, xi)
        xr, xi = self.pwconv1(xr, xi)
        xr, xi = self.dwconv(xr, xi)
        xr, xi = self.sg(xr, xi)
        xr, xi = self.cse(xr, xi)
        xr, xi = self.pwconv2(xr, xi)

        yr = xr * self.alpha_r - xi * self.alpha_i
        yi = xi * self.alpha_r + xr * self.alpha_i
        xr = skip1_r + yr
        xi = skip1_i + yi
        xr, xi = self.GCGFN(xr, xi)

        return xr, xi

class TS_BLOCK(nn.Module):
    def __init__(self, dense_channel=64):
        super(TS_BLOCK, self).__init__()
        self.dense_channel = dense_channel
        self.time1 = LKFCA_Block(dense_channel)
        self.time2 = LKFCA_Block(dense_channel)
        self.freq1 = LKFCA_Block(dense_channel)
        self.freq2 = LKFCA_Block(dense_channel)
        self.beta_r = nn.Parameter(torch.zeros((1, 1, dense_channel, 1)), requires_grad=True)
        self.beta_i = nn.Parameter(torch.zeros((1, 1, dense_channel, 1)), requires_grad=True)
        self.gamma_r = nn.Parameter(torch.zeros((1, 1, dense_channel, 1)), requires_grad=True)
        self.gamma_i = nn.Parameter(torch.zeros((1, 1, dense_channel, 1)), requires_grad=True)
    def forward(self, xr, xi):
        b, c, t, f = xr.size()
        xr = xr.permute(0, 3, 1, 2).contiguous().view(b*f, c, t) 
        xi = xi.permute(0, 3, 1, 2).contiguous().view(b*f, c, t)

        skip_r, skip_i = xr, xi

        xr, xi = self.time1(xr, xi)
        xr, xi = self.time2(xr, xi)
        xr = xr + skip_r * self.beta_r - skip_i * self.beta_i
        xi = xi + skip_i * self.beta_r + skip_r * self.beta_i

        xr = xr.view(b, f, c, t).permute(0, 3, 2, 1).contiguous().view(b*t, c, f)
        xi = xi.view(b, f, c, t).permute(0, 3, 2, 1).contiguous().view(b*t, c, f)

        skip_r, skip_i = xr, xi

        xr, xi = self.freq1(xr, xi)
        xr, xi = self.freq2(xr, xi)
        xr = xr + skip_r * self.gamma_r - skip_i * self.gamma_i
        xi = xi + skip_i * self.gamma_r + skip_r * self.gamma_i

        xr = xr.view(b, t, c, f).permute(0, 2, 1, 3)
        xi = xi.view(b, t, c, f).permute(0, 2, 1, 3)
        return xr, xi

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
                    "act": ComplexGubermanModLeakyReLU2d(dense_channel, negative_slope=0.2, bias=True, init_bias=-0.05, mask_type='vonmises')
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
        self.pre_act = ComplexGubermanModLeakyReLU2d(dense_channel, negative_slope=0.2, bias=True, init_bias=-0.05, mask_type='vonmises')
        self.block = ComplexDSDDB(dense_channel, depth=4)
        self.post_conv = ComplexConv2d(dense_channel, dense_channel, (1, 3), stride=(1, 2), bias=True)
        self.post_norm = ComplexIN2d(dense_channel)
        self.post_act = ComplexGubermanModLeakyReLU2d(dense_channel, negative_slope=0.2, bias=True, init_bias=-0.05, mask_type='vonmises')

    def forward(self, xr, xi):
        xr, xi = self.pre_conv(xr, xi)
        xr, xi = self.pre_norm(xr, xi)
        xr, xi = self.pre_act(xr, xi)
        xr, xi = self.block(xr, xi)
        xr, xi = self.post_conv(xr, xi)
        xr, xi = self.post_norm(xr, xi)
        xr, xi = self.post_act(xr, xi)
        return xr, xi


class ComplexMaskDecoder(nn.Module):
    def __init__(self, 
                 dense_channel=64,
                 beta=2.0):
        super().__init__()
        self.block = ComplexDSDDB(dense_channel, depth=4)
        self.mask_convtr = ComplexConvTranspose2d(dense_channel, dense_channel, kernel_size=(1, 3), stride=(1, 2))
        self.mask_convpw1 = ComplexConv2d(dense_channel, 1, (1, 1), bias=True)
        self.beta = beta

    def forward(self, xr, xi):
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

        fr, fi = self.encoder(xr, xi)
        for blk in self.ts_blocks:
            fr, fi = blk(fr, fi)
        mr, mi = self.decoder(fr, fi)

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
