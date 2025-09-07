

import torch
import torch.nn as nn
import torch.nn.functional as F
import complextorch



def inv_sqrtm2x2(v_rr, v_ir, v_ii, eps):
    """
    Computes the inverse square root of a 2x2 symmetric matrix.
    The matrix is [[v_rr, v_ir], [v_ir, v_ii]].
    """
    # Add epsilon for numerical stability, especially to the diagonal
    v_rr = v_rr + eps
    v_ii = v_ii + eps

    # Calculate determinant and trace
    delta = v_rr * v_ii - v_ir * v_ir
    delta = torch.clamp(delta, min=eps**2)

    s = torch.sqrt(delta)
    tau = v_rr + v_ii
    t_arg = torch.clamp(tau + 2.0 * s, min=eps)
    t = torch.sqrt(t_arg)
    
    denom = s * t
    denom = torch.clamp(denom, min=eps)
    # Inverse square root matrix elements
    # M^(-1/2) = 1/(s*t) * [[v_ii + s, -v_ir], [-v_ir, v_rr + s]]
    coeff = 1.0 / denom
    p = coeff * (v_ii + s)  # M_11
    q = -coeff * v_ir       # M_12 and M_21
    r = coeff * (v_rr + s) # M_22

    return p, q, r


class ComplexLayerNorm1d(nn.Module):
    """
    Implements Complex-Valued Layer Normalization using einsum for robust, clear operations.
    """
    def __init__(self, channels, eps=1e-6, elementwise_affine=True):
        super(ComplexLayerNorm1d, self).__init__()
        self.channels = channels
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(2, 2, channels))
            self.bias = nn.Parameter(torch.empty(2, channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            identity = torch.eye(2).unsqueeze(-1).expand(2, 2, self.channels)
            self.weight.data.copy_(identity * 0.70710678118)
            nn.init.zeros_(self.bias)

    def forward(self, xr, xi):
        assert xr.dim() == 3, "Input tensor must have 3 dimensions (B, C, T)"
        
        # --- 1. Center the data ---
        # Stack real/imag parts: shape (B, C, T, 2)
        x = torch.stack([xr, xi], dim=-1)
        # Calculate mean over channel dim: shape (B, 1, T, 2)
        mu = x.mean(dim=1, keepdim=True)
        x = x - mu

        x_r_centered = x[..., 0]
        x_i_centered = x[..., 1]

        # --- 2. Calculate Covariance & Whitening Matrix ---
        v_rr = (x_r_centered ** 2).mean(dim=1, keepdim=True)
        v_ii = (x_i_centered ** 2).mean(dim=1, keepdim=True)
        v_ir = (x_r_centered * x_i_centered).mean(dim=1, keepdim=True)
        p, q, r = inv_sqrtm2x2(v_rr, v_ir, v_ii, self.eps)

        # --- 3. Apply Whitening ---
        z_real = p * x_r_centered + q * x_i_centered
        z_imag = q * x_r_centered + r * x_i_centered
        
        # --- 4. Apply Affine Transformation (if enabled) ---
        if self.elementwise_affine:
            z = torch.stack([z_real, z_imag], dim=-1)
            y = torch.einsum('ijc,bctj->bcti', self.weight, z)
            bias_reshaped = self.bias.permute(1, 0).view(1, self.channels, 1, 2)
            y = y + bias_reshaped
            y_real, y_imag = y[..., 0], y[..., 1]
            
            return y_real, y_imag
        else:
            return z_real, z_imag



class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_i = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    def forward(self, xr, xi):
        yr = self.conv_r(xr) - self.conv_i(xi)
        yi = self.conv_r(xi) + self.conv_i(xr)
        return yr, yi

class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, 
                    kernel_size=(1,1), stride=(1,1), dilation=(1, 1),
                    padding=(0,0), groups=1, bias=True):
        super().__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=bias)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=bias)

    def forward(self, xr, xi):
        yr = self.conv_r(xr) - self.conv_i(xi)
        yi = self.conv_r(xi) + self.conv_i(xr)
        return yr, yi



class ComplexConvTranspose2d(nn.Module):
    """Complex 2-D ConvTranspose following Cauchy rule.
    y = ConvT(x_r, W_r) - ConvT(x_i, W_i)  +  j[ ConvT(x_r, W_i) + ConvT(x_i, W_r) ]
    """
    def __init__(self, Cin, Cout, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1):
        super().__init__()
        self.t_r = nn.ConvTranspose2d(Cin, Cout, kernel_size, stride=stride, padding=padding,
                                      output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
        self.t_i = nn.ConvTranspose2d(Cin, Cout, kernel_size, stride=stride, padding=padding,
                                      output_padding=output_padding, groups=groups, bias=bias, dilation=dilation)
    def forward(self, xr, xi):
        yr = self.t_r(xr) - self.t_i(xi)
        yi = self.t_r(xi) + self.t_i(xr)
        return yr, yi


class ComplexIN2d(nn.Module):
    """Simple RI-RMS normalisation shared for real/imag pair."""
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.eps = eps

    def forward(self, xr, xi):
        # compute channel-wise rms
        rms = torch.sqrt(torch.mean(xr ** 2 + xi ** 2, dim=[2, 3], keepdim=True) + self.eps)
        xr = (xr / rms) * self.gamma + self.beta
        xi = (xi / rms) * self.gamma + self.beta  # same affine params
        return xr, xi


class ComplexLeakyModReLU1d(nn.Module):
    def __init__(self, C, negative_slope=0.2, bias=True, eps=1e-8):
        super().__init__()
        self.negative_slope = negative_slope
        self.eps = eps
        self.b = nn.Parameter(torch.zeros(1, C, 1)) if bias else None
    def forward(self, xr, xi):
        r = torch.sqrt(xr * xr + xi * xi + self.eps)
        t = r + (self.b if self.b is not None else 0.0)
        a = F.leaky_relu(t, negative_slope=self.negative_slope)
        s = a / (r + self.eps)
        return s * xr, s * xi

class ComplexLeakyModReLU2d(nn.Module):
    """Magnitude-gated leaky ReLU that preserves phase.
    y = (LReLU(|z| + b) / (|z| + eps)) * z
    """
    def __init__(self, C, negative_slope=0.2, bias=True, init_bias = 0.0, eps=1e-8):
        super().__init__()
        self.negative_slope = negative_slope
        self.eps = eps
        self.b = nn.Parameter(torch.zeros(1, C, 1, 1) + init_bias) if bias else None
    def forward(self, xr, xi):
        r = torch.sqrt(xr * xr + xi * xi + self.eps)
        t = r + (self.b if self.b is not None else 0.0)
        a = F.leaky_relu(t, negative_slope=self.negative_slope)
        s = a / (r + self.eps)
        return s * xr, s * xi

class ComplexModPReLU2d(nn.Module):
    def __init__(self, C, negative_slope=0.2, bias=True, init_bias = 0.0, eps=1e-8):
        super().__init__()
        self.negative_slope = negative_slope
        self.eps = eps
        self.b = nn.Parameter(torch.zeros(1, C, 1, 1) + init_bias) if bias else None
        self.act = nn.PReLU(C)
    def forward(self, xr, xi):
        r = torch.sqrt(xr * xr + xi * xi + self.eps)
        t = r + (self.b if self.b is not None else 0.0)
        a = self.act(t)
        s = a / (r + self.eps)
        return s * xr, s * xi


class zReLU(nn.Module):
    def __init__(self) -> None:
        super(zReLU, self).__init__()

    def forward(self, xr, xi):
        x_angle = torch.atan2(xi, xr)
        mask = (0 <= x_angle) & (x_angle <= torch.pi / 2)
        return xr * mask, xi * mask
 

class ComplexGubermanModLeakyReLU2d(nn.Module):
    """Magnitude-gated leaky ReLU with angular gating in the complex plane.
    - Preserves phase by scaling real/imag with the same factor.
    - Replaces hard quadrant mask with configurable leaky angular masks.
    """
    def __init__(self, C, negative_slope=0.2, bias=True, init_bias=0.0, eps=1e-8,
                 mask_type='hard', alpha=0.1, delta=0.2, theta0=None, beta=4.0,
                 learnable_alpha=False, learnable_theta0=False, learnable_beta=False, learnable_delta=False):
        super().__init__()
        self.negative_slope = negative_slope
        self.eps = eps
        self.b = nn.Parameter(torch.zeros(1, C, 1, 1) + init_bias) if bias else None
        self.mask_type = mask_type
        if theta0 is None:
            theta0 = float(torch.pi / 4)
        shape = (1, C, 1, 1)
        # alpha in [0,1]
        if learnable_alpha:
            self.alpha = nn.Parameter(torch.full(shape, float(alpha)))
        else:
            self.register_buffer('alpha', torch.full((1, 1, 1, 1), float(alpha)))
        # delta > 0
        if learnable_delta:
            self.delta = nn.Parameter(torch.full(shape, float(delta)))
        else:
            self.register_buffer('delta', torch.full((1, 1, 1, 1), float(delta)))
        # theta0 center angle
        if learnable_theta0:
            self.theta0 = nn.Parameter(torch.full(shape, float(theta0)))
        else:
            self.register_buffer('theta0', torch.full((1, 1, 1, 1), float(theta0)))
        # beta sharpness > 0
        if learnable_beta:
            self.beta = nn.Parameter(torch.full(shape, float(beta)))
        else:
            self.register_buffer('beta', torch.full((1, 1, 1, 1), float(beta)))

    def _angular_mask(self, theta):
        """Compute angular mask in [alpha, 1] depending on mask_type."""
        if self.mask_type == 'hard':
            inside = ((theta >= 0.0) & (theta <= torch.pi / 2)).to(theta.dtype)
            return inside
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        if self.mask_type == 'const':
            inside = ((theta >= 0.0) & (theta <= torch.pi / 2)).to(theta.dtype)
            return alpha + (1.0 - alpha) * inside
        if self.mask_type == 'cosine':
            delta = torch.clamp(self.delta, min=self.eps)
            d_left = torch.clamp(-theta, min=0.0)
            d_right = torch.clamp(theta - (torch.pi / 2), min=0.0)
            d = d_left + d_right
            l_inside = (d < 1e-12).to(theta.dtype)
            # Cosine ramp for 0 < d < delta
            ramp = 0.5 * (1.0 + torch.cos(torch.pi * torch.clamp(d / delta, max=1.0)))
            l = torch.where(d < delta, ramp, torch.zeros_like(theta))
            l = torch.where(d < 1e-12, torch.ones_like(theta), l)
            return alpha + (1.0 - alpha) * l
        if self.mask_type == 'vonmises':
            beta = torch.clamp(self.beta, min=0.0)
            theta0 = self.theta0
            logits = beta * torch.cos(theta - theta0)
            return alpha + (1.0 - alpha) * torch.sigmoid(logits)
        # default fallback: const
        inside = ((theta >= 0.0) & (theta <= torch.pi / 2)).to(theta.dtype)
        return alpha + (1.0 - alpha) * inside

    def forward(self, xr, xi):
        # Magnitude gating (Guberman-style leaky ModReLU)
        x_mag = torch.sqrt(xr * xr + xi * xi + self.eps)
        t = x_mag + (self.b if self.b is not None else 0.0)
        a = F.leaky_relu(t, negative_slope=self.negative_slope)
        s = a / (x_mag + self.eps)
        # Angular gating
        x_angle = torch.atan2(xi, xr)
        mask = self._angular_mask(x_angle)
        return s * xr * mask, s * xi * mask


class CVCardiod(nn.Module):
    def __init__(self) -> None:
        super(CVCardiod, self).__init__()

    def forward(self, xr, xi):
        x_angle = torch.atan2(xi, xr)
        return 0.5 * (1 + torch.cos(x_angle)) * xr, 0.5 * (1 + torch.cos(x_angle)) * xi