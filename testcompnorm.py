import torch
import torch.nn as nn
import unittest

# ===================================================================
# 테스트할 코드 (einsum을 사용한 최종 안정화 버전)
# ===================================================================

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

    def forward(self, x_tuple):
        x_real, x_imag = x_tuple
        assert x_real.dim() == 3, "Input tensor must have 3 dimensions (B, C, T)"
        
        # --- 1. Center the data ---
        # Stack real/imag parts: shape (B, C, T, 2)
        x = torch.stack([x_real, x_imag], dim=-1)
        # Calculate mean over channel dim: shape (B, 1, T, 2)
        mu = x.mean(dim=1, keepdim=True)
        x_centered = x - mu

        # --- 2. Calculate Covariance & Whitening Matrix via Eigendecomposition ---
        # Calculate covariance V using einsum. V shape: (B, T, 2, 2)
        V = torch.einsum('bcti,bctj->btij', x_centered, x_centered) / self.channels
        V = V + self.eps * torch.eye(2, device=x.device).expand_as(V)
        
        # Eigen-decomposition: V = Q * L * Q^T
        L_sq, Q = torch.linalg.eigh(V)
        
        # Calculate inverse square root of eigenvalues
        L_inv_sqrt = L_sq.pow(-0.5)
        
        # Reconstruct the whitening matrix V^(-1/2) = Q * L^(-1/2) * Q^T
        whitening_matrix = Q @ torch.diag_embed(L_inv_sqrt) @ Q.transpose(-2, -1)
        
        # --- 3. Apply Whitening ---
        # Apply whitening matrix using einsum. z shape: (B, C, T, 2)
        z = torch.einsum('bcti,btij->bctj', x_centered, whitening_matrix)
        
        # --- 4. Apply Affine Transformation ---
        if self.elementwise_affine:
            # Apply weight matrix using einsum. y shape: (B, C, T, 2)
            y = torch.einsum('bcti,ijc->bctj', z, self.weight)
            
            # Prepare and add bias
            # bias shape (2, C) -> (C, 2) -> (1, C, 1, 2) for broadcasting
            bias_v = self.bias.T.view(1, self.channels, 1, 2)
            y = y + bias_v
            
            y_real, y_imag = y[..., 0], y[..., 1]
            return y_real, y_imag
        else:
            z_real, z_imag = z[..., 0], z[..., 1]
            return z_real, z_imag

# ===================================================================
# unittest (기존과 동일, 수정 필요 없음)
# ===================================================================
class TestComplexLayerNorm1d(unittest.TestCase):
    def setUp(self):
        self.batch_size = 16
        self.channels = 32
        self.seq_len = 128
        self.real_input = torch.randn(self.batch_size, self.channels, self.seq_len)
        self.imag_input = torch.randn(self.batch_size, self.channels, self.seq_len)
        self.input_tuple = (self.real_input, self.imag_input)

    def test_output_shape(self):
        ln = ComplexLayerNorm1d(self.channels)
        out_real, out_imag = ln(self.input_tuple)
        self.assertEqual(self.real_input.shape, out_real.shape)
        self.assertEqual(self.imag_input.shape, out_imag.shape)
        print("✅ test_output_shape: Passed")

    def test_normalization_no_affine(self):
        ln = ComplexLayerNorm1d(self.channels, elementwise_affine=False)
        out_real, out_imag = ln(self.input_tuple)
        mean_real = out_real.mean(dim=1)
        mean_imag = out_imag.mean(dim=1)
        self.assertTrue(torch.allclose(mean_real, torch.zeros_like(mean_real), atol=1e-6))
        self.assertTrue(torch.allclose(mean_imag, torch.zeros_like(mean_imag), atol=1e-6))
        var_real = out_real.var(dim=1, unbiased=False)
        var_imag = out_imag.var(dim=1, unbiased=False)
        self.assertTrue(torch.allclose(var_real, torch.ones_like(var_real), atol=1e-5))
        self.assertTrue(torch.allclose(var_imag, torch.ones_like(var_imag), atol=1e-5))
        covar = (out_real * out_imag).mean(dim=1)
        self.assertTrue(torch.allclose(covar, torch.zeros_like(covar), atol=1e-4))
        print("✅ test_normalization_no_affine: Passed")

    def test_learning_capability(self):
        ln = ComplexLayerNorm1d(self.channels, elementwise_affine=True)
        optimizer = torch.optim.SGD(ln.parameters(), lr=0.1)
        target_real = torch.randn_like(self.real_input)
        target_imag = torch.randn_like(self.imag_input)
        initial_loss = None
        for i in range(5):
            optimizer.zero_grad()
            out_real, out_imag = ln(self.input_tuple)
            loss = ((out_real - target_real)**2 + (out_imag - target_imag)**2).mean()
            loss.backward()
            optimizer.step()
            if i == 0: initial_loss = loss.item()
        final_loss = loss.item()
        self.assertLess(final_loss, initial_loss)
        print("✅ test_learning_capability: Passed (Loss decreased)")

    def test_eval_mode(self):
        ln = ComplexLayerNorm1d(self.channels)
        ln.train()
        train_out_real, train_out_imag = ln(self.input_tuple)
        ln.eval()
        eval_out_real, eval_out_imag = ln(self.input_tuple)
        self.assertTrue(torch.equal(train_out_real, eval_out_real))
        self.assertTrue(torch.equal(train_out_imag, eval_out_imag))
        print("✅ test_eval_mode: Passed")

if __name__ == '__main__':
    unittest.main()