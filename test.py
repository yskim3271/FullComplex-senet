import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

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


def test_dataset():
    import datasets
    from torch.utils.data import DataLoader
    from data import mag_pha_istft

    from data import VoiceBankDataset
    dataset = datasets.load_dataset("JacobLinCool/VoiceBank-DEMAND-16k")
    # dataset_train = VoiceBankDataset(
    #     datapair_list=dataset['test'],
    #     sampling_rate=16000,
    #     segment=32000,
    #     type="train"
    # )
    dataset_valid = VoiceBankDataset(
        datapair_list=dataset['test'],
        sampling_rate=16000,
        segment=32000,  
        compress_factor=0.3,
        type="valid"
    )
    # dataset_test = VoiceBankDataset(
    #     datapair_list=dataset['test'],
    #     sampling_rate=16000,
    #     type="test"
    # )

    # dataloader = DataLoader(dataset_train, batch_size=1, shuffle=True)
    # for noisy, clean in dataloader:
    #     print("Train")
    #     print(noisy['wav'].shape)
    #     print(clean['wav'].shape)
    #     print(noisy['magnitude'].shape)
    #     print(clean['magnitude'].shape)
    #     print(noisy['phase'].shape)
    #     print(clean['phase'].shape)
    #     print(noisy['length'])
    #     break
    dataloader = DataLoader(dataset_valid, batch_size=1, shuffle=False)
    for i, (noisy, clean) in enumerate(dataloader):
        length = noisy['length']
        noisy_wav_stft = mag_pha_istft(noisy['magnitude'].squeeze(0), noisy['phase'].squeeze(0), 400, 100, 400, 0.3)
        
        segments, segment_size = noisy_wav_stft.shape

        if length <= segment_size:
            noisy_wav = noisy_wav_stft[:, :length]
        else:
            noisy_wav = noisy_wav_stft.view(1, -1)
            noisy_wav = torch.cat([
                noisy_wav[:, :(segments - 2) * segment_size + length % segment_size],
                noisy_wav[:, -segment_size:]
            ], dim=1)
        torchaudio.save(f"noisy_wav_ref_{i}.wav", noisy['wav'].squeeze(0), 16000)
        torchaudio.save(f"noisy_wav_{i}.wav", noisy_wav, 16000)


        if i > 10:
            break


    # print("Test")
    # dataloader = DataLoader(dataset_test, batch_size=1, shuffle=True)
    # for noisy, clean in dataloader:
    #     print("Test")
    #     print(noisy['wav'].shape)
    #     print(clean['wav'].shape)
    #     print(noisy['magnitude'].shape)
    #     print(clean['magnitude'].shape)
    #     print(noisy['phase'].shape)
    #     print(noisy['phase'].shape)
    #     print(noisy['length'])
    #     break


def test_compute_metrics():
    from compute_metrics import compute_metrics
    clean = torch.randn(16000).detach().cpu().numpy()
    enhanced = torch.randn(16000).detach().cpu().numpy()
    metrics = compute_metrics(clean, enhanced, Fs=16000, path=False)
    print(metrics)

def test_primeknet():
    from models.primeKnet import PrimeKnet
    model = PrimeKnet(
        fft_len=400,
        dense_channel=64,
        sigmoid_beta=2,
        num_tsblock=4
    )
    x = dict(
        magnitude=torch.randn(1, 201, 400),
        phase=torch.randn(1, 201, 400)
    )
    y = model(x)

def test_TFconv():
    import torch.nn as nn
    x = torch.randn(1, 1, 5, 5)
    conv = nn.Conv2d(1, 1, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=1, bias=False)
    y = conv(x)
    print(f"x: {x}")
    print(f"y: {y}")
    print(f"weight: {conv.weight}")

def test_primeknetv6():
    from models.primeKnetv6 import PrimeKnetv6
    model = PrimeKnetv6(
        fft_len=400,
        dense_channel=64,
        sigmoid_beta=2,
    )
    x = dict(
        magnitude=torch.randn(1, 201, 400),
        phase=torch.randn(1, 201, 400)
    )
    y = model(x)

def test_primeknetv3():
    from models.primeKnetv3 import PrimeKnetv3
    model = PrimeKnetv3(
        fft_len=400,
        dense_channel=64,
        sigmoid_beta=2,
        num_tsblock=4
    )
    x = dict(
        magnitude=torch.randn(1, 201, 400),
        phase=torch.randn(1, 201, 400)
    )
    y = model(x)

def test_layernorm_vs_instancenorm():
    """
    Compare LayerNorm1d with InstanceNorm to analyze their differences
    """
    print("=== LayerNorm1d vs InstanceNorm Comparison ===")
    
    # Test parameters
    batch_size = 2
    channels = 64
    time_steps = 100
    
    # Create test input
    x = torch.randn(batch_size, channels, time_steps)
    print(f"Input shape: {x.shape}")
    print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")
    
    # Initialize LayerNorm1d
    layer_norm = LayerNorm1d(channels)
    
    # Initialize InstanceNorm1d
    instance_norm = nn.InstanceNorm1d(channels, affine=True)
    
    # Forward pass
    with torch.no_grad():
        out_layer = layer_norm(x)
        out_instance = instance_norm(x)
    
    print("\n--- Output Statistics ---")
    print(f"LayerNorm1d output - mean: {out_layer.mean():.4f}, std: {out_layer.std():.4f}")
    print(f"InstanceNorm1d output - mean: {out_instance.mean():.4f}, std: {out_instance.std():.4f}")
    
    # Compare per-sample statistics
    print("\n--- Per-Sample Statistics ---")
    for i in range(batch_size):
        layer_sample = out_layer[i]
        instance_sample = out_instance[i]
        
        print(f"Sample {i}:")
        print(f"  LayerNorm1d - mean: {layer_sample.mean():.4f}, std: {layer_sample.std():.4f}")
        print(f"  InstanceNorm1d - mean: {instance_sample.mean():.4f}, std: {instance_sample.std():.4f}")
    
    # Compare per-channel statistics
    print("\n--- Per-Channel Statistics (first 5 channels) ---")
    for c in range(min(5, channels)):
        layer_channel = out_layer[:, c, :]
        instance_channel = out_instance[:, c, :]
        
        print(f"Channel {c}:")
        print(f"  LayerNorm1d - mean: {layer_channel.mean():.4f}, std: {layer_channel.std():.4f}")
        print(f"  InstanceNorm1d - mean: {instance_channel.mean():.4f}, std: {instance_channel.std():.4f}")
    
    # Compute difference between outputs
    diff = torch.abs(out_layer - out_instance)
    print(f"\n--- Difference Analysis ---")
    print(f"Mean absolute difference: {diff.mean():.4f}")
    print(f"Max absolute difference: {diff.max():.4f}")
    print(f"Min absolute difference: {diff.min():.4f}")
    
    # Test with gradient computation
    print("\n--- Gradient Test ---")
    x_grad = torch.randn(batch_size, channels, time_steps, requires_grad=True)
    
    # LayerNorm1d gradient
    layer_norm_grad = LayerNorm1d(channels)
    out_layer_grad = layer_norm_grad(x_grad)
    loss_layer = out_layer_grad.sum()
    loss_layer.backward()
    layer_grad = x_grad.grad.clone()
    
    # Reset gradient
    x_grad.grad.zero_()
    
    # InstanceNorm1d gradient
    instance_norm_grad = nn.InstanceNorm1d(channels, affine=True)
    out_instance_grad = instance_norm_grad(x_grad)
    loss_instance = out_instance_grad.sum()
    loss_instance.backward()
    instance_grad = x_grad.grad.clone()
    
    print(f"LayerNorm1d gradient - mean: {layer_grad.mean():.4f}, std: {layer_grad.std():.4f}")
    print(f"InstanceNorm1d gradient - mean: {instance_grad.mean():.4f}, std: {instance_grad.std():.4f}")
    
    grad_diff = torch.abs(layer_grad - instance_grad)
    print(f"Gradient difference - mean: {grad_diff.mean():.4f}, max: {grad_diff.max():.4f}")
    
    print("\n=== Test Complete ===\n")

if __name__ == "__main__":
    # test_ghostsenet()
    # test_ghostsenetv2()
    # test_dataset()
    # test_compute_metrics()
    # test_ghostsenet()
    # test_primeknet()
    # test_TFconv()
    # test_primeknetv3()
    # test_primeknetv4()
    test_primeknetv3()
    # test_layernorm_vs_instancenorm()