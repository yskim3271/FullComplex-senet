
from models.ghostsenet import GhostSEnet
from models.ghostsenetv2 import GhostSEnet as GhostSEnetV2
import torch


def test_ghostsenet():
    
    model = GhostSEnet(
        win_len=400,
        hop_len=100,
        fft_len=400,
        dense_channel=64,
        sigmoid_beta=2,
        compress_factor=0.3,
        num_tsblock=4
    )

    x = torch.randn(1, 16000)

    y = model(x)
    print(y.shape)
    
def test_ghostsenetv2():
    model = GhostSEnetV2(
        win_len=400,
        hop_len=100,
        fft_len=400,
        dense_channel=64,
        sigmoid_beta=2,
        compress_factor=0.3,
        num_tsblock=4
    )

    x = torch.randn(1, 16000)

    y = model(x)
    print(y.shape)

if __name__ == "__main__":
    # test_ghostsenet()
    test_ghostsenetv2()