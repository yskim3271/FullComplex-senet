import torch
import torch.nn.functional as F
import torchaudio


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

def test_ghostsenetv2():
    from models.ghostsenetv2 import GhostSEnet
    model = GhostSEnet(
        fft_len=400,
        channel=64,
        sigmoid_beta=2,
        num_tsblock=4
    )
    x = dict(
        magnitude=torch.randn(1, 201, 400),
        phase=torch.randn(1, 201, 400)
    )

    y = model(x)
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = (total_params) / (1024 * 1024)
    print(f"Model size: {model_size_mb:.2f} MB")



if __name__ == "__main__":
    # test_ghostsenet()
    # test_ghostsenetv2()
    # test_dataset()
    # test_compute_metrics()
    test_ghostsenetv2()
    # test_primeknet()