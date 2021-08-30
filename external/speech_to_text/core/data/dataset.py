import torch
import torch.nn as nn
import torchaudio
from core.utils import *
from core.data.transforms import *
from torchvision.transforms import Compose


def load_librispeech(cfg, tokenizer, n_mels=80, mode="train"):
    if mode == "train":
        transforms = Compose([
            ApplyByKey(Compose([
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=n_mels),
                torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
                torchaudio.transforms.TimeMasking(time_mask_param=35),
                TensorSqueeze(0),
                TensorTranspose(0, 1)
            ]), "audio"),
            AddKey(tokenizer, "text", "tokens")
        ])
    else:
        transforms = Compose([
            ApplyByKey(Compose([
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=n_mels),
                TensorSqueeze(0),
                TensorTranspose(0, 1)
            ]), "audio"),
            AddKey(tokenizer, "text", "tokens")
        ])
    dataset = Librispeech(
        root = cfg.data.root,
        url = cfg.data.url,
        transforms = transforms,
        download = cfg.data.download
    )
    return dataset


class Librispeech(torchaudio.datasets.LIBRISPEECH):
    def __init__(self, transforms, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transforms = transforms

    def __getitem__(self, idx):
        audio, sample_rate, text, _, _, _ = super().__getitem__(idx)
        sample = {"audio": audio, "text": text}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample
