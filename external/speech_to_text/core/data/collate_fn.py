import torch
import torch.nn as nn
from core.data.transforms import *
from torchvision.transforms import Compose


class CollateManager(object):
    def __init__(self, transforms=None):
        super().__init__()
        self.transforms = Compose([
            AddKey(ComputeLengths(), "audio", "audio_lengths"),
            AddKey(ComputeLengths(), "tokens", "tokens_lengths"),
            ApplyByKey(
                Compose([
                    PadSequence(),
                    TensorTranspose(1, 2)
                ]), "audio"),
            ApplyByKey(PadSequence(), "tokens")
        ])

    def __call__(self, batch):
        out = {"audio": [], "text": [], "tokens": []}
        for sample in batch:
            out["audio"].append(sample["audio"])
            out["text"].append(sample["text"])
            out["tokens"].append(sample["tokens"])
        if self.transforms is not None:
            out = self.transforms(out)
        return out
