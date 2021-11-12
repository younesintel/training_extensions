from core.data.audio_dataset import AudioDataset
from core.transforms.spectrogram import *
from core.transforms.waveform import *
from torchvision.transforms import Compose

data = AudioDataset(data_path="/mnt/data_nvme/audio/data/LibriSpeech/dev-clean")
# t = Compose([
#     # WaveformTransformGain(),
#     NormalizedMelSpectrogram(mel_norm="uniform", n_mels=80),
#     MelSpecTransform(freq_max=5, time_max=50),
#     MelCutoutTransform(freq_max=10, time_max=50)
# ])

t = NormalizedMelSpectrogram(mel_norm="uniform", n_mels=80)

for sample in data:
    print(sample.keys())
    mel = t(sample["audio"], sample["sample_rate"])
    print(mel.size())
    break
