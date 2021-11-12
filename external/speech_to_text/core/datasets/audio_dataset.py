import os
from typing import List, Callable, Optional, Tuple
from pathlib import Path
import torch
import torch.nn as nn
import torchaudio
from core.transforms import TextTokenizerYTTM


def load_librispeech_item(
        path_audio: str,
        path_text: str,
        speaker_id: str,
        chapter_id: str,
        utterance_id: str,
        load_audio: bool = True
) -> Tuple[torch.Tensor, int, str]:
    # Load audio
    if load_audio:
        waveform, sample_rate = torchaudio.load(path_audio)
    else:
        waveform, sample_rate = None, None

    # Load text
    fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
    with open(path_text) as ft:
        for line in ft:
            fileid_text, utterance = line.strip().split(" ", 1)
            if fileid_audio == fileid_text:
                break
        else:
            # Translation not found
            raise FileNotFoundError("Translation not found for " + fileid_audio)

    return waveform, sample_rate, utterance


class AudioDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_path: str,
            ext_audio: str = ".flac",
            ext_text: str = ".trans.txt",
            load_audio: bool = True,
            audio_transforms: Optional[List[Callable]] = None,
            tokenizer: Optional[TextTokenizerYTTM] = None,
            **kwargs
    ):
        self.ext_audio = ext_audio
        self.ext_text = ext_text
        if not isinstance(data_path, list):
            data_path = [data_path]
        self.samples = []
        for path in data_path:
            self.samples.extend(self._parse_folder(path))
        self.load_audio = load_audio
        self.audio_transforms = audio_transforms
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio, sample_rate, text = load_librispeech_item(
            *self.samples[idx],
            load_audio=self.load_audio
        )
        if self.audio_transforms is not None:
            audio, sample_rate = self.audio_transforms(audio, sample_rate)
        sample = {
            "audio": audio,
            "text": text,
            "sample_rate": sample_rate
        }
        if self.tokenizer is not None:
            sample["tokens"] = self.tokenizer.encode(sample["text"])
        return sample

    def _parse_folder(self, path):
        walker = sorted(str(p.stem) for p in Path(path).glob('*/*/*' + self.ext_audio))
        samples = []
        for fileid in walker:
            speaker_id, chapter_id, utterance_id = fileid.split("-")

            file_text = speaker_id + "-" + chapter_id + self.ext_text
            file_text = os.path.join(path, speaker_id, chapter_id, file_text)

            fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
            file_audio = fileid_audio + self.ext_audio
            file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)
            samples.append(
                (file_audio, file_text, speaker_id, chapter_id, utterance_id)
            )
        return samples

    @staticmethod
    def get_collate_fn(audio_pad_id, tokens_pad_id):
        def _collate_fn(batch):
            out = {
                "audio": [], "audio_lengths": [],
                "tokens": [], "tokens_lengths": [],
                "text": []
            }
            for sample in batch:
                audio = sample["audio"].squeeze(0).transpose(0, 1)
                out["audio"].append(audio)
                out["audio_lengths"].append(audio.shape[0])
                out["tokens"].append(sample["tokens"])
                out["tokens_lengths"].append(sample["tokens"].shape[0])
                out["text"].append(sample["text"])
            out["audio"] = nn.utils.rnn.pad_sequence(out["audio"], batch_first=True, padding_value=audio_pad_id)
            out["audio"] = out["audio"].transpose(1, 2)
            out["tokens"] = nn.utils.rnn.pad_sequence(out["tokens"], batch_first=True, padding_value=tokens_pad_id)
            out["audio_lengths"] = torch.LongTensor(out["audio_lengths"])
            out["tokens_lengths"] = torch.LongTensor(out["tokens_lengths"])
            return out
        return _collate_fn
