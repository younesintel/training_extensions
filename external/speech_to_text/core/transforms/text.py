import os
import string
import typing
import torch
import numpy as np
import youtokentome as yttm


def format_text(
        text: str,
        to_lowercase: bool = True,
        strip: bool = True,
        punctuation: str = string.punctuation + '—–«»−…‑'
) -> str:
    if to_lowercase:
        text = text.lower()
    if strip:
        text = text.strip()
    if punctuation is not None:
        text = text.translate(str.maketrans('', '', punctuation))
    return text


def tokens_to_tensor(
        tokens: typing.List[int],
        pad_id: int = 0,
        target_length: typing.Optional[int] = None,
) -> torch.LongTensor:
    if target_length is not None:
        pad_size = target_length - len(tokens)
        if pad_size > 0:
            tokens = np.hstack((tokens, np.full(pad_size, pad_id)))
        if len(tokens) > target_length:
            tokens = tokens[:target_length]
    return torch.tensor(tokens).long()


class TextTokenizerYTTM:
    eos_id = 3
    bos_id = 2
    unk_id = 1
    pad_id = 0

    def __init__(
            self,
            model_path: str,
            target_length: typing.Optional[int] = None,
            preprocess: bool = True,
            **kwargs
    ):
        self.tokenizer = yttm.BPE(model=model_path)
        self.target_length = target_length
        self.preprocess = preprocess

    def __len__(self):
        return self.vocab_size()

    def vocab_size(self):
        return self.tokenizer.vocab_size()

    def vocab(self):
        return self.tokenizer.vocab()

    def encode(self, text: str, dropout_prob: float = 0.0) -> torch.LongTensor:
        if self.preprocess:
            text = format_text(text)
        tokens = self.tokenizer.encode(
            [text],
            output_type=yttm.OutputType.ID,
            dropout_prob=dropout_prob
        )[0]
        tokens = [self.bos_id] + tokens + [self.eos_id]
        return tokens_to_tensor(tokens, self.pad_id, self.target_length)

    def decode(self, tokens: torch.LongTensor) -> typing.List[str]:
        return self.tokenizer.decode(
            tokens.cpu().numpy().tolist(),
            ignore_ids=[self.eos_id, self.bos_id, self.unk_id, self.pad_id]
        )

    def itoc(self, i: int) -> str:
        return self.tokenizer.id_to_subword[i]

    @classmethod
    def train(
            cls,
            data_path: str,
            model_path: str,
            vocab_size: int = 80,
            preprocess: bool = True,
            tmp_file: str = "/tmp/yttm_corpus.txt",
            force_train: bool = False,
            **kwargs
    ):
        from core.datasets.audio_dataset import AudioDataset
        from tqdm.auto import tqdm
        if os.path.isfile(model_path) and not force_train:
            print("tokenizer model is already exist")
            return
        print(f"prepare temporary corpus file: {tmp_file}")
        with open(tmp_file, 'w') as f:
            data = AudioDataset(data_path, load_audio=False)
            for sample in tqdm(data):
                text = format_text(sample["text"]) if preprocess else sample["text"]
                f.write(text + "\n")
        print(f"training model {model_path} with vocab_size = {vocab_size}")
        yttm.BPE.train(
            data=tmp_file,
            vocab_size=vocab_size,
            model=model_path,
            pad_id = cls.pad_id,
            unk_id = cls.unk_id,
            bos_id = cls.bos_id,
            eos_id = cls.eos_id
        )
