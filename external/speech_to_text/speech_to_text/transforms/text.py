# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import os
import string
import typing
import torch
import numpy as np
import youtokentome as yttm
from speech_to_text.transforms.utils import format_text, tokens_to_tensor
from speech_to_text.datasets import parse_librispeech_dataset


class TextTokenizerYTTM:
    eos_id = 3
    bos_id = 2
    unk_id = 1
    pad_id = 0

    def __init__(
            self,
            vocab_size: int = None,
            model_path: str = None,
            target_length: typing.Optional[int] = None,
            preprocess: bool = True,
            **kwargs
    ):
        if model_path is not None:
            self.tokenizer = yttm.BPE(model=model_path)
            self.is_initialized = True
        else:
            self.is_initialized = False
        if vocab_size is None and self.is_initialized:
            self.vocab_size = self.tokenizer.vocab_size()
        else:
            self.vocab_size = vocab_size
        self.target_length = target_length
        self.preprocess = preprocess

    def __len__(self):
        return self.vocab_size

    def vocab(self):
        self._is_initialized()
        return self.tokenizer.vocab()

    def state_dict(self):
        return {
            "vocab": self.vocab(),
            "space_symbol": "▁",
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "unk_id": self.unk_id,
            "pad_id": self.pad_id
        }

    def encode(self, text: str, dropout_prob: float = 0.0) -> torch.LongTensor:
        self._is_initialized()
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
        self._is_initialized()
        return self.tokenizer.decode(
            tokens.cpu().numpy().tolist(),
            ignore_ids=[self.eos_id, self.bos_id, self.unk_id, self.pad_id]
        )

    def itoc(self, i: int) -> str:
        self._is_initialized()
        return self.tokenizer.id_to_subword[i]

    def initialize(self, dataset, model_path, tmp_file="/tmp/yttm_corpus.txt", preprocess=True):
        from tqdm.auto import tqdm
        if os.path.isfile(model_path) and not force_train:
            print("tokenizer model is already exist")
        else:
            print(f"prepare temporary corpus file: {tmp_file}")
            with open(tmp_file, 'w') as f:
                for sample in dataset.samples:
                    text = format_text(sample["text"]) if preprocess else sample["text"]
                    f.write(text + "\n")
            print(f"training model {model_path} with vocab_size = {self.vocab_size}")
            yttm.BPE.train(
                data=tmp_file,
                vocab_size=self.vocab_size,
                model=model_path,
                pad_id = self.pad_id,
                unk_id = self.unk_id,
                bos_id = self.bos_id,
                eos_id = self.eos_id
            )
        self.tokenizer = yttm.BPE(model=model_path)
        self.is_initialized = True

    def _is_initialized(self):
        assert self.is_initialized, "tokenizer doesn't initialized"

    @classmethod
    def train(
            cls,
            data_path: typing.List[str],
            model_path: str,
            vocab_size: int = 80,
            preprocess: bool = True,
            tmp_file: str = "/tmp/yttm_corpus.txt",
            force_train: bool = False,
            **kwargs
    ):
        from tqdm.auto import tqdm
        if os.path.isfile(model_path) and not force_train:
            print("tokenizer model is already exist")
            return
        print(f"prepare temporary corpus file: {tmp_file}")
        with open(tmp_file, 'w') as f:
            # data = AudioDataset(data_path, load_audio=False)
            data = []
            for path in data_path:
                data.extend(parse_librispeech_dataset(path))
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
