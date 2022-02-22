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

import typing
import string
import torch
import numpy as np


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


class AudioCompose:
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, data, sample_rate):
        for t in self.transforms:
            data, sample_rate = t(data, sample_rate)
        return data, sample_rate


class ToNumpy:
    def __init__(self, flatten=False):
        self.flatten = flatten

    def __call__(self, data, sample_rate):
        if self.flatten:
            data = data.flatten()
        return data.numpy(), sample_rate


class ToTensor:
    def __call__(self, data, sample_rate):
        return torch.from_numpy(data), sample_rate
