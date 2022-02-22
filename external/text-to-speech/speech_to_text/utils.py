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

from subprocess import run, DEVNULL, CalledProcessError
from collections import OrderedDict
import json
import torch
from addict import Dict
import speech_to_text.transforms as transforms
import speech_to_text.metrics as metrics
from speech_to_text.datasets import AudioDataset, parse_librispeech_dataset


def build_dataset(data_path, ext_audio: str = ".flac", ext_text: str = ".trans.txt", load_audio=True, load_text=True, **kwargs):
    if not isinstance(data_path, list):
        data_path = [data_path]
    samples = []
    for path in data_path:
        samples.extend(parse_librispeech_dataset(path, ext_audio, ext_text))
    return AudioDataset(samples, load_audio=load_audio, load_text=load_text)


def build_audio_transforms(cfg):
    def build_transform(name, params):
        if hasattr(transforms, name):
            return getattr(transforms, name)(**params)
        else:
            return None
    out = []
    for p in cfg:
        t = build_transform(p["name"], p["params"])
        if t is not None:
            out.append(t)
    return transforms.AudioCompose(out)


def build_metrics(names):
    out = []
    for name in names:
        if hasattr(metrics, name):
            out.append(getattr(metrics, name)())
    return metrics.MetricAggregator(out)


def build_tokenizer(data_path: str, model_path: str, vocab_size: int) -> transforms.TextTokenizerYTTM:
    transforms.TextTokenizerYTTM.train(
        data_path=data_path,
        model_path=model_path,
        vocab_size=vocab_size
    )
    return transforms.TextTokenizerYTTM(model_path=model_path, vocab_size=vocab_size)


def build_dataloader(dataset, tokenizer, audio_transforms_cfg, batch_size, num_workers, shuffle):
    dataset.tokenizer = tokenizer
    dataset.audio_transforms = build_audio_transforms(audio_transforms_cfg)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        collate_fn = AudioDataset.get_collate_fn(
            audio_pad_id = 0,
            text_pad_id = tokenizer.pad_id if tokenizer is not None else 0
        ),
        shuffle = shuffle
    )


def extract_annotation(dataset):
    annotation = []
    for sample in dataset:
        text = sample["text"]
        if dataset.tokenizer is not None:
            text = dataset.tokenizer.decode(text)[0]
        annotation.append(text)
    return annotation


def load_cfg(path):
    with open(path) as stream:
        cfg = Dict(json.load(stream))
    return cfg


def load_weights(target, source_state):
    new_dict = OrderedDict()
    for k, v in target.state_dict().items():
        if k in source_state and v.size() == source_state[k].size():
            new_dict[k] = source_state[k]
        elif k in source_state and v.size() != source_state[k].size():
            print(f"src: {source_state[k].size()}, tgt: {v.size()}")
            new_dict[k] = v
        else:
            print(f"key {k} not loaded...")
            new_dict[k] = v
    target.load_state_dict(new_dict)


def export_ir(onnx_model_path, optimized_model_dir='./ir_model', input_shape=None, data_type='FP32'):
    def get_mo_cmd():
        for mo_cmd in ('mo', 'mo.py'):
            try:
                run([mo_cmd, '-h'], stdout=DEVNULL, stderr=DEVNULL, shell=False, check=True)
                return mo_cmd
            except CalledProcessError:
                pass
        raise RuntimeError('OpenVINO Model Optimizer is not found or configured improperly')
    mo_cmd = get_mo_cmd()
    command_line = [mo_cmd, f'--input_model={onnx_model_path}',
                    f'--output_dir={optimized_model_dir}',
                    '--data_type', f'{data_type}']
    if input_shape:
        command_line.extend(['--input_shape', f"{input_shape}"])
    run(command_line, shell=False, check=True)
