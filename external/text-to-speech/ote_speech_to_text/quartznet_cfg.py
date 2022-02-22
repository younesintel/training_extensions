"""
 Copyright (c) 2020-2021 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import os
import typing
from addict import Dict

# pylint: disable=protected-access,too-many-statements,unspecified-encoding

def get_quartznet_cfg() -> typing.Dict:
    """Default configuration of QuartzNet training pipeline."""
    cfg = {
        "tokenizer": {
            "vocab_size": 80
        },
        "audio_transforms": {
            "train": [
                {"name": "ToNumpy", "params": {"flatten": True}},
                {"name": "MelSpectrogram", "params": {"n_mels": 64}},
                {"name": "ToTensor", "params": {}},
                {"name": "MelSpecTransform", "params": {"freq_max": 1, "time_max": 1}},
                {"name": "MelCutoutTransform", "params": {"holes": 1, "freq_max": 30, "time_max": 30}}
            ],
            "val": [
                {"name": "ToNumpy", "params": {"flatten": True}},
                {"name": "MelSpectrogram", "params": {"n_mels": 64}},
                {"name": "ToTensor", "params": {}}
            ]
        },
        "metrics": ["CharacterErrorRate", "WordErrorRate"],
        "model": {
            "dtype": "QuartzNet",
            "params": {
                "cfg": [
                    {"channels_out": 256, "kernel_size": 33, "stride": 2, "dilation": 1, "groups": 1, "repeat": 1, "n_blocks": 1, "separable": True},
                    {"channels_out": 256, "kernel_size": 33, "stride": 1, "dilation": 1, "groups": 1, "repeat": 5, "n_blocks": 1, "separable": True},
                    {"channels_out": 256, "kernel_size": 39, "stride": 1, "dilation": 1, "groups": 1, "repeat": 5, "n_blocks": 1, "separable": True},
                    {"channels_out": 512, "kernel_size": 51, "stride": 1, "dilation": 1, "groups": 1, "repeat": 5, "n_blocks": 1, "separable": True},
                    {"channels_out": 512, "kernel_size": 63, "stride": 1, "dilation": 1, "groups": 1, "repeat": 5, "n_blocks": 1, "separable": True},
                    {"channels_out": 512, "kernel_size": 75, "stride": 1, "dilation": 1, "groups": 1, "repeat": 5, "n_blocks": 1, "separable": True},
                    {"channels_out": 512, "kernel_size": 87, "stride": 1, "dilation": 2, "groups": 1, "repeat": 1, "n_blocks": 1, "separable": True},
                    {"channels_out": 1024, "kernel_size": 1, "stride": 1, "dilation": 1, "groups": 1, "repeat": 1, "n_blocks": 1, "separable": False}
                ],
                "n_mels": 64
            }
        },
        "optimizer": {
            "optimizer": "Adam",
            "learning_rate": 0.0005,
            "weight_decay": 0.0001,
            "betas": [0.95, 0.5],
            "epochs": 260,
            "lr_scheduler": True,
            "warmup_steps": 0
        },
        "pipeline": {
            "monitor": "wer",
            "monitor_mode": "min",
            "num_workers": 0,
            "grad_batches": 3,
            "distributed_backend": "ddp",
            "val_check_interval": 500,
            "gradient_clip_val": 15
        }
    }
    return Dict(cfg)
