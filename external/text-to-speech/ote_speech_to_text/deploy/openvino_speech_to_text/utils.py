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
from .quartznet_engine import QuartzNet


def create_model(model_path: str = None, vocab_path: str = None, device: str = "CPU") -> QuartzNet:
    """ Create Speech To Text inferencer.

    Arguments:
        model_path (str): path to model's .xml file (Optional).
        vocab_path (str): path to .json vocab file (Optional).
        device (str): target device (Optional).

    Returns:
        model (QuartzNet): speech to text inference engine.
    """
    if model_path is None or vocab_path is None:
        return create_default_model(device)
    return QuartzNet(
        model_path = model_path,
        vocab_path = vocab_path,
        device = device
    )

def create_default_model(device: str = "CPU") -> QuartzNet:
    """ Create default Speech To Text inferencer.

    Arguments:
        device (str): target device (Optional).

    Returns:
        model (QuartzNet): speech to text inference engine.
    """
    model_path = res_path("data/openvino.xml")
    vocab_path = res_path("data/vocab.json")
    return create_model(model_path, vocab_path, device)


def get_module_path() -> str:
    """ Get module path
    Returns:
        path (str): path to current module.
    """
    file_path = os.path.abspath(__file__)
    module_path = os.path.dirname(file_path)
    return module_path


def res_path(path: str) -> str:
    """ Resource path
    Arguments:
        path (str): related path from module dir to some resources.
    Returns:
        path (str): absolute path to module dir.
    """
    return os.path.join(get_module_path(), path)
