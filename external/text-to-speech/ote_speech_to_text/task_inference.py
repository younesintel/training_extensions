"""
OpenVINO Speech To Text Task
"""

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

import inspect
import json
import logging
import os
import struct
import subprocess  # nosec
import sys
import tempfile
from shutil import copyfile, copytree
from typing import Any, Dict, List, Optional, Union, cast
from zipfile import ZipFile

import numpy as np
from addict import Dict as ADDict
from omegaconf import ListConfig
from omegaconf.dictconfig import DictConfig
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.inference_parameters import (
    InferenceParameters,
    default_progress_callback,
)
from ote_sdk.entities.model import (
    ModelEntity,
    ModelFormat,
    ModelOptimizationType,
    ModelPrecision,
    ModelStatus,
    OptimizationMethod,
)
from ote_sdk.entities.optimization_parameters import OptimizationParameters
from ote_sdk.entities.resultset import ResultSetEntity
from ote_sdk.entities.task_environment import TaskEnvironment
from ote_sdk.serialization.label_mapper import LabelSchemaMapper, label_schema_to_bytes

from ote_sdk.usecases.tasks.interfaces.deployment_interface import IDeploymentTask
from ote_sdk.usecases.tasks.interfaces.inference_interface import IInferenceTask
from ote_sdk.usecases.tasks.interfaces.evaluate_interface import IEvaluationTask

import ote_speech_to_text.deploy as deploy
from ote_speech_to_text.deploy.openvino_speech_to_text import QuartzNet
from ote_speech_to_text.parameters import OTESpeechToTextTaskParameters
import ote_speech_to_text.utils as ote_utils
from speech_to_text.metrics import WordErrorRate
from tqdm.autonotebook import tqdm

logger = logging.getLogger(__name__)

class OpenVINOSpeechToTextTask(IInferenceTask, IEvaluationTask, IDeploymentTask):
    """
    OpenVINO inference task

    Args:
        task_environment (TaskEnvironment): task environment of the trained anomaly model
    """

    def __init__(self, task_environment: TaskEnvironment) -> None:
        logger.info("Loading OpenVINOSpeechToTextTask.")
        self.scratch_space = tempfile.mkdtemp(prefix="ote-stt-scratch-")
        logger.info(f"Scratch space created at {self.scratch_space}")
        self.task_environment = task_environment
        self.hparams = self.task_environment.get_hyper_parameters(OTESpeechToTextTaskParameters)
        self.inferencer = self.load_inferencer()
        self.metric = WordErrorRate()

    def env_data_file(self, name: str) -> str:
        """Get file path from environment."""
        path = os.path.join(self.scratch_space, name)
        if not os.path.exists(path):
            with open(os.path.join(self.scratch_space, name), "wb") as f:
                f.write(self.task_environment.model.get_data(name))
        return path

    def load_inferencer(self) -> QuartzNet:
        """
        Create the OpenVINO inferencer object

        Returns:
            QuartzNet object
        """
        self.env_data_file("openvino.bin")
        return QuartzNet(
            model_path = self.env_data_file("openvino.xml"),
            vocab_path = self.env_data_file("vocab.json")
        )

    def infer(self, dataset: DatasetEntity, inference_parameters: InferenceParameters) -> DatasetEntity:
        """Perform Inference.

        Args:
            dataset (DatasetEntity): Inference dataset
            inference_parameters (InferenceParameters): Inference parameters.

        Returns:
            DatasetEntity: Output dataset storing inference predictions.
        """
        if self.task_environment.model is None:
            raise Exception("task_environment.model is None. Cannot access threshold to calculate labels.")

        logger.info("Start OpenVINO inference")
        update_progress_callback = default_progress_callback
        if inference_parameters is not None:
            update_progress_callback = inference_parameters.update_progress

        for idx, dataset_item in tqdm(enumerate(dataset)):

            audio, sampling_rate = dataset_item.media.numpy
            text = self.inferencer(audio, sampling_rate)
            dataset_item.annotation_scene.append_annotations([ote_utils.text_to_annotation(text)])
            update_progress_callback(int((idx + 1) / len(dataset) * 100))
        return dataset

    def evaluate(self, output_resultset: ResultSetEntity, evaluation_metric: Optional[str] = None):
        """Evaluate the performance of the model.

        Args:
            output_resultset (ResultSetEntity): Result set storing ground truth and predicted dataset.
            evaluation_metric (Optional[str], optional): Evaluation metric. Defaults to None.
        """
        self.metric.reset()
        data = ote_utils.ote_extract_eval_annotation(
            output_resultset.prediction_dataset,
            output_resultset.ground_truth_dataset
        )
        for pred, tgt in zip(data["pred"], data["tgt"]):
            self.metric.update(pred, tgt)
        output_resultset.performance = self.metric.compute()

    def deploy(self, output_model: ModelEntity) -> None:
        """Exports the weights from ``output_model`` along with exportable code.

        Args:
            output_model (ModelEntity): Model with ``vocab.json``, ``openvino.xml`` and ``.bin`` keys

        Raises:
            Exception: If ``task_environment.model`` is None
        """
        logger.info("Deploying Model")

        if self.task_environment.model is None:
            raise Exception("task_environment.model is None. Cannot load weights.")

        work_dir = os.path.dirname(deploy.__file__)
        name_of_package = "demo_package"

        with tempfile.TemporaryDirectory() as tempdir:
            copytree(work_dir, os.path.join(tempdir, "package"))
            for datafile in ["openvino.xml", "openvino.bin", "vocab.json"]:
                copyfile(
                    os.path.join(self.env_data_file(datafile)),
                    os.path.join(tempdir, "package/openvino_speech_to_text/data/", datafile)
                )

            # create wheel package
            subprocess.run(
                [
                    sys.executable,
                    os.path.join(tempdir, "package/setup.py"),
                    "bdist_wheel",
                    "--dist-dir",
                    tempdir,
                    "clean",
                    "--all",
                ],
                check=True,
            )
            wheel_file_name = [f for f in os.listdir(tempdir) if f.endswith(".whl")][0]

            with ZipFile(os.path.join(tempdir, "openvino.zip"), "w") as arch:
                arch.write(os.path.join(tempdir, wheel_file_name), os.path.join("python", wheel_file_name))
            with open(os.path.join(tempdir, "openvino.zip"), "rb") as output_arch:
                output_model.exportable_code = output_arch.read()
        logger.info("Deploying completed")
