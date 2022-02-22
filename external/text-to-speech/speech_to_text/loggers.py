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
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment


class LightningSpeechToTextLogger(LightningLoggerBase):
    """Simple logger for pytorch lightning pipeline."""
    def __init__(self, monitor, mode="min"):
        super().__init__()
        assert mode in ["min", "max"]
        self.monitor_fn = min if mode == "min" else max
        self.monitor = monitor
        self.reset()

    @property
    def name(self):
        return "STTLogger"

    @rank_zero_only
    def log_hyperparams(self, params):
        self.params = params

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        if not self.monitor in metrics:
            return
        self.step = step if step is not None else self.step + 1
        self.log[self.step] = metrics
        if self.best_metric is None:
            self.best_metric = metrics[self.monitor]
        self.best_metric = self.monitor_fn(self.best_metric, metrics[self.monitor])

    def reset(self):
        self.log = {}
        self.best_metric = None
        self.params = None
        self.step = 0

    @property
    @rank_zero_experiment
    def experiment(self):
        pass

    @property
    def version(self):
        return "0.1"
