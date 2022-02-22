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
from pytorch_lightning.callbacks import Callback


class StopCallback(Callback):
    """Stop training callback"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def on_batch_end(self, trainer, pl_module):
        trainer.should_stop = self.should_stop

    def stop(self):
        self.should_stop = True

    def reset(self):
        self.should_stop = False

    def check_stop(self):
        return self.should_stop
