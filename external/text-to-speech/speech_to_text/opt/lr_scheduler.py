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

from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR


class CosineAnnealingWithWarmupLR(CosineAnnealingLR):
    """Cosine annealing learning rate scheduler."""
    def __init__(self, optimizer, T_warmup, T_max, eta_min=0, last_epoch=-1, is_warmup=True, verbose=False):
        assert T_warmup < T_max
        self.T_warmup = T_warmup
        self.is_warmup = is_warmup
        super().__init__(optimizer, T_max - T_warmup, eta_min, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.T_warmup and self.is_warmup:
            return [base_lr * self.last_epoch / self.T_warmup
                    for base_lr in self.base_lrs]
        else:
            self.switch_warmup()
            return super().get_lr()

    def switch_warmup(self):
        if self.is_warmup:
            self.last_epoch = -1
            self.is_warmup = False
