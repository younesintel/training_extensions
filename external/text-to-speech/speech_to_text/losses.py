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
import torch
import torch.nn as nn


class CTCLoss(nn.Module):
    """Basic CTC loss for speech to text task."""
    def __init__(self, blank_id: int = 0):
        super().__init__()
        self.criterion = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)

    def forward(
            self,
            pred: torch.Tensor,
            gt: torch.Tensor,
            pred_lengths: torch.Tensor,
            gt_lengths: torch.Tensor
    ):
        pred = pred.permute(1, 0, 2).log_softmax(dim=-1)
        loss = self.criterion(pred, gt, pred_lengths, gt_lengths)
        return {"loss": loss}
