import torch
import torch.nn as nn


class CTCLoss(nn.Module):
    def __init__(self, blank_id=0):
        super().__init__()
        self.criterion = nn.CTCLoss(blank=blank_id, reduction="mean", zero_infinity=True)

    def forward(self, pred, gt, pred_lengths, gt_lengths):
        pred = pred.permute(1, 0, 2).log_softmax(dim=-1)
        loss = self.criterion(pred, gt, pred_lengths, gt_lengths)
        return {"loss": loss}
