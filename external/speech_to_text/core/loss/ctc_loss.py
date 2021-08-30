import torch
import torch.nn as nn


class CTCLoss(nn.Module):
    def __init__(self, blank_id):
        super().__init__()
        self.loss_fn = nn.CTCLoss(blank=blank_id)

    def forward(self, preds, gt, preds_lengths, gt_lengths):
        preds = preds.permute(2, 0, 1).log_softmax(dim=-1)
        loss = self.loss_fn(preds, gt, preds_lengths, gt_lengths)
        return {"loss": loss}
