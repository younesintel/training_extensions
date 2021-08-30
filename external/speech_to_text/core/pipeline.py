import os
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
# tokenizer
from core.tokenizer import CharLevelTokenizer
# dataset
from core.data import load_librispeech
from core.data.collate_fn import CollateManager
# model
import core.models as models
# loss
from core.loss import CTCLoss
# utils
from core.utils import *
# decoder
from core.decoder import CTCDecoder
# metric
from core.metrics import WordErrorRate


class STTPipeline(pl.LightningModule):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        # config
        self.cfg = cfg
        # tokenizer
        self.tokenizer = CharLevelTokenizer()
        self.tokenizer.load_state_dict(load_cfg(cfg.tokenizer.model_path))
        # trainset
        self.trainset = load_librispeech(cfg.trainset, self.tokenizer, n_mels=cfg.model.params.channels_in, mode="train")
        # valset
        self.valset = load_librispeech(cfg.valset, self.tokenizer, n_mels=cfg.model.params.channels_in, mode="val")
        # model
        self.cfg.model.params.vocab_size = len(self.tokenizer)
        self.model = getattr(models, self.cfg.model.dtype)(**self.cfg.model.params)
        # loss
        self.loss = CTCLoss(blank_id=self.tokenizer.get_blank_symbol()[1])
        # decoder
        self.decoder = CTCDecoder(self.tokenizer)
        # metric
        self.metric = WordErrorRate()

    def training_step(self, batch, batch_nb, optimizer_idx=0):
        preds = self.model(batch["audio"])
        batch["audio_lengths"] = torch.div(batch["audio_lengths"], self.model.stride, rounding_mode='trunc')
        loss = self.loss(preds, batch["tokens"], batch["audio_lengths"], batch["tokens_lengths"])
        self._log_loss(loss)
        return {"loss": loss["loss"]}

    def validation_step(self, batch, batch_nb):
        preds = self.model(batch["audio"]).permute(0, 2, 1).log_softmax(dim=-1)
        strings = self.decoder(preds, batch["tokens_lengths"])
        for i in range(len(strings)):
            self.metric.update(strings[i], batch["text"][i])
        return {}

    def validation_epoch_end(self, outputs):
        metric = self.metric.compute()
        self.metric.reset()
        self.log(self.metric.get_name(), metric, prog_bar=True)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.cfg.trainset.batch_size,
            num_workers=self.cfg.pipeline.num_workers,
            collate_fn = CollateManager(),
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valset,
            batch_size=self.cfg.valset.batch_size,
            num_workers=self.cfg.pipeline.num_workers,
            collate_fn = CollateManager(),
            shuffle=True
        )

    def configure_optimizers(self):
        opt = optim.AdamW(
            self.model.parameters(),
            self.cfg.optimizer.learning_rate
        )
        num_steps = len(self.trainset) * self.cfg.optimizer.epochs
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_steps)
        return [opt], [sch]

    def _log_loss(self, loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, exclude=["loss"]):
        for k, v in loss.items():
            if not k in exclude:
                self.log(k, v, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, logger=logger)
