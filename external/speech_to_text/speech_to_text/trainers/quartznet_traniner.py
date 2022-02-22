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

import os
import json
# pytorch
import torch
import torch.nn as nn
import pytorch_lightning as pl
# numpy
import numpy as np
# tokenizer
import speech_to_text.utils as utils
# dataset
from speech_to_text.datasets import AudioDataset
# model
import speech_to_text.models as models
# loss
from speech_to_text.losses import CTCLoss
# optimizer
from speech_to_text.opt import NovoGrad, CosineAnnealingWithWarmupLR
# decoders
from speech_to_text.decoders import GreedyDecoder


class LightningQuartzNetTrainer(pl.LightningModule):
    """Training pipeline for QuartzNet"""
    def __init__(self, cfg, tokenizer, **kwargs):
        super().__init__()
        # config
        self.cfg = cfg
        # tokenizer
        self.tokenizer = tokenizer
        # model
        assert self.cfg.model.dtype == "QuartzNet"
        self.model = models.QuartzNet(
            vocab_size = self.tokenizer.vocab_size,
            **self.cfg.model.params
        )
        # criterion
        self.criterion = CTCLoss(blank_id=self.tokenizer.pad_id)
        # metrics
        self.metrics = utils.build_metrics(self.cfg.metrics)
        # decoder
        self.decoder = GreedyDecoder(
            tokenizer=self.tokenizer,
            blank_id=self.tokenizer.pad_id
        )

    def training_step(self, batch, batch_nb):
        preds = self.model(batch["audio"])
        output_lengths = torch.ceil(batch["audio_lengths"].float() / self.model.stride).int()
        loss = self.criterion(preds, batch["text"], output_lengths, batch["text_lengths"])
        self._log_dict(loss)
        # update learning rate for every step
        if self.lr_schedulers() is not None:
            self.lr_schedulers().step()
            self._log_dict({"lr": torch.tensor(self.lr_schedulers().get_last_lr()[0])})
        return {"loss": loss["loss"]}

    def validation_step(self, batch, batch_nb):
        preds = self.model(batch["audio"])
        output_lengths = torch.ceil(batch["audio_lengths"].float() / self.model.stride).int()
        loss = self.criterion(preds, batch["text"], output_lengths, batch["text_lengths"])
        tgt_strings = self.decoder.convert_to_strings(batch["text"], batch["text_lengths"], remove_repetitions=False)
        pred_strings = self.decoder.decode(preds, output_lengths, remove_repetitions=True)
        for i in range(len(tgt_strings)):
            self.metrics.update(pred_strings[i], tgt_strings[i])
        return {"loss_val": loss["loss"].item()}

    def validation_epoch_end(self, outputs):
        loss_val = np.mean([i["loss_val"] for i in outputs])
        self.log("loss_val", loss_val, prog_bar=True)
        self._log_dict(self.metrics.compute(), on_step=False, on_epoch=True)
        if self.logger is not None:
            self.logger.log_metrics(self.metrics.compute())
        self.metrics.reset()

    def test_step(self, batch, batch_nb):
        preds = self.model(batch["audio"])
        output_lengths = torch.ceil(batch["audio_lengths"].float() / self.model.stride).int()
        loss = self.criterion(preds, batch["text"], output_lengths, batch["text_lengths"])
        tgt_strings = self.decoder.convert_to_strings(batch["text"], batch["text_lengths"], remove_repetitions=False)
        pred_strings = self.decoder.decode(preds, output_lengths, remove_repetitions=True)
        for i in range(len(tgt_strings)):
            self.metrics.update(pred_strings[i], tgt_strings[i])
        return {"loss_val": loss["loss"].item()}

    def test_epoch_end(self, outputs):
        loss_val = np.mean([i["loss_val"] for i in outputs])
        self.log("loss_val", loss_val, prog_bar=True)
        self._log_dict(self.metrics.compute(), on_step=False, on_epoch=True)
        if self.logger is not None:
            self.logger.log_metrics(self.metrics.compute())
        self.metrics.reset()

    def predict_step(self, batch, batch_nb):
        preds = self.model(batch["audio"])
        output_lengths = torch.ceil(batch["audio_lengths"].float() / self.model.stride).int()
        pred_strings = self.decoder.decode(preds, output_lengths, remove_repetitions=True)
        return pred_strings

    def compute_metrics(self, outputs):
        self.metrics.reset()
        for pred, tgt in zip(outputs["pred"], outputs["tgt"]):
            self.metrics.update(pred, tgt)
        metrics = self.metrics.compute()
        self.metrics.reset()
        return metrics

    def configure_optimizers(self):
        opt = NovoGrad(
            self.model.parameters(),
            lr=self.cfg.optimizer.learning_rate,
            weight_decay=self.cfg.optimizer.weight_decay,
            betas=self.cfg.optimizer.betas
        )
        sch = None
        if self.cfg.optimizer.lr_scheduler:
            num_steps = self.cfg.optimizer.epoch_size * self.cfg.optimizer.epochs
            sch = CosineAnnealingWithWarmupLR(
                opt,
                T_max=num_steps,
                T_warmup=self.cfg.optimizer.warmup_steps,
                is_warmup=True
            )
        return [opt], [] if sch is None else [sch]

    def _log_dict(self, loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, exclude=["loss"]):
        for k, v in loss.items():
            if not k in exclude:
                self.log(k, v, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, logger=logger)

    def export(self, output_dir, sequence_length=128, output_name="model.onnx", to_openvino=False):
        # export vocab
        with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.tokenizer.state_dict(), f, indent=4, ensure_ascii=False)

        # set model to eval mode
        default_mode = self.model.training
        self.model.eval()
        # prepare random variable
        var = torch.randn(1, self.model.n_mels, sequence_length)
        # convert model to ONNX
        torch.onnx.export(
            self.model,
            (var,),
            os.path.join(output_dir, output_name),
            input_names = ['mel'],
            output_names = ['preds'],
            verbose=True
        )
        if to_openvino:
            utils.export_ir(os.path.join(output_dir, output_name), output_dir)
        if default_mode:
            self.model.train()
