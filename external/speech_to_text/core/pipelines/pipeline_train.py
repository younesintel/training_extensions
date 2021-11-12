import os
# pytorch
import torch
import torch.nn as nn
import pytorch_lightning as pl
# tokenizer
from core.transforms import TextTokenizerYTTM
# dataset
from core.datasets import AudioDataset
# model
import core.models as models
# loss
from core.loss import CTCLoss
# metrics
from core.metrics import MetricAggregator
# utils
from core.utils import get_metrics, get_audio_transforms
# optimizer
from core.opt import NovoGrad, CosineAnnealingWithWarmupLR


class PipelineTrain(pl.LightningModule):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()
        # config
        self.cfg = cfg
        # tokenizer
        self.tokenizer = self._get_tokenizer()
        print(self.tokenizer.vocab())
        exit()
        # datasets
        self.trainset = AudioDataset(
            tokenizer = self.tokenizer,
            audio_transforms = get_audio_transforms(self.cfg.audio_transforms.train),
            **self.cfg.trainset
        )
        self.valset = AudioDataset(
            tokenizer = self.tokenizer,
            audio_transforms = get_audio_transforms(self.cfg.audio_transforms.val),
            **self.cfg.valset
        )
        # model
        self.model = getattr(models, self.cfg.model.dtype)(
            vocab_size = self.tokenizer.vocab_size(),
            **self.cfg.model.params
        )
        # criterion
        self.criterion = CTCLoss(blank_id=self.tokenizer.pad_id)
        # metrics
        self.metrics = MetricAggregator(get_metrics(self.cfg.metrics))

    def training_step(self, batch, batch_nb):
        preds = self.model(batch["audio"])
        batch["audio_lengths"] = torch.div(batch["audio_lengths"], self.model.stride, rounding_mode='trunc')
        loss = self.criterion(preds, batch["tokens"], batch["audio_lengths"], batch["tokens_lengths"])
        self._log_dict(loss)
        # update learning rate for every step
        self.lr_schedulers().step()
        self._log_dict({"lr": torch.tensor(self.lr_schedulers().get_last_lr()[0])})
        return {"loss": loss["loss"]}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.cfg.trainset.batch_size,
            num_workers=self.cfg.pipeline.num_workers,
            collate_fn = AudioDataset.get_collate_fn(
                audio_pad_id=0,
                tokens_pad_id=self.tokenizer.pad_id
            ),
            shuffle=True
        )

    def configure_optimizers(self):
        opt = NovoGrad(
            self.model.parameters(),
            lr=self.cfg.optimizer.learning_rate,
            weight_decay=self.cfg.optimizer.weight_decay,
            betas=self.cfg.optimizer.betas
        )
        num_steps = len(self.trainset) * self.cfg.optimizer.epochs
        sch = CosineAnnealingWithWarmupLR(
            opt,
            T_max=num_steps,
            T_warmup=self.cfg.optimizer.warmup_steps,
            is_warmup=True
        )
        return [opt], [sch]

    def _log_dict(self, loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, exclude=["loss"]):
        for k, v in loss.items():
            if not k in exclude:
                self.log(k, v, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, logger=logger)

    def _get_tokenizer(self):
        TextTokenizerYTTM.train(
            data_path=self.cfg.trainset.data_path,
            **self.cfg.tokenizer
        )
        return TextTokenizerYTTM(**self.cfg.tokenizer)
