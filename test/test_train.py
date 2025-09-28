from torch.utils.data import Dataset, DataLoader
import time
import torch
from torch import nn
import numpy as np
import os
import functools
from typing import List, Any
import itertools
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import matplotlib
import matplotlib.pyplot
import time

from trackertraincode.datasets.batch import Batch, Metadata
import trackertraincode.train as train


def test_plotter():
    plotter = train.TrainHistoryPlotter()
    names = ["foo", "bar", "baz", "lr"]
    for e in range(4):
        for t in range(5):
            for name in names[:-2]:
                plotter.add_train_point(e, t, name, 10.0 + e + np.random.normal(0.0, 1.0, (1,)))
        for name in names[1:]:
            plotter.add_test_point(e, name, 9.0 + e + np.random.normal())
        plotter.summarize_train_values()
        plotter.update_graph()
    plotter.close()


class MseLoss(object):
    def __call__(self, pred, batch):
        return torch.nn.functional.mse_loss(pred["test_head_out"], batch["y"], reduction="none")


class L1Loss(object):
    def __call__(self, pred, batch):
        return torch.nn.functional.l1_loss(pred["test_head_out"], batch["y"], reduction="none")


class CosineDataset(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x = torch.rand((1,))
        y = torch.cos(x)
        return Batch(Metadata(0, batchsize=0), {"image": x, "y": y})


class MockupModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor):
        return {"test_head_out": self.layers(x)}

    def get_config(self):
        return {}


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self._model = MockupModel()
        self._train_criterions = self.__setup_train_criterions()
        self._test_criterion = train.Criterion("test_head_out_c1", MseLoss(), 1.0)

    def __setup_train_criterions(self):
        c1 = train.Criterion("c1", MseLoss(), 0.42)
        c2 = train.Criterion("c2", L1Loss(), 0.7)
        return train.CriterionGroup([c1, c2], "test_head_out_")

    def training_step(self, batch: Batch, batch_idx):
        preds = self._model(batch["image"])
        loss_sum, all_lossvals = train.default_compute_loss(
            preds, [batch], self.current_epoch, self._train_criterions
        )
        loss_val_by_name = {
            name: val
            for name, (val, _) in train.concatenated_lossvals_by_name(
                itertools.chain.from_iterable(all_lossvals)
            ).items()
        }
        self.log("loss", loss_sum, on_epoch=True, prog_bar=True, batch_size=batch.meta.batchsize)
        return {"loss": loss_sum, "mt_losses": loss_val_by_name}

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor | dict[str, Any] | None:
        images = batch["image"]
        pred = self._model(images)
        values = self._test_criterion.evaluate(pred, batch, batch_idx)
        val_loss = torch.cat([(lv.val * lv.weight) for lv in values]).sum()
        self.log("val_loss", val_loss, on_epoch=True, batch_size=batch.meta.batchsize)
        return values

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1.0e-4, total_steps=50)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @property
    def model(self):
        return self._model


def test_train_smoketest(tmp_path):
    batchsize = 32
    epochs = 50
    train_loader = DataLoader(CosineDataset(20), batch_size=batchsize, collate_fn=Batch.collate)
    test_loader = DataLoader(CosineDataset(8), batch_size=batchsize, collate_fn=Batch.collate)
    model = LitModel()
    model_out_dir = os.path.join(tmp_path, "models")

    checkpoint_cb = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        monitor="val_loss",
        enable_version_counter=False,
        filename="best",
        dirpath=model_out_dir,
        save_weights_only=False,
    )

    progress_cb = train.SimpleProgressBar(batchsize)
    visu_cb = train.MetricsGraphing()
    callbacks = [visu_cb, checkpoint_cb, progress_cb, train.SwaCallback(start_epoch=epochs // 2)]

    trainer = pl.Trainer(
        fast_dev_run=False,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        default_root_dir=model_out_dir,
        # limit_train_batches=((10 * 1024) // batchsize),
        callbacks=callbacks,
        enable_checkpointing=True,
        max_epochs=epochs,
        log_every_n_steps=1,
        logger=False,
        enable_progress_bar=False,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    visu_cb.close()

    assert os.path.isfile(tmp_path / "models" / "swa.ckpt")
    assert os.path.isfile(tmp_path / "models" / "best.ckpt")
    assert os.path.isfile(tmp_path / "models" / "train.pdf")
