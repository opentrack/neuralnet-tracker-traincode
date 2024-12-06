#!/usr/bin/env python
# coding: utf-8

# Seems to run a bit faster than with default settings and less bugged
# See https://github.com/pytorch/pytorch/issues/67864
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

from typing import List, NamedTuple, Optional, Any, Mapping
from os.path import join, dirname, realpath
import numpy as np
import cv2
import argparse
import functools
import itertools
from collections import defaultdict
import tqdm

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

# from pytorch_lightning.loggers import Logger,
from pytorch_lightning.utilities import rank_zero_only
import torch.optim as optim
import torch
import torch.nn as nn

import trackertraincode.neuralnets.losses as losses
import trackertraincode.neuralnets.models as models
import trackertraincode.neuralnets.negloglikelihood as NLL
import trackertraincode.train as train
import trackertraincode.pipelines

from trackertraincode.neuralnets.io import complement_lightning_checkpoint
from scripts.export_model import convert_posemodel_onnx
from trackertraincode.datasets.batch import Batch
from trackertraincode.pipelines import Tag


class MyArgs(argparse.Namespace):
    backbone: str
    batchsize: int
    lr: float
    find_lr: bool
    epochs: int
    ds: str
    plotting: bool
    plot_save_filename: Optional[str]
    swa: bool
    outdir: str
    ds_weight_are_sampling_frequencies: bool
    with_pointhead: bool
    with_nll_loss: bool
    rotation_aug_angle: float
    with_image_aug: bool
    with_blurpool: bool
    export_onnx: bool
    input_size: int
    roi_override: str
    with_roi_train: bool
    dropout_prob: float
    rampup_nll_losses: bool


def parse_dataset_definition(arg: str):
    """Parses CLI dataset specifications

    Of the form <name1>[:<weight1>]+<name2>[:<weight2>]+...
    """

    dsmap = {
        "300wlp": trackertraincode.pipelines.Id._300WLP,
        "synface": trackertraincode.pipelines.Id.SYNFACE,
        "aflw2k": trackertraincode.pipelines.Id.AFLW2k3d,
        "biwi": trackertraincode.pipelines.Id.BIWI,
        "wider": trackertraincode.pipelines.Id.WIDER,
        "repro_300_wlp": trackertraincode.pipelines.Id.REPO_300WLP,
        "repro_300_wlp_woextra": trackertraincode.pipelines.Id.REPO_300WLP_WO_EXTRA,
        "wflw_lp": trackertraincode.pipelines.Id.WFLW_LP,
        "lapa_megaface_lp": trackertraincode.pipelines.Id.LAPA_MEGAFACE_LP,
    }

    splitted = arg.split("+")

    # Find dataset specification which has weights in them
    # and add them to a dict.
    it = (tuple(s.split(":")) for s in splitted if ":" in s)
    dataset_weights = {dsmap[k]: float(v) for k, v in it}

    # Then consider all datasets listed
    dsids = [dsmap[s.split(":")[0]] for s in splitted]
    dsids = list(frozenset(dsids))

    return dsids, dataset_weights


def setup_datasets(args: MyArgs):
    dsids, dataset_weights = parse_dataset_definition(args.ds)

    train_loader, test_loader, ds_size = trackertraincode.pipelines.make_pose_estimation_loaders(
        inputsize=args.input_size,
        batchsize=args.batchsize,
        datasets=dsids,
        dataset_weights=dataset_weights,
        use_weights_as_sampling_frequency=args.ds_weight_are_sampling_frequencies,
        enable_image_aug=args.with_image_aug,
        rotation_aug_angle=args.rotation_aug_angle,
        roi_override=args.roi_override,
    )

    return train_loader, test_loader, ds_size


def find_variance_parameters(net: nn.Module):
    if isinstance(net, (NLL.FeaturesAsTriangularScale, NLL.FeaturesAsDiagonalScale, NLL.DiagonalScaleParameter)):
        return list(net.parameters())
    else:
        return sum((find_variance_parameters(x) for x in net.children()), start=[])


def find_transformer_parameters(net: nn.Module):
    if isinstance(net, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
        return list(net.parameters())
    else:
        return sum((find_transformer_parameters(x) for x in net.children()), start=[])


def setup_lr_with_slower_variance_training(net, base_lr):
    variance_params = find_variance_parameters(net)
    transformer_params = find_transformer_parameters(net)
    # print ("Transformer param shapes: ", [p.shape for p in transformer_params])
    other_params = list(
        frozenset(net.parameters()).difference(frozenset(variance_params) | frozenset(transformer_params))
    )
    return [
        {"params": other_params, "lr": base_lr},
        {"params": variance_params, "lr": 0.1 * base_lr},
        {"params": transformer_params, "lr": 0.01 * base_lr, "weight_decay": 0.01},
    ]


def create_optimizer(net, args: MyArgs):
    optimizer = optim.Adam(
        setup_lr_with_slower_variance_training(net, args.lr),
        lr=args.lr,
    )
    # if args.find_lr:
    #     print("LR finding mode!")
    #     n_epochs = args.epochs
    #     lr_max = 1.0e-1
    #     base = np.power(lr_max / args.lr, 1.0 / n_epochs)
    #     scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda e: base**e, verbose=True)
    # else:

    n_epochs = args.epochs
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [n_epochs//2], 0.1)
    scheduler = train.ExponentialUpThenSteps(optimizer, max(1, n_epochs // (10)), 0.1, [n_epochs // 2])
    # scheduler = train.LinearUpThenSteps(optimizer, max(1,n_epochs//(10)), 0.1, [n_epochs//2])

    return optimizer, scheduler


class SaveBestSpec(NamedTuple):
    weights: List[float]
    names: List[str]


def setup_losses(args: MyArgs, net):
    C = train.Criterion
    cregularize = [
        C("quatregularization1", losses.QuaternionNormalizationSoftConstraint(), 1.0e-6),
    ]
    poselosses = []
    roilosses = []
    pointlosses = []
    pointlosses25d = []
    shapeparamloss = []

    if args.with_nll_loss:

        def ramped_up_nll_weight(multiplier):
            if args.rampup_nll_losses:

                def wrapped(step):
                    strength = min(1.0, max(0.0, (step / args.epochs - 0.1) * 10.0))
                    return 0.01 * strength * multiplier

                return wrapped
            else:
                return multiplier * 0.01

        poselosses += [
            C("nllrot", NLL.QuatPoseNLLLoss().to("cuda"), ramped_up_nll_weight(0.5)),
            C("nllcoord", NLL.CorrelatedCoordPoseNLLLoss().cuda(), ramped_up_nll_weight(0.5)),
        ]
        if args.with_roi_train:
            roilosses += [C("nllbox", NLL.BoxNLLLoss(distribution="gaussian"), ramped_up_nll_weight(0.01))]
        if args.with_pointhead:
            pointlosses += [
                C(
                    "nllpoints3d",
                    NLL.Points3dNLLLoss(chin_weight=0.8, eye_weight=0.0, distribution="gaussian").cuda(),
                    ramped_up_nll_weight(0.5),
                )
            ]
            pointlosses25d = [
                C(
                    "nllpoints3d",
                    NLL.Points3dNLLLoss(
                        chin_weight=0.8, eye_weight=0.0, pointdimension=2, distribution="gaussian"
                    ).cuda(),
                    ramped_up_nll_weight(0.5),
                )
            ]
            shapeparamloss += [
                # C('nllshape', NLL.ShapeParamsNLLLoss(distribution='gaussian'), ramped_up_nll_weight(0.01))
            ]
    if 1:
        poselosses += [
            C("rot", losses.QuatPoseLoss("approx_distance"), 0.5),
            C("xy", losses.PoseXYLoss("l2"), 0.5 * 0.5),
            C("sz", losses.PoseSizeLoss("l2"), 0.5 * 0.5),
        ]
        if args.with_roi_train:
            roilosses += [C("box", losses.BoxLoss("l2"), 0.01)]
        if args.with_pointhead:
            pointlosses += [
                C("points3d", losses.Points3dLoss("l2", chin_weight=0.8, eye_weights=0.0).cuda(), 0.5),
            ]
            pointlosses25d += [
                C(
                    "points3d",
                    losses.Points3dLoss("l2", pointdimension=2, chin_weight=0.8, eye_weights=0.0).cuda(),
                    0.5,
                ),
            ]
            shapeparamloss += [
                C("shp_l2", losses.ShapeParameterLoss(), 0.1),
            ]
            cregularize += [
                C("nll_shp_gmm", losses.ShapePlausibilityLoss().cuda(), 0.1),
            ]

    train_criterions = {
        Tag.ONLY_POSE: train.CriterionGroup(poselosses + cregularize),
        Tag.POSE_WITH_LANDMARKS: train.CriterionGroup(
            poselosses + cregularize + pointlosses + shapeparamloss + roilosses
        ),
        Tag.POSE_WITH_LANDMARKS_3D_AND_2D: train.CriterionGroup(
            poselosses + cregularize + pointlosses + shapeparamloss + roilosses
        ),
        Tag.ONLY_LANDMARKS: train.CriterionGroup(pointlosses + cregularize),
        Tag.ONLY_LANDMARKS_25D: train.CriterionGroup(pointlosses25d + cregularize),
    }
    test_criterions = {
        Tag.POSE_WITH_LANDMARKS: train.CriterionGroup(
            poselosses + pointlosses + roilosses + shapeparamloss + cregularize
        ),
    }

    savebest = SaveBestSpec([1.0, 1.0, 1.0], ["rot", "xy", "sz"])

    return train_criterions, test_criterions, savebest


def create_net(args: MyArgs):
    return models.NetworkWithPointHead(
        enable_point_head=args.with_pointhead,
        enable_face_detector=False,
        config=args.backbone,
        enable_uncertainty=args.with_nll_loss,
        backbone_args={},
    )


class LitModel(pl.LightningModule):
    # TODO: plot gradient magnitudes

    def __init__(self, args: MyArgs):
        super().__init__()
        self._args = args
        self._model = create_net(args)
        train_criterions, test_criterions, savebest = setup_losses(args, self._model)
        self._train_criterions = train_criterions
        self._test_criterions = test_criterions

    def training_step(self, batches: list[Batch], batch_idx):
        loss_sum, all_lossvals = train.default_compute_loss(
            self._model, batches, self.current_epoch, self._train_criterions
        )
        loss_val_by_name = {
            name: val
            for name, (val, _) in train.concatenated_lossvals_by_name(
                itertools.chain.from_iterable(all_lossvals)
            ).items()
        }
        self.log("loss", loss_sum, on_epoch=True, prog_bar=True, batch_size=sum(b.meta.batchsize for b in batches))
        return {"loss": loss_sum, "mt_losses": loss_val_by_name}

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor | dict[str, Any] | None:
        images = batch["image"]
        pred = self._model(images)
        values = self._test_criterions[batch.meta.tag].evaluate(pred, batch, batch_idx)
        val_loss = torch.cat([(lv.val * lv.weight) for lv in values]).sum()
        self.log("val_loss", val_loss, on_epoch=True, batch_size=batch.meta.batchsize)
        return values

    def configure_optimizers(self):
        optimizer, scheduler = create_optimizer(self._model, self._args)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @property
    def model(self):
        return self._model


class SwaCallback(Callback):
    def __init__(self, start_epoch):
        super().__init__()
        self._swa_model: optim.swa_utils.AveragedModel | None = None
        self._start_epoch = start_epoch

    @property
    def swa_model(self):
        return self._swa_model.module

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        assert isinstance(pl_module, LitModel)
        self._swa_model = optim.swa_utils.AveragedModel(pl_module.model, device="cpu", use_buffers=True)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        assert isinstance(pl_module, LitModel)
        if trainer.current_epoch > self._start_epoch:
            self._swa_model.update_parameters(pl_module.model)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        assert self._swa_model is not None
        swa_filename = join(trainer.default_root_dir, f"swa.ckpt")
        models.save_model(self._swa_model.module, swa_filename)


class MetricsGraphing(Callback):
    def __init__(self):
        super().__init__()
        self._visu: train.TrainHistoryPlotter | None = None
        self._metrics_accumulator = defaultdict(list)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        assert self._visu is None
        self._visu = train.TrainHistoryPlotter(save_filename=join(trainer.default_root_dir, "train.pdf"))

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int
    ):
        mt_losses: dict[str, torch.Tensor] = outputs["mt_losses"]
        for k, v in mt_losses.items():
            self._visu.add_train_point(trainer.current_epoch, batch_idx, k, v.numpy())
        self._visu.add_train_point(trainer.current_epoch, batch_idx, "loss", outputs["loss"].detach().cpu().numpy())

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.lr_scheduler_configs:  # scheduler is not None:
            scheduler = next(
                iter(trainer.lr_scheduler_configs)
            ).scheduler  # Pick the first scheduler (and there should only be one)
            last_lr = next(iter(scheduler.get_last_lr()))  # LR from the first parameter group
            self._visu.add_test_point(trainer.current_epoch, "lr", last_lr)

        self._visu.summarize_train_values()

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._metrics_accumulator = defaultdict(list)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list[train.LossVal],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        for val in outputs:
            self._metrics_accumulator[val.name].append(val.val)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._visu is None:
            return
        for k, v in self._metrics_accumulator.items():
            self._visu.add_test_point(trainer.current_epoch - 1, k, torch.cat(v).mean().cpu().numpy())
        if trainer.current_epoch > 0:
            self._visu.update_graph()


class SimpleProgressBar(Callback):
    """Creates progress bars for total training time and progress of per epoch."""

    def __init__(self, batchsize: int):
        super().__init__()
        self._bar: tqdm.tqdm | None = None
        self._epoch_bar: tqdm.tqdm | None = None
        self._batchsize = batchsize

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._bar = tqdm.tqdm(total=trainer.max_epochs, desc='Training', position=0)
        self._epoch_bar = tqdm.tqdm(total=trainer.num_training_batches * self._batchsize, desc="Epoch", position=1)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._bar.close()
        self._epoch_bar.close()
        self._bar = None
        self._epoch_bar = None

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._epoch_bar.reset(self._epoch_bar.total)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._bar.update(1)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Mapping[str, Any],
        batch: list[Batch] | Batch,
        batch_idx: int,
    ) -> None:
        n = sum(b.meta.batchsize for b in batch) if isinstance(batch, list) else batch.meta.batchsize
        self._epoch_bar.update(n)


def main():
    np.seterr(all="raise")
    cv2.setNumThreads(1)

    parser = argparse.ArgumentParser(description="Trains the model")
    parser.add_argument("--backbone", help="Which backbone the net uses", default="mobilenetv1")
    parser.add_argument("--batchsize", help="The batch size to train with", type=int, default=64)
    parser.add_argument("--lr", help="learning rate", type=float, default=1.0e-3)
    # parser.add_argument("--find-lr", help="Enable learning rate finder mode", action="store_true", default=False)
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=200)
    parser.add_argument("--ds", help="Which datasets to train on. See code.", type=str, default="300wlp")
    # parser.add_argument(
    #     "--no-plotting", help="Disable plotting of losses", action="store_false", default=True, dest="plotting"
    # )
    # parser.add_argument(
    #     "--save-plot",
    #     help="Filename to enable saving the train history as plot",
    #     default=None,
    #     type=str,
    #     dest="plot_save_filename",
    # )
    parser.add_argument(
        "--with-swa", help="Enable stochastic weight averaging", action="store_true", default=False, dest="swa"
    )
    parser.add_argument(
        "--outdir", help="Output sub-directory", type=str, default=join(dirname(__file__), "..", "model_files")
    )
    parser.add_argument(
        "--ds-weighting",
        help="Sample dataset with equal probability and use weights for scaling their losses",
        action="store_false",
        default=True,
        dest="ds_weight_are_sampling_frequencies",
    )
    parser.add_argument(
        "--no-pointhead", help="Disable landmark prediction", action="store_false", default=True, dest="with_pointhead"
    )
    parser.add_argument("--with-nll-loss", default=False, action="store_true")
    parser.add_argument("--raug", default=30, type=float, dest="rotation_aug_angle")
    parser.add_argument("--no-imgaug", default=True, action="store_false", dest="with_image_aug")
    # parser.add_argument('--no-blurpool', default=True, action='store_false', dest='with_blurpool')
    # parser.add_argument("--no-onnx", default=True, action="store_false", dest="export_onnx")
    parser.add_argument(
        "--roi-override",
        default="original",
        type=str,
        choices=["extent_to_forehead", "original", "landmarks"],
        dest="roi_override",
    )
    parser.add_argument("--no-roi-train", default=True, action="store_false", dest="with_roi_train")
    parser.add_argument("--rampup-nll-losses", default=False, action="store_true")

    args: MyArgs = parser.parse_args()
    args.input_size = 129

    train_loader, test_loader, _ = setup_datasets(args)

    model = LitModel(args)
    model_out_dir = join(args.outdir, model.model.name)

    checkpoint_cb = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        monitor="val_loss",
        enable_version_counter=False,
        filename="best",
        dirpath=model_out_dir,
        save_weights_only=False,
    )

    progress_cb = SimpleProgressBar(args.batchsize)

    callbacks = [MetricsGraphing(), checkpoint_cb, progress_cb]

    swa_callback = None
    if args.swa:
        swa_callback = SwaCallback(start_epoch=args.epochs * 2 // 3)
        callbacks.append(swa_callback)

    # TODO: inf norm?
    trainer = pl.Trainer(
        fast_dev_run=False,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        default_root_dir=model_out_dir,
        limit_train_batches=((10 * 1024) // args.batchsize),
        callbacks=callbacks,
        enable_checkpointing=True,
        max_epochs=args.epochs,
        log_every_n_steps=10,
        logger=False,
        enable_progress_bar=False,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    if checkpoint_cb is not None:
        # Overwrite the lightning checkpoint!
        model = LitModel.load_from_checkpoint(checkpoint_cb.last_model_path, args=args).to("cpu")
        models.save_model(model.model, checkpoint_cb.last_model_path)
        model = LitModel.load_from_checkpoint(checkpoint_cb.best_model_path, args=args).to("cpu")
        models.save_model(model.model, checkpoint_cb.best_model_path)


if __name__ == "__main__":
    main()
