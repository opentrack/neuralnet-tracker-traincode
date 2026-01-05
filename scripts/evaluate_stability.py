#!/usr/bin/env python
# coding: utf-8

# Workaround for "RuntimeError: received 0 items of ancdata" in data loader.
# See https://github.com/pytorch/pytorch/issues/973
import gc
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096 * 2, rlimit[1]))

import numpy as np
import argparse
import tqdm
from typing import Any, NamedTuple, Optional, List, Tuple, Union, Callable, Dict
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
import scipy.signal
import tabulate
from matplotlib import pyplot
import itertools
from collections import defaultdict
import torch
import os
import copy
import glob
import matplotlib
import matplotlib.lines
import matplotlib.patches
import matplotlib.collections
import pickle
from torchvision.transforms import Compose
import torchmetrics
from torch import Tensor
from torch.utils.data import Subset

from trackertraincode.datasets.dshdf5pose import Hdf5PoseDataset
from trackertraincode.datasets.batch import Batch
import trackertraincode.datatransformation as dtr
import trackertraincode.pipelines
import trackertraincode.vis as vis
import trackertraincode.utils as utils
import trackertraincode.train as train
import trackertraincode.neuralnets.torchquaternion as quat
import trackertraincode.neuralnets.math
import trackertraincode.eval as eval


# Frame ranges where eyes closed
MYSELF_DATASET_BLINKS = np.asarray(
    [
        (42, 85),
        (103, 157),
        (189, 190),
        (209, 210),
        (222, 223),
        (254, 255),
        (298, 299),
        (360, 363),
        (398, 404),
        (466, 472),
        (504, 507),
        (567, 597),
    ]
)
MYSELF_DATASET_SMOOTH_SEGMENTS = [slice(0, 666), slice(670, None)]
MYSELF_DATASET_SMOOTHNESS_FILTER_SIZE = 15.0
MYSELF_DATASET_IMPORTANT_SEGMENTS = [
    slice(0, 215),
    slice(245, 310),
    slice(345, 425),
    slice(460, 515),
    slice(550, 610),
    slice(630, 665),
]


def _find_models(path: str):
    if os.path.isfile(path):
        return [path]
    else:
        return glob.glob(path)


class Poses(NamedTuple):
    hpb: NDArray[np.float64]
    xy: NDArray[np.float64]
    sz: NDArray[np.float64]
    roi_center: NDArray[np.float64]
    roi_area: NDArray[np.float64]
    pose_scales_tril: Optional[NDArray[np.float64]] = None
    coord_scales: Optional[NDArray[np.float64]] = None


def convertlabels(labels: dict | Batch) -> Poses:
    hpb = utils.as_hpb(Rotation.from_quat(labels.pop("pose")))
    coord = labels.pop("coord").numpy()
    xy = coord[..., :2]
    sz = coord[..., 2]
    roi = labels.pop("roi").numpy()
    roi_center = 0.5 * (roi[:, [0, 1]] + roi[:, [2, 3]])
    roi_area = np.prod(roi[:, [2, 3]] - roi[:, [0, 1]], axis=-1)
    labels = {k: v.numpy() for k, v in labels.items() if k in Poses._fields}
    return Poses(hpb, xy, sz, roi_center, roi_area, **labels)


def _low_pass_filtered_poses(poses: Poses, sequences):
    cutoff = 1.0 / MYSELF_DATASET_SMOOTHNESS_FILTER_SIZE
    sos = scipy.signal.butter(N=5, Wn=cutoff, btype="lowpass", fs=1.0, output="sos")
    new_poses = copy.deepcopy(poses)
    for a, b in sequences:
        s = slice(a, b)
        new_poses.hpb[s, ...] = scipy.signal.sosfiltfilt(sos, poses.hpb[s, ...], axis=0)
        new_poses.xy[s, ...] = scipy.signal.sosfiltfilt(sos, poses.xy[s, ...], axis=0)
        new_poses.sz[s, ...] = scipy.signal.sosfiltfilt(sos, poses.sz[s, ...], axis=0)
        new_poses.roi_center[s, ...] = scipy.signal.sosfiltfilt(
            sos, poses.roi_center[s, ...], axis=0
        )
        new_poses.roi_area[s, ...] = scipy.signal.sosfiltfilt(sos, poses.roi_area[s, ...], axis=0)
    return new_poses


def plot_coord(ax, pose: Poses, lowpass_pose: Poses | None = None, **kwargs):
    line0: matplotlib.lines.Line2D = ax[0].plot(pose.xy[:1000, 0], **kwargs)[0]
    line1: matplotlib.lines.Line2D = ax[1].plot(pose.xy[:1000, 1], **kwargs)[0]
    line2: matplotlib.lines.Line2D = ax[2].plot(pose.sz[:1000], **kwargs)[0]
    if lowpass_pose is not None:
        x = np.arange(len(pose.sz))
        ymin = np.minimum(pose.xy, lowpass_pose.xy)
        ymax = np.maximum(pose.xy, lowpass_pose.xy)
        ax[0].fill_between(
            x[:1000], ymin[:1000, 0], ymax[:1000, 0], alpha=0.5, color=line0.get_color(), **kwargs
        )
        ax[1].fill_between(
            x[:1000], ymin[:1000, 1], ymax[:1000, 1], alpha=0.5, color=line1.get_color(), **kwargs
        )
        ymin = np.minimum(pose.sz, lowpass_pose.sz)
        ymax = np.maximum(pose.sz, lowpass_pose.sz)
        ax[2].fill_between(
            x[:1000], ymin[:1000], ymax[:1000], alpha=0.5, color=line2.get_color(), **kwargs
        )


def plot_hpb(ax, pose: Poses, lowpass_pose: Poses | None = None, **kwargs):
    rad2deg = 180.0 / np.pi
    main_line = lowpass_pose if lowpass_pose is not None else pose
    if not "alpha" in kwargs:
        kwargs["alpha"] = 0.5

    line0 = ax[0].plot(main_line.hpb[:1000, 0] * rad2deg, **kwargs)[0]
    line1 = ax[1].plot(main_line.hpb[:1000, 1] * rad2deg, **kwargs)[0]
    line2 = ax[2].plot(main_line.hpb[:1000, 2] * rad2deg, **kwargs)[0]
    if lowpass_pose is not None:
        x = np.arange(len(pose.sz))
        ymin = np.minimum(pose.hpb, lowpass_pose.hpb) * rad2deg
        ymax = np.maximum(pose.hpb, lowpass_pose.hpb) * rad2deg
        ax[0].fill_between(
            x[:1000], ymin[:1000, 0], ymax[:1000, 0], color=line0.get_color(), **kwargs
        )
        ax[1].fill_between(
            x[:1000], ymin[:1000, 1], ymax[:1000, 1], color=line1.get_color(), **kwargs
        )
        ax[2].fill_between(
            x[:1000], ymin[:1000, 2], ymax[:1000, 2], color=line2.get_color(), **kwargs
        )
    return line0


def plot_roi(ax, pose: Poses, lowpass_pose: Poses | None = None, **kwargs):
    line0: matplotlib.lines.Line2D = ax[0].plot(pose.roi_center[:1000, 0], **kwargs)[0]
    line1: matplotlib.lines.Line2D = ax[1].plot(pose.roi_center[:1000, 1], **kwargs)[0]
    line2: matplotlib.lines.Line2D = ax[2].plot(pose.roi_area[:1000], **kwargs)[0]
    if lowpass_pose is not None:
        x = np.arange(len(pose.roi_area))
        ymin = np.minimum(pose.roi_center, lowpass_pose.roi_center)
        ymax = np.maximum(pose.roi_center, lowpass_pose.roi_center)
        ax[0].fill_between(
            x[:1000], ymin[:1000, 0], ymax[:1000, 0], alpha=0.5, color=line0.get_color(), **kwargs
        )
        ax[1].fill_between(
            x[:1000], ymin[:1000, 1], ymax[:1000, 1], alpha=0.5, color=line1.get_color(), **kwargs
        )
        ymin = np.minimum(pose.roi_area, lowpass_pose.roi_area)
        ymax = np.maximum(pose.roi_area, lowpass_pose.roi_area)
        ax[2].fill_between(
            x[:1000], ymin[:1000], ymax[:1000], alpha=0.5, color=line2.get_color(), **kwargs
        )


def _has_coord_cov_matrix(preds: Poses):
    return preds.coord_scales is not None and preds.coord_scales.shape[-2:] == (3, 3)


def visualize(
    preds: List[Poses],
    checkpoints: List[str],
    gt: Poses | None = None,
    sequences: NDArray[np.int32] | None = None,
    bars: NDArray[np.int32] | None = None,
):
    def make_nice(axes):
        for ax in axes[:-1]:
            ax.xaxis.set_visible(False)
        axes[0].legend()
        for i, ax in enumerate(axes):
            ax.patch.set_visible(False)  # Hide Background
            ax2 = ax.twinx()
            ax2.yaxis.set_visible(False)
            ax2.set_zorder(ax.get_zorder() - 1)
            if bars is not None:
                for a, b in bars:
                    ax2.bar(0.5 * (a + b), 1, width=b - a, bottom=0, color="yellow")
            if sequences is not None:
                for a, b in sequences[::2]:
                    if a >= 1000:
                        break
                    ax2.axvline(x=a, color="yellow", alpha=0.3)
                    ax2.axvline(x=b, color="yellow", alpha=0.3)
                    # ax2.bar(0.5 * (a + b), 1, width=b - a, bottom=0, color="yellow",alpha=0.1)

    fig, axes = pyplot.subplots(5, 2, figsize=(16, 10), sharex="all")
    axes = axes.ravel()

    if gt is not None:
        line = plot_hpb(axes[:3], gt, color="k", alpha=0.2)
        line.set_label("gt")
        plot_coord(axes[3:6], gt, color="k", alpha=0.2)
        plot_roi(axes[6:9], gt, color="k", alpha=0.2)

    for i, pred in enumerate(preds):
        lowpass_pose = _low_pass_filtered_poses(pred, sequences)
        line = plot_hpb(axes[:3], pred, lowpass_pose=lowpass_pose)
        line.set_label(checkpoints[i])
        plot_coord(axes[3:6], pred, lowpass_pose=lowpass_pose)
        plot_roi(axes[6:9], pred, lowpass_pose=lowpass_pose)

    for ax, label in zip(axes, "yaw,pitch,roll,x,y,size,bbx,bby,bbsize".split(",")):
        ax.set(ylabel=label)

    make_nice(axes)
    pyplot.tight_layout()

    if any(p.pose_scales_tril is not None for p in preds):
        axes_needed = 12 if any(_has_coord_cov_matrix(p) for p in preds) else 7

        fig2, axes = pyplot.subplots(axes_needed, 1, figsize=(18, 5))
        for checkpoint, pred in zip(checkpoints, preds):
            if pred.pose_scales_tril is None:
                continue
            ylim = np.amin(pred.pose_scales_tril), np.amax(pred.pose_scales_tril)
            k = 0
            for i in range(3):
                for j in range(i + 1):
                    axes[k].plot(pred.pose_scales_tril[..., i, j], label=checkpoint)
                    axes[k].set(ylabel=f"r cov[{i},{j}]")
                    axes[k].set(ylim=ylim)
                    k += 1
            if _has_coord_cov_matrix(pred):
                assert pred.coord_scales is not None
                axes[k + 0].plot(pred.coord_scales[..., 2, 2])
                axes[k + 1].plot(pred.coord_scales[..., 0, 0])
                axes[k + 2].plot(pred.coord_scales[..., 1, 1])
                axes[k + 3].plot(pred.coord_scales[..., 1, 0])
                axes[k + 4].plot(pred.coord_scales[..., 2, 0])
                axes[k + 5].plot(pred.coord_scales[..., 2, 1])
                for i, label in zip(range(k, k + 6), ["sz", "x", "y", "x-y", "y-sz", "x-sz"]):
                    axes[i].set(ylabel=label)
            elif pred.coord_scales is not None:
                axes[k].plot(pred.coord_scales[..., 2])
                axes[k].set(ylabel="sz")
        make_nice(axes)
        pyplot.tight_layout()
        return [fig, fig2]

    return [fig]


def report_blink_stability(
    poses_by_checkpoints: List[Poses],
    sequences: NDArray[np.int32],
    weights: NDArray[np.floating] | None,
):
    def mse(vals1, vals2):
        diffsqr = np.square(vals1 - vals2)
        return np.sqrt(np.average(diffsqr, axis=0, weights=weights))

    for poses in poses_by_checkpoints:
        lowpass_poses = _low_pass_filtered_poses(poses, sequences)
        e_hpb = np.average(mse(poses.hpb, lowpass_poses.hpb), axis=-1) * 180.0 / np.pi
        e_xy = np.average(mse(poses.xy, lowpass_poses.xy), axis=-1)
        e_sz = mse(poses.sz, lowpass_poses.sz)
        print(
            "\t"
            + ", ".join(
                f"{name:4s}: {mse_val:0.2f}"
                for name, mse_val in zip(["hpb", "sz", "xy"], [e_hpb, e_xy, e_sz])
            )
        )

    # xs = np.asarray([a for a, b in MYSELF_DATASET_BLINKS] + [b for a, b in MYSELF_DATASET_BLINKS], dtype=np.int64)
    # lefts = xs - 5
    # rights = xs + 5

    # def mse(vals):
    #     diffsqr = np.square(vals[lefts] - vals[rights])
    #     return np.sqrt(np.mean(diffsqr, axis=0))

    # def param_average_mse(name):
    #     return np.average([mse(getattr(poses, name)) for poses in poses_by_checkpoints], axis=0)

    # for name in ["hpb", "sz", "xy"]:
    #     mse_val = np.atleast_1d(param_average_mse(name))
    #     if name == "hpb":
    #         mse_val *= 180.0 / np.pi
    #     print(f"\t {name:4s}: " + ", ".join(f"{x:0.2f}" for x in mse_val))


def get_sequences(loader: dtr.SampleBySampleLoader) -> NDArray[np.int32]:
    assert hasattr(loader.dataset, "sequence_starts")
    assert isinstance(loader.dataset.sequences, np.ndarray)
    return loader.dataset.sequences


def closed_loop_tracking(model: eval.Predictor, loader: dtr.SampleBySampleLoader):
    sequences = get_sequences(loader)

    bar = tqdm.tqdm(total=len(loader.dataset))
    sample_iter = iter(loader)
    preds = []

    def predict_sequence(num_frames):
        current_roi = None
        for _ in range(num_frames):
            sample: Batch = next(sample_iter)
            image, roi = sample["image"], sample["roi"]
            if current_roi is not None:
                roi[...] = current_roi
            pred = model.predict_batch(image[None, ...], roi[None, ...])
            x0, y0, x1, y1 = pred["roi"][0]
            w, h = sample.meta.image_wh
            current_roi = torch.tensor([max(0.0, x0), max(0.0, y0), min(x1, w), min(y1, h)])
            preds.append(pred)
            bar.update(1)

    for a, b in sequences:
        predict_sequence(b - a)

    preds_batch = Batch.collate(preds)
    poses = convertlabels(preds_batch)
    return poses


def open_loop_tracking(model: eval.Predictor, loader: dtr.SampleBySampleLoader):
    sample = loader.dataset[0]
    output_keys = model.predict_batch(sample["image"][None, ...], sample["roi"][None, ...]).keys()
    metric = torchmetrics.MetricCollection({k: eval.PredExtractor(k) for k in output_keys})
    preds = model.evaluate(metric, loader)
    return convertlabels(preds)


def _track_multiple_networks(
    path: str,
    device: str,
    loader: Any,
    prediction_func: Callable[[eval.Predictor, dtr.SampleBySampleLoader[Batch]], Poses],
    crop_size_factor: float,
) -> list[Poses]:
    """Predictions from a single or a group of networks"""
    checkpoints = _find_models(path)
    print(f"Evaluating {path}. Found {len(checkpoints)} checkpoints.")
    preds = []
    for checkpoint in checkpoints:
        net = eval.Predictor(checkpoint, device=device, focus_roi_expansion_factor=crop_size_factor)
        preds.append(prediction_func(net, loader))
    return preds


def main_open_loop(paths: List[str], device):
    loader = trackertraincode.pipelines.make_validation_loader("myself", return_single_samples=True)

    def process(path, crop_size_factor):
        return _track_multiple_networks(path, device, loader, open_loop_tracking, crop_size_factor)

    poses_by_checkpoints = defaultdict(list)
    for crop_size_factor in [1.0, 1.2]:
        poses_lists = list(
            itertools.chain.from_iterable((process(fn, crop_size_factor) for fn in paths))
        )
        figs = visualize(poses_lists, paths)
        for fig in figs:
            fig.suptitle(f"cropsize={crop_size_factor:.1f}")
        for fn, poses in zip(paths, poses_lists):
            poses_by_checkpoints[fn].append(poses)
    pyplot.show()

    for name in paths:
        print(f"Checkpoint: {name}")
        report_blink_stability(poses_by_checkpoints[name])


def main_closed_loop(paths: List[str], device, dataset: str = "biwi-snippets"):
    loader = trackertraincode.pipelines.make_validation_loader(dataset, return_single_samples=True)

    weights: NDArray[np.floating] | None = None
    if dataset == "myself":
        gt = None
        bars = MYSELF_DATASET_BLINKS
        if 0:
            weights = np.full((len(loader.dataset),), fill_value=0.1, dtype=np.float32)
            for s in MYSELF_DATASET_IMPORTANT_SEGMENTS:
                weights[s] = 1.0
    else:
        gt_coord = loader.dataset.get_data_channel("coords")
        gt_pose = loader.dataset.get_data_channel("quats")
        gt_roi = loader.dataset.get_data_channel("rois")
        gt = {"coord": gt_coord, "pose": gt_pose, "roi": gt_roi}
        gt = convertlabels(gt)
        bars = None

    def process(path, crop_size_factor):
        return _track_multiple_networks(
            path, device, loader, closed_loop_tracking, crop_size_factor
        )

    poses_by_checkpoints = defaultdict(list)
    for crop_size_factor in [1.0, 1.2]:
        poses_lists = list(
            itertools.chain.from_iterable((process(fn, crop_size_factor) for fn in paths))
        )
        figs = visualize(poses_lists, paths, gt=gt, sequences=get_sequences(loader), bars=bars)
        for fig in figs:
            fig.suptitle(f"closed-loop cropsize={crop_size_factor:.1f}")
        for fn, poses in zip(paths, poses_lists):
            poses_by_checkpoints[fn].append(poses)
    pyplot.show()

    for name in paths:
        print(f"Checkpoint: {name}")
        report_blink_stability(poses_by_checkpoints[name], get_sequences(loader), weights=weights)


def _create_biwi_sections_loader():
    intervals = [(145, 216), (1360, 1464), (3030, 3120), (8020, 8100), (6570, 6600), (9030, 9080)]
    indices = np.concatenate([np.arange(a, b) for a, b in intervals])
    loader = trackertraincode.pipelines.make_validation_loader(
        "biwi", return_single_samples=True, order=indices
    )
    sequence_starts = np.cumsum([0] + [(b - a) for a, b in intervals])
    return loader, sequence_starts


def main_analyze_pitch_vs_yaw(checkpoints: List[str]):
    fig, axes = pyplot.subplots(2, 1, figsize=(20, 5))

    loader = trackertraincode.pipelines.make_validation_loader(
        "myself_yaw", return_single_samples=True
    )

    def predict_all_nets(loader):
        poses_vs_model = {}
        for checkpoint in checkpoints:
            predictor = eval.Predictor(checkpoint, device="cuda")
            metrics = torchmetrics.MetricCollection(
                {"pose": eval.PredExtractor("pose"), "coord": eval.PredExtractor("coord")}
            )
            results = predictor.evaluate(metrics, loader)
            poses = convertlabels(results)
            poses.hpb[...] *= 180.0 / np.pi
            poses_vs_model[checkpoint] = poses
        return poses_vs_model

    poses_vs_model = predict_all_nets(loader)

    ax = axes[0]
    for name, poses in poses_vs_model.items():
        ax.scatter(poses.hpb[:, 0], poses.hpb[:, 1], label=name, s=5.0)
    ax.set(xlabel="yaw", ylabel="pitch")
    ax.legend()
    ax.axhline(0.0, color="k")
    ax.axvline(0.0, color="k")

    loader, sequence_starts = _create_biwi_sections_loader()
    poses_vs_model = predict_all_nets(loader)

    ax = axes[1]
    colors = "rgbcmy"
    alphas = [0.5] * len(checkpoints)
    alphas[0] = 1.0
    for j, (name, poses) in enumerate(poses_vs_model.items()):
        assert poses.hpb.shape[0] == sequence_starts[-1]
        for i, (a, b) in enumerate(zip(sequence_starts[:-1], sequence_starts[1:])):
            hpb = poses.hpb[a:b]
            # order = np.argsort(hpb[:,0])
            ax.plot(hpb[:, 0], hpb[:, 1], c=colors[i], alpha=alphas[j])
    ax.set(xlabel="yaw", ylabel="pitch")
    ax.legend()
    ax.axhline(0.0, color="k")
    ax.axvline(0.0, color="k")

    pyplot.show()


class NoisifyBatch:
    def __init__(self, noise_scale: float):
        self._noise_scale = noise_scale

    def __call__(self, batch: Batch):
        batch = copy.copy(batch)
        batch["image"] = batch["image"] + self._noise_scale * torch.randn_like(batch["image"])
        return batch


def _vis_one_noise_resist(noiselevels, metrics_by_noise_and_quantity, quantity, lbl, ax):
    values = np.asarray([metrics_by_noise_and_quantity[l, quantity] for l in noiselevels])
    values *= 180.0 / np.pi
    avg = np.average(values, axis=-1)
    std = np.std(values, axis=-1)
    ax[0].errorbar(noiselevels, avg, yerr=std, capsize=10.0, label=lbl)
    ax[0].legend()
    if quantity == "pose":
        ax[0].set(xlim=(0.0, 64), ylim=(4.0, 11.0), xlabel="input noise", ylabel="rot err [deg]")
    elif quantity == "roi":
        ax[0].set(
            xlim=(0.0, 64), ylim=(0.0, 0.02), xlabel="input noise", ylabel="coordinate err [%]"
        )


def main_vis_noise_resist(paths: list[str]):
    datas = []
    for i, path in enumerate(paths):
        print(f"({i}) ", path)
        with open(path, "rb") as f:
            datas.append((path, pickle.load(f)))

    for quantity in ["pose"]:
        fig, ax = pyplot.subplots(1, 1)
        ax = [ax]
        for i, (name, (noiselevels, metrics_by_noise_and_quantity)) in enumerate(datas):
            # noiselevels = list(sorted(set(l for (l,q) in metrics_by_noise_and_quantity.keys)))
            _vis_one_noise_resist(noiselevels, metrics_by_noise_and_quantity, quantity, f"{i}", ax)
            # ax[0].legend()
    pyplot.show()


def main_analyze_noise_resist(paths: List[str]):
    data_samples = None
    noiselevels = [0.0, 2.0, 8.0, 16.0, 32.0, 48.0, 64.0]

    # data_samples = 10
    # noiselevels = [ 1., 8., 16.,  ]

    def predict_noisy_dataset(
        predictor: eval.Predictor, noiselevel, quantity_names
    ) -> dict[str, Tensor]:
        """Predicts given noisy inputs.

        Return:
            Dicts indexed by quantity name, returning each:
            Predictions (B x Noise x Feature) or GT (B x Feature).
        """

        loader = trackertraincode.pipelines.make_validation_loader(
            "aflw2k3d",
            use_head_roi=True,
            order=(None if data_samples is None else np.arange(data_samples)),
            additional_sample_transform=Compose(predictor.normalize_crop_transform),
            additional_batch_transform=NoisifyBatch(noiselevel / 256.0),
        )

        metrics = torchmetrics.MetricCollection({})
        if "pose" in quantity_names:
            metrics.add_metrics({"pose": eval.GeodesicError()})

        return predictor.evaluate_cropped_normalized(metrics, loader)

    assert all(bool(_find_models(path)) for path in paths)
    metrics_by_noise_and_quantity = defaultdict(list)
    for path in paths:
        predictor = eval.Predictor(path, device="cuda")
        for noiselevel in noiselevels:
            results = predict_noisy_dataset(predictor, noiselevel, ("pose"))
            metrics_by_noise_and_quantity[noiselevel, "pose"].append(results["pose"].mean().numpy())

    filename = os.path.join("/tmp", os.path.splitext(paths[0])[0] + "_noise_resist_v3.pkl")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump((noiselevels, metrics_by_noise_and_quantity), f)

    if 1:
        main_vis_noise_resist([filename])


def main_analyze_uncertainty_error_correlation(paths: List[str]):
    results_by_paths = {}

    checkpoints = itertools.chain.from_iterable(_find_models(path) for path in paths)

    for checkpoint in checkpoints:
        metrics = torchmetrics.MetricCollection(
            {
                "pose": eval.GeodesicError(),
                "pose_scales_tril": eval.PredExtractor("pose_scales_tril"),
            }
        )
        predictor = eval.Predictor(checkpoint, device="cuda")
        loader = trackertraincode.pipelines.make_validation_loader(
            "aflw2k3d",
            use_head_roi=True,
            additional_sample_transform=Compose(predictor.normalize_crop_transform),
        )
        results = predictor.evaluate_cropped_normalized(metrics, loader)
        results["undertainty"] = torch.norm(
            torch.matmul(results["pose_scales_tril"], results["pose_scales_tril"].mT), dim=(-1, -2)
        )
        results_by_paths[checkpoint] = (results["pose"], results["undertainty"])

    fig, ax = pyplot.subplots(1, 1, dpi=120, figsize=(3, 2))
    ax = [ax]
    for path, (rot_err, uncertainty) in results_by_paths.items():
        ax[0].set_axisbelow(True)
        ax[0].grid()
        ax[0].scatter(
            rot_err * 180.0 / np.pi,
            np.sqrt(uncertainty) * 180.0 / np.pi,
            rasterized=True,
            s=10.0,
            edgecolor="none",
            alpha=0.5,
        )
        ax[0].set(xlabel="geo. err. °", ylabel="uncertainty °")
    pyplot.tight_layout
    fig.savefig("/tmp/uncertainty_vs_err.svg")
    pyplot.show()


def main_analyze_stability_vs_variations(checkpoints: List[str]):
    """Shows the prediction variance within 'sequences' in the dataset.

    This is intended to analyze variance wrt changes in environment and facial expressions.
    To use it, the dataset should contain different individuals with fixed poses and 'sequences'
    which are not actual sequences but contain different renderings with such variations.

    The plot shows the yaw-pitch plane.
    * Black X = GT
    * Colored X = Mean rotation over the variations
    * Colored circle = Kind of a rotational standard deviation over the variations.

    For stability against environmental conditions and say, talking, the circles should be
    small, whereas the X deviations from GT shows independent pose biases.
    """

    loader = trackertraincode.pipelines.make_validation_loader(
        "replicantface-stability",
        return_single_samples=True,
        # order=np.arange(128)
    )

    def rotational_centers_and_deviations(quats, individuals):
        uniques = np.unique(individuals)
        out_means = []
        out_deviations = []
        for u in uniques:
            rots = Rotation.from_quat(quats[individuals == u, :])
            mean_rot = eval.compute_mean_rotation(rots)
            deviations = np.average((mean_rot.inv() * rots).magnitude())
            out_means.append(mean_rot)
            out_deviations.append(deviations)
        return Rotation.concatenate(out_means), np.stack(out_deviations)

    def predict_all_nets(loader):
        poses_vs_model = {}
        for checkpoint in checkpoints:
            predictor = eval.Predictor(checkpoint, device="cuda")
            metrics = torchmetrics.MetricCollection(
                {
                    "pose": eval.PredExtractor("pose"),
                    "individual": eval.LabelExtractor("individual"),
                    "pose_gt": eval.LabelExtractor("pose"),
                }
            )
            results = predictor.evaluate(metrics, loader)
            quats = results.pop("pose")
            hpb = utils.as_hpb(Rotation.from_quat(quats))
            hpb_gt = utils.as_hpb(Rotation.from_quat(results.pop("pose_gt")))
            individuals = results.pop("individual")
            means, deviations = rotational_centers_and_deviations(quats, individuals)
            poses_vs_model[checkpoint] = {
                "hpb": hpb,
                "hpb_gt": hpb_gt,
                "means": utils.as_hpb(means),
                "deviations": deviations,
            }
        return poses_vs_model

    poses_vs_model = predict_all_nets(loader)

    fig, axes = pyplot.subplots(1, 1, figsize=(10, 10))
    ax = axes

    def plot_rots(hpb, color, **kwargs):
        hpb = hpb * 180.0 / np.pi
        ax.scatter(hpb[:, 0], hpb[:, 1], c=color, s=kwargs.pop("s", 5.0), **kwargs)

    def plot_circles(hpb, deviations, color):
        hpb = hpb * 180.0 / np.pi
        deviations = deviations * 180.0 / np.pi
        circles = [
            matplotlib.patches.Circle((x, y), radius=d, fill=False)
            for (x, y, _), d in zip(hpb, deviations)
        ]
        c = matplotlib.collections.PatchCollection(circles, linewidth=1.0, ec=color, fc="none")
        ax.add_collection(c)

    if 1:  # Plot GT
        gt = next(iter(poses_vs_model.values()))["hpb_gt"]
        plot_rots(gt, "k", marker="x", s=50.0, label="GT")

    for (name, poses), color in zip(poses_vs_model.items(), "rgbcmy"):
        plot_rots(poses["means"], color, marker="x", label=name[-20:], s=50)
        plot_circles(poses["means"], poses["deviations"], color)
        print(f"Deviation ({name}): {np.average(poses['deviations'])*180./np.pi:.2f} deg")
    if len(poses_vs_model) > 1:
        print(
            f"mean deviation: {np.average([poses['deviations'] for poses in poses_vs_model.values()])*180./np.pi:.2f} deg"
        )

    ax.legend()
    ax.axhline(0.0, color="k")
    ax.axvline(0.0, color="k")
    pyplot.show()


if __name__ == "__main__":
    np.seterr(all="raise")
    parser = argparse.ArgumentParser(description="Evaluates the model")
    parser.add_argument(
        "mode",
        choices=[
            "closed-loop",
            "pitch-yaw",
            "open-loop",
            "noise-resist",
            "uncertainty-correlation",
            "variation-resist",
        ],
        default="none",
    )
    parser.add_argument("filename", nargs="+", help="input filenames", type=str)
    parser.add_argument("--vis", action="store_true", default=False)
    parser.add_argument("--ds", default="", type=str)
    args = parser.parse_args()
    if args.mode == "open-loop":
        assert not args.ds
        main_open_loop(args.filename, "cuda")
    elif args.mode == "closed-loop":
        assert args.ds in ("myself", "biwi-snippets", "")
        main_closed_loop(args.filename, "cuda", dataset=args.ds if args.ds else "myself")
    elif args.mode == "pitch-yaw":
        assert not args.ds
        main_analyze_pitch_vs_yaw(args.filename)
    elif args.mode == "noise-resist":
        assert not args.ds
        if args.vis:
            main_vis_noise_resist(args.filename)
        else:
            main_analyze_noise_resist(args.filename)
    elif args.mode == "variation-resist":
        assert not args.ds
        main_analyze_stability_vs_variations(args.filename)
    elif args.mode == "uncertainty-correlation":
        assert not args.ds
        main_analyze_uncertainty_error_correlation(args.filename)
    else:
        raise RuntimeError(f"Unkown mode {args.mode}")
