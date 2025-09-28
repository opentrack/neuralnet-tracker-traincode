#!/usr/bin/env python
# coding: utf-8

# Workaround for "RuntimeError: received 0 items of ancdata" in data loader.
# See https://github.com/pytorch/pytorch/issues/973
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import numpy as np
import argparse
import tqdm
from typing import NamedTuple, Optional, List, Tuple, Union, Callable, Dict
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
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
import pickle
from torchvision.transforms import Compose
import torchmetrics
from torch import Tensor
from torch.utils.data import Subset

from trackertraincode.neuralnets.torchquaternion import quat_average, geodesicdistance
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
blinks = [
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


def _find_models(path: str):
    if os.path.isfile(path):
        return [path]
    else:
        return glob.glob(path)


class Poses(NamedTuple):
    hpb: NDArray[np.float64]
    xy: NDArray[np.float64]
    sz: NDArray[np.float64]
    pose_scales_tril: Optional[NDArray[np.float64]] = None
    coord_scales: Optional[NDArray[np.float64]] = None


class PosesWithStd(NamedTuple):
    hpb: NDArray[np.float64]
    xy: NDArray[np.float64]
    sz: NDArray[np.float64]
    hpb_std: NDArray[np.float64]
    xy_std: NDArray[np.float64]
    sz_std: NDArray[np.float64]
    pose_scales_tril: Optional[NDArray[np.float64]] = None
    coord_scales: Optional[NDArray[np.float64]] = None
    pose_scales_tril_std: Optional[NDArray[np.float64]] = None
    coord_scales_std: Optional[NDArray[np.float64]] = None

    @staticmethod
    def from_poses(poses: List[Poses]):
        by_field = defaultdict(list)
        for pose in poses:
            for field in Poses._fields:
                by_field[field].append(getattr(pose, field))
        items = {k + "_std": np.std(v, axis=0) for k, v in by_field.items()}
        items.update({k: np.average(v, axis=0) for k, v in by_field.items()})
        return PosesWithStd(**items)


def convertlabels(labels: dict) -> Poses:
    hpb = utils.as_hpb(Rotation.from_quat(labels.pop("pose")))
    coord = labels.pop("coord").numpy()
    xy = coord[..., :2]
    sz = coord[..., 2]
    labels = {k: v.numpy() for k, v in labels.items() if k in Poses._fields}
    return Poses(hpb, xy, sz, **labels)


def plot_coord(ax, pose: Union[Poses, PosesWithStd], **kwargs):
    line0: matplotlib.lines.Line2D = ax[0].plot(pose.xy[:1000, 0], **kwargs)[0]
    line1: matplotlib.lines.Line2D = ax[1].plot(pose.xy[:1000, 1], **kwargs)[0]
    line2: matplotlib.lines.Line2D = ax[2].plot(pose.sz[:1000], **kwargs)[0]
    if isinstance(pose, PosesWithStd):
        x = np.arange(len(pose.sz))
        ax[0].fill_between(
            x,
            pose.xy[:1000, 0] - pose.xy_std[:1000, 0],
            pose.xy[:1000, 0] + pose.xy_std[:1000, 0],
            alpha=0.2,
            color=line0.get_color(),
            **kwargs,
        )
        ax[1].fill_between(
            x,
            pose.xy[:1000, 1] - pose.xy_std[:1000, 1],
            pose.xy[:1000, 1] + pose.xy_std[:1000, 1],
            alpha=0.2,
            color=line1.get_color(),
            **kwargs,
        )
        ax[2].fill_between(
            x,
            pose.sz[:1000] - pose.sz_std[:1000],
            pose.sz[:1000] + pose.sz_std[:1000],
            alpha=0.2,
            color=line2.get_color(),
            **kwargs,
        )


def plot_hpb(ax, pose: Union[Poses, PosesWithStd], **kwargs):
    rad2deg = 180.0 / np.pi
    line0 = ax[0].plot(pose.hpb[:1000, 0] * rad2deg, **kwargs)[0]
    line1 = ax[1].plot(pose.hpb[:1000, 1] * rad2deg, **kwargs)[0]
    line2 = ax[2].plot(pose.hpb[:1000, 2] * rad2deg, **kwargs)[0]
    if isinstance(pose, PosesWithStd):
        x = np.arange(len(pose.sz))
        ymin = (pose.hpb - pose.hpb_std) * rad2deg
        ymax = (pose.hpb + pose.hpb_std) * rad2deg
        ax[0].fill_between(
            x, ymin[:1000, 0], ymax[:1000, 0], alpha=0.2, color=line0.get_color(), **kwargs
        )
        ax[1].fill_between(
            x, ymin[:1000, 1], ymax[:1000, 1], alpha=0.2, color=line1.get_color(), **kwargs
        )
        ax[2].fill_between(
            x, ymin[:1000, 2], ymax[:1000, 2], alpha=0.2, color=line2.get_color(), **kwargs
        )
    return line0


def _has_coord_cov_matrix(preds: Poses):
    return preds.coord_scales is not None and preds.coord_scales.shape[-2:] == (3, 3)


def visualize(preds: List[Union[Poses, PosesWithStd]], checkpoints: List[str]):
    def make_nice(axes):
        for ax in axes[:-1]:
            ax.xaxis.set_visible(False)
        axes[0].legend()
        for i, ax in enumerate(axes):
            ax.patch.set_visible(False)  # Hide Background
            ax2 = ax.twinx()
            ax2.yaxis.set_visible(False)
            ax2.set_zorder(ax.get_zorder() - 1)
            for a, b in blinks:
                ax2.bar(0.5 * (a + b), 1, width=b - a, bottom=0, color="yellow")
        pyplot.tight_layout()

    fig, axes = pyplot.subplots(6, 1, figsize=(18, 5))

    for i, pred in enumerate(preds):
        line = plot_hpb(axes[:3], pred)
        line.set_label(checkpoints[i])
        plot_coord(axes[3:6], pred)

    for ax, label in zip(axes, "yaw,pitch,roll,x,y,size".split(",")):
        ax.set(ylabel=label)

    make_nice(axes)

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

        return [fig, fig2]

    return [fig]


def report_blink_stability(poses_by_parameters: List[Poses]):
    xs = np.asarray([a for a, b in blinks] + [b for a, b in blinks], dtype=np.int64)
    lefts = xs - 5
    rights = xs + 5

    def mse(vals):
        diffsqr = np.square(vals[lefts] - vals[rights])
        return np.sqrt(np.mean(diffsqr, axis=0))

    def param_average_mse(name):
        return np.average([mse(getattr(poses, name)) for poses in poses_by_parameters], axis=0)

    for name in ["hpb", "sz", "xy"]:
        mse_val = np.atleast_1d(param_average_mse(name))
        if name == "hpb":
            mse_val *= 180.0 / np.pi
        print(f"\t {name:4s}: " + ", ".join(f"{x:0.2f}" for x in mse_val))


def closed_loop_tracking(model: eval.Predictor, loader):
    current_roi = None
    preds = []
    bar = tqdm.tqdm(total=len(loader.dataset))
    for sample in loader:
        image, roi = sample["image"], sample["roi"]
        if current_roi is not None:
            roi[...] = current_roi
        pred = model.predict_batch(image[None, ...], roi[None, ...])
        x0, y0, x1, y1 = pred["roi"][0]
        w, h = sample.meta.image_wh
        current_roi = torch.tensor([max(0.0, x0), max(0.0, y0), min(x1, w), min(y1, h)])
        preds.append(pred)
        bar.update(1)
    preds = Batch.collate(preds)
    poses = convertlabels(preds)
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
    loader: dtr.SampleBySampleLoader,
    prediction_func: Callable[[torch.nn.Module, dtr.SampleBySampleLoader], Poses],
    crop_size_factor: float,
) -> Union[Poses, PosesWithStd]:
    """Predictions from a single or a group of networks"""
    checkpoints = _find_models(path)
    print(f"Evaluating {path}. Found {len(checkpoints)} checkpoints.")
    preds = []
    for checkpoint in checkpoints:
        net = eval.Predictor(checkpoint, device=device, focus_roi_expansion_factor=crop_size_factor)
        preds.append(prediction_func(net, loader))
    aggregated = PosesWithStd.from_poses(preds) if len(preds) > 1 else next(iter(preds))
    return aggregated, preds


def main_open_loop(paths: List[str], device):
    loader = trackertraincode.pipelines.make_validation_loader("myself", return_single_samples=True)

    def process(path, crop_size_factor):
        return _track_multiple_networks(path, device, loader, open_loop_tracking, crop_size_factor)

    poses_by_checkpoints = defaultdict(list)
    for crop_size_factor in [1.0, 1.2]:
        poses_aggregated, poses_lists = zip(*[process(fn, crop_size_factor) for fn in paths])
        figs = visualize(poses_aggregated, paths)
        for fig in figs:
            fig.suptitle(f"cropsize={crop_size_factor:.1f}")
        for fn, poses_list in zip(paths, poses_lists):
            poses_by_checkpoints[fn] += poses_list
    pyplot.show()

    for name in paths:
        print(f"Checkpoint: {name}")
        report_blink_stability(poses_by_checkpoints[name])


def main_closed_loop(paths: List[str], device):
    loader = trackertraincode.pipelines.make_validation_loader("myself", return_single_samples=True)

    def process(path, crop_size_factor):
        return _track_multiple_networks(
            path, device, loader, closed_loop_tracking, crop_size_factor
        )

    for crop_size_factor in [1.0, 1.2]:
        poses_aggregated, poses_lists = zip(*[process(fn, crop_size_factor) for fn in paths])
        figs = visualize(poses_aggregated, paths)
        for fig in figs:
            fig.suptitle(f"closed-loop cropsize={crop_size_factor:.1f}")

    pyplot.show()


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
        ],
        default="none",
    )
    parser.add_argument("filename", nargs="+", help="input filenames", type=str)
    parser.add_argument("--vis", action="store_true", default=False)
    args = parser.parse_args()
    if args.mode == "open-loop":
        main_open_loop(args.filename, "cuda")
    elif args.mode == "closed-loop":
        main_closed_loop(args.filename, "cuda")
    elif args.mode == "pitch-yaw":
        main_analyze_pitch_vs_yaw(args.filename)
    elif args.mode == "noise-resist":
        if args.vis:
            main_vis_noise_resist(args.filename)
        else:
            main_analyze_noise_resist(args.filename)
    elif args.mode == "uncertainty-correlation":
        main_analyze_uncertainty_error_correlation(args.filename)
    else:
        raise RuntimeError(f"Unkown mode {args.mode}")
