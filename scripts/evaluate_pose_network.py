#!/usr/bin/env python
# coding: utf-8

# Seems to run a bit faster than with default settings and less bugged
# See https://github.com/pytorch/pytorch/issues/67864
from collections.abc import Sequence

from sklearn import get_config
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

from typing import Any, List, NamedTuple, Tuple, Dict, Callable, Literal, cast, Type, TypeAlias
import numpy as np
import argparse
import tqdm
import tabulate
import json
import os
import copy
import pprint
from numpy.typing import NDArray
from matplotlib import pyplot
import functools
import torch
from torch import Tensor
from collections import defaultdict
from os.path import commonprefix, relpath
import torchmetrics
import pandas as pd

from trackertraincode.datasets.batch import Batch, Metadata
import trackertraincode.datatransformation as dtr
import trackertraincode.pipelines
import trackertraincode.vis as vis
import trackertraincode.utils as utils
import trackertraincode.eval as eval

# from trackertraincode.eval import load_pose_network, Predictor, compute_opal_paper_alignment, PerspectiveCorrector

load_pose_network = functools.lru_cache(maxsize=1)(eval.load_pose_network)

# According to this https://gmv.cast.uark.edu/scanning/hardware/microsoft-kinect-resourceshardware/
# The horizontal field of view of the kinect is ..
BIWI_HORIZONTAL_FOV = 57.0


class RoiConfig(NamedTuple):
    expansion_factor: float = 1.1
    center_crop: bool = False
    use_head_roi: bool = True

    def __str__(self):
        crop = ["ROI", "CC"][self.center_crop]
        return f'{"(H_roi)" if self.use_head_roi else "(F_roi)"}{crop}{self.expansion_factor:0.1f}'


comprehensive_roi_configs = [
    RoiConfig(*x)
    for x in [
        (1.2, False),
        (1.1, False),
        (1.0, False),
        (1.2, False, False),
        (1.1, False, False),
        (1.0, False, False),
    ]
]


def determine_roi(sample: Batch, use_center_crop: bool):
    if not use_center_crop:
        return sample["roi"]
    w, h = sample.meta.image_wh
    return torch.tensor([0, 0, h, w], dtype=torch.float32)


EvalResults = Dict[str, Tensor]
BatchPerspectiveCorrector = Callable[[Tensor, EvalResults], EvalResults]

AlignmentScheme = Literal["perspective", "opal23", "none"]


class DrawPredictionsWithHistory:
    def __init__(self, name):
        self.index_by_individual = defaultdict(list)
        self.samples_chronologically = []
        self.name = name

    def print_viewed(self):
        print(self.name)
        print("by individual: ", pprint.pformat(dict(self.index_by_individual), compact=True))
        print("chronologically: ", pprint.pformat(self.samples_chronologically, compact=True))

    def __call__(self, gt_pred: Tuple[Batch, Dict[str, Tensor]]):
        gt, _ = gt_pred
        try:
            individual = gt["individual"].item()
        except KeyError:
            individual = "unkown"
        self.index_by_individual[individual].append(gt["index"].item())
        self.samples_chronologically.append(gt["index"].item())
        return vis.draw_prediction(gt_pred)


def interleaved(a, b):
    c = np.empty((a.size + b.size,), dtype=a.dtype)
    c[0::2] = a.ravel()
    c[1::2] = b.ravel()
    return c


def make_model_path_headers(model_paths: List[str]) -> List[str]:
    prefix = commonprefix(model_paths)
    return [relpath(m, prefix) for m in model_paths]


class PoseErrAverages(NamedTuple):
    euler_angles: tuple[float, float, float]
    geodesic: float
    rmse_pos: float
    rmse_size: float
    unweighted_nme_3d: float | None
    nme_2d: eval.KptNmeResults | None


class PoseDetectionResults(NamedTuple):
    size_errs: NDArray[Any]
    geodesic_errs: NDArray[Any]
    unweighted_nme_3d: NDArray[Any] | None
    averages: PoseErrAverages

    @staticmethod
    def from_metrics_output(
        results: dict[str, Tensor | eval.KptNmeResults],
    ) -> "PoseDetectionResults":
        poseerrs: NDArray[Any] = results["pose_errs"].cpu().numpy()
        geodesic_errs: NDArray[Any] = results["geodesic_errs"].cpu().numpy()
        eulererrs: NDArray[Any] = results["euler_errs"].cpu().numpy()
        if "uw_nme_3d" in results and "nme_2d" in results:
            uw_nme_3d = results["uw_nme_3d"].cpu().numpy()
            nme_2d = results["nme_2d"]
        else:
            uw_nme_3d = nme_2d = None
        assert isinstance(nme_2d, (eval.KptNmeResults, type(None)))
        e_posx, e_posy, e_size = poseerrs.T
        rmse_pos = np.sqrt(
            np.average(np.sum(np.square(np.vstack([e_posx, e_posy]).T), axis=1), axis=0)
        )
        rmse_size = np.sqrt(np.average(np.square(e_size)))
        averages = PoseErrAverages(
            euler_angles=(np.average(np.abs(eulererrs), axis=0) * utils.rad2deg).tolist(),
            geodesic=(np.average(geodesic_errs) * utils.rad2deg).tolist(),
            rmse_pos=(rmse_pos * 100.0).tolist(),
            rmse_size=(rmse_size * 100.0).tolist(),
            unweighted_nme_3d=np.average(uw_nme_3d).tolist() if uw_nme_3d is not None else None,
            nme_2d=nme_2d,
        )
        return PoseDetectionResults(
            size_errs=e_size,
            geodesic_errs=geodesic_errs,
            unweighted_nme_3d=uw_nme_3d,
            averages=averages,
        )


class BaseResultTableBuilder:
    Row: TypeAlias = Sequence[str | float | int]

    def __init__(self):
        self._header: list[str] = []
        self._non_averaged_rows: dict[str, list[BaseResultTableBuilder.Row]] = defaultdict(list)

    @staticmethod
    def _aggregate_col(col: pd.Series):
        if col.dtype in (np.float32, np.float64, np.float16):
            mean = col.mean()
            std = col.std()
            return f"{mean:0.2f} ± {std:0.2f}"
        elif (col[0] == col).all():
            return col[0]
        else:
            "???"

    def build(self) -> str:
        tables = []
        for name, rows in self._non_averaged_rows.items():
            t = pd.DataFrame(rows, columns=self._header)
            if len(t.index) > 1:
                # t = t.mean(axis=0, skipna=False, numeric_only=False)
                t = t.aggregate(self._aggregate_col)
                t = t.to_frame().T
            tables.append(t)
        table = cast(pd.DataFrame, pd.concat(tables, ignore_index=False))
        table = table.set_index("Data")
        return table.to_markdown(tablefmt="github", floatfmt=".2f")
        # return tabulate.tabulate(self._entries, self._header, tablefmt="github", floatfmt=".2f")

    def build_json_dict(self) -> dict[str, list[str | float | int]]:
        raise NotImplementedError()
        by_header = defaultdict(list)
        for row in self._entries:
            for name, value in zip(self._header, row):
                by_header[name].append(value)
        return by_header


class FacedetResultTableBuilder(BaseResultTableBuilder):
    def __init__(self):
        super().__init__()
        thresholds = eval.IsFaceAccuracy().thresholds.tolist()
        self._header = ["Data"] + [f"acc@{t:.2f}" for t in thresholds]

    def add_row(
        self,
        dataset: str,
        results: eval.AccuracyResults,
        data_aux_string: str | None,
    ):
        row_name: str = dataset + cast(str, data_aux_string if data_aux_string is not None else "")
        self._non_averaged_rows[row_name].append([row_name] + results.accuracies.tolist())


def report_facedetection(net_filename, builder: FacedetResultTableBuilder):
    loader = trackertraincode.pipelines.make_validation_loader("wider", return_single_samples=True)
    net = load_pose_network(net_filename, args.device)
    predictor = eval.Predictor(net, 1.1)
    metric = eval.IsFaceAccuracy()
    results: eval.AccuracyResults = predictor.evaluate(metric, loader)
    builder.add_row("wider", results, " / 1.1")


class PoseResultTableBuilder(BaseResultTableBuilder):
    data_name_table = {"aflw2k3d": "AFLW 2k 3d", "aflw2k3d_grimaces": "grimaces"}

    def __init__(self):
        super().__init__()
        # Position and size errors are measured relative to the ROI size. Hence in percent.
        self._header = [
            "Data",
            "Pitch°",
            "Yaw°",
            "Roll°",
            "Mean°",
            "Geodesic°",
            "XY%",
            "S%",
            "NME3d%",
            "NME2d%_30",
            "NME2d%_60",
            "NME2d%_90",
            "NME2d%_avg",
        ]

    def add_row(
        self,
        dataset: str,
        results: PoseErrAverages,
        data_aux_string: str | None,
    ):
        unweighted_nme_3d = (
            results.unweighted_nme_3d * 100 if results.unweighted_nme_3d is not None else "n/a"
        )
        nme_2d_30, nme_2d_60, nme_2d_90, nme_2d_avg = (
            ["n/a" for _ in range(4)]
            if results.nme_2d is None
            else [x * 100 for x in results.nme_2d]
        )
        row_name: str = self.data_name_table.get(dataset, dataset) + cast(
            str, data_aux_string if data_aux_string is not None else ""
        )
        self._non_averaged_rows[row_name].append(
            [row_name]
            + list(results.euler_angles)
            + [
                cast(float, np.average(results.euler_angles).tolist()),
                results.geodesic,
                results.rmse_pos,
                results.rmse_size,
                unweighted_nme_3d,
                nme_2d_30,
                nme_2d_60,
                nme_2d_90,
                nme_2d_avg,
            ]
        )


def compute_pred_keys(loader: dtr.SampleBySampleLoader, net: eval.InferenceNetwork):
    # Check if model and data support landmark prediction
    keys = ["coord", "pose", "roi"]
    sample = next(iter(loader))
    preds = net(
        torch.zeros((1, 1, net.input_resolution, net.input_resolution), device=net.device_for_input)
    )
    if "pt3d_68" in sample and "pt3d_68" in preds:
        keys.append("pt3d_68")
    return keys


def report(
    net_filename,
    data_name,
    roi_config: RoiConfig,
    args: argparse.Namespace,
    builder: PoseResultTableBuilder,
):
    alignment: AlignmentScheme = args.alignment_scheme

    loader = trackertraincode.pipelines.make_validation_loader(
        data_name, use_head_roi=roi_config.use_head_roi, return_single_samples=True
    )
    net = load_pose_network(net_filename, args.device)

    pred_keys = compute_pred_keys(loader, net)
    predictor = eval.Predictor(net, roi_config.expansion_factor)

    metrics = torchmetrics.MetricCollection({"pose_errs": eval.NormalizedXYSError()})
    if alignment == "none":
        metrics.add_metrics(
            {"geodesic_errs": eval.GeodesicError(), "euler_errs": eval.EulerAngleErrors()}
        )
    else:
        metrics.add_metrics(
            {
                "geodesic_errs": eval.AlignedRotationErrorMetric(
                    error_mode="geo", correction_mode=alignment, fov=BIWI_HORIZONTAL_FOV
                ),
                "euler_errs": eval.AlignedRotationErrorMetric(
                    error_mode="euler", correction_mode=alignment, fov=BIWI_HORIZONTAL_FOV
                ),
            }
        )
    if "pt3d_68" in pred_keys:
        metrics.add_metrics(
            {"uw_nme_3d": eval.UnweightedKptNME(), "nme_2d": eval.KptNME(dimensions=2)}
        )

    results: dict[str, Tensor | eval.KptNmeResults] = predictor.evaluate(metrics, loader)

    pose_results_intermediate = PoseDetectionResults.from_metrics_output(results)

    builder.add_row(
        dataset=data_name,
        results=pose_results_intermediate.averages,
        data_aux_string=" / " + str(roi_config),
    )

    if args.vis != "none":
        quantity = {
            "kpts": pose_results_intermediate.unweighted_nme_3d,
            "rot": pose_results_intermediate.geodesic_errs,
            "size": pose_results_intermediate.size_errs,
        }[args.vis]
        if quantity is None:
            print(f"Prediction for {args.vis} is not available.")
            return []

        order = np.ascontiguousarray(np.argsort(quantity)[::-1])
        loader = trackertraincode.pipelines.make_validation_loader(
            data_name, order=order, return_single_samples=True
        )

        def iter_gt_and_preds():
            for sample in loader:
                pred = predictor.predict_batch([sample["image"]], sample["roi"][None, ...])
                pred = Batch(Metadata(0, batchsize=1), pred)
                sample = dtr.batch.to_numpy(sample)
                pred = dtr.batch.to_numpy(pred)
                # There is only one sample to unpack. Needs to be done though.
                yield from zip(sample.undo_collate(), pred.undo_collate())

        history = DrawPredictionsWithHistory(data_name + "/" + net_filename)
        fig, btn = vis.matplotlib_plot_iterable(iter_gt_and_preds(), history)
        fig.suptitle(data_name + " / " + str(roi_config) + " / " + net_filename)
        return [fig, btn, history]
    else:
        return []


def run(args):
    gui = []

    pose_table_builders = defaultdict(PoseResultTableBuilder)
    facedet_table_builders = defaultdict(FacedetResultTableBuilder)

    if not args.comprehensive_roi:
        if args.roi_expansion is not None:
            roi_configs = [RoiConfig(expansion_factor=args.roi_expansion)]
        else:
            roi_configs = [RoiConfig()]
    else:
        assert args.roi_expansion is None, "Conflicting arguments"
        roi_configs = comprehensive_roi_configs

    datasets = args.ds.split("+")
    for net_filename in args.filenames:
        if args.hasface:
            builder = facedet_table_builders[net_filename]
            report_facedetection(net_filename, builder)
        for name in datasets:
            for roi_config in roi_configs:
                builder = (
                    pose_table_builders["average"]
                    if args.aggregate
                    else pose_table_builders[net_filename]
                )
                gui += report(net_filename, name, roi_config, args, builder)

    net_names = dict(
        zip(cast(Sequence[str], args.filenames), make_model_path_headers(args.filenames))
    )

    # if args.json:
    #     assert args.json.endswith(".json")
    #     print(f"writing {args.json}")
    #     with open(args.json, "w") as f:
    #         f.write(table_builder.build_json_dict())
    # else:

    builders = list(pose_table_builders.items()) + list(facedet_table_builders.items())

    for net_filename, builder in builders:
        print(f"## {net_names.get(net_filename, net_filename)}")
        print(builder.build())

    pyplot.show()
    print("Viewed samples:")
    for thing in gui:
        if not isinstance(thing, DrawPredictionsWithHistory):
            continue
        thing.print_viewed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pose networks")
    parser.add_argument(
        "filenames", help="filenames of checkpoint or onnx model file", type=str, nargs="*"
    )
    parser.add_argument(
        "--vis",
        dest="vis",
        help="visualization of worst",
        default="none",
        choices=["none", "kpts", "rot", "size"],
    )
    parser.add_argument("--hasface", action="store_true", default=False)
    parser.add_argument("--device", help="select device: cpu or cuda", default="cuda", type=str)
    parser.add_argument("--comprehensive-roi", action="store_true", default=False)
    parser.add_argument(
        "--alignment-scheme", choices=["perspective", "opal23", "none"], default="none"
    )
    parser.add_argument("--roi-expansion", default=None, type=float)
    # parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--aggregate", action="store_true", default=False)
    parser.add_argument("--ds", type=str, default="aflw2k3d")
    args = parser.parse_args()
    run(args)
