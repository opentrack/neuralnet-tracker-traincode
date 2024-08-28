#!/usr/bin/env python
# coding: utf-8

# Seems to run a bit faster than with default settings and less bugged
# See https://github.com/pytorch/pytorch/issues/67864
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from typing import Any, List, NamedTuple, Tuple, Dict, Callable
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
from os.path import basename
import functools
import torch
from torch import Tensor
from collections import defaultdict
from os.path import commonprefix, relpath

from trackertraincode.datasets.batch import Batch, Metadata
import trackertraincode.datatransformation as dtr
import trackertraincode.eval
import trackertraincode.pipelines
import trackertraincode.vis as vis
import trackertraincode.utils as utils
import trackertraincode.neuralnets.modelcomponents as modelcomponents

from trackertraincode.eval import load_pose_network, predict

load_pose_network = functools.lru_cache(maxsize=1)(load_pose_network)

# According to this https://gmv.cast.uark.edu/scanning/hardware/microsoft-kinect-resourceshardware/
# The horizontal field of view of the kinect is ..
BIWI_HORIZONTAL_FOV = 57.


class RoiConfig(NamedTuple):
    expansion_factor : float = 1.1
    center_crop : bool = False
    use_head_roi : bool = True

    def __str__(self):
        crop = ['ROI','CC'][self.center_crop]
        return f'{"(H_roi)" if self.use_head_roi else "(F_roi)"}{crop}{self.expansion_factor:0.1f}'

normal_roi_configs = [ RoiConfig() ]
comprehensive_roi_configs = [ RoiConfig(*x) for x in [
    (1.2, False),
    (1.1, False),
    (1.0, False), 
    (1.2, False, False),
    (1.1, False, False),
    (1.0, False, False), 
]]


def determine_roi(sample : Batch, use_center_crop : bool):
    if not use_center_crop:
        return sample['roi']
    w,h = sample.meta.image_wh
    return torch.tensor([0,0,h,w], dtype=torch.float32)


EvalResults = Dict[str, Tensor]
BatchPerspectiveCorrector = Callable[[Tensor,EvalResults],EvalResults]

def make_perspective_corrector(fov : float) -> BatchPerspectiveCorrector:
    '''
    Args: 
        fov Horizontal FOV in degree
    '''
    corrector = modelcomponents.PerspectiveCorrector(fov)
    def apply_to_batch(image_sizes : Tensor, b : Dict[str,Tensor]):
        assert 'pt3d_68' not in b, "Unsupported. Must be computed after correction."
        out = copy.copy(b)
        out['pose'] = corrector.corrected_rotation(image_sizes, b['coord'], b['pose'])
        return out
    return apply_to_batch


def compute_predictions_and_targets(loader, net, keys, roi_config : RoiConfig, perspective_corrector : BatchPerspectiveCorrector | None) -> Tuple[EvalResults, EvalResults]:
    """
    Return:
        Prediction and GT, each in a dict.
    """
    preds   = defaultdict(list)
    targets = defaultdict(list)
    first = True
    bar = tqdm.tqdm(total = len(loader.dataset))
    for batch in utils.iter_batched(loader, 32):
        images = [ sample['image'] for sample in batch ]
        rois = torch.stack([ determine_roi(sample, roi_config.center_crop) for sample in batch ])
        pred = predict(
            net, 
            images, 
            rois=rois, 
            focus_roi_expansion_factor=roi_config.expansion_factor)
        if first:
            keys = list(frozenset(pred.keys()).intersection(frozenset(keys)))
            first = False
        if perspective_corrector is not None:
            pred = { k:pred[k] for k in keys }
            image_sizes = torch.as_tensor([ img.shape[:2][::-1] for img in images ], device=images[0].device)
            pred = perspective_corrector(image_sizes, pred)
        for k in keys:
            preds[k].append(pred[k])
            targets[k].append(torch.stack([sample[k] for sample in batch]))
        bar.update(len(batch))
    preds = { k:torch.cat(v) for k,v in preds.items() }
    targets = { k:torch.cat(v) for k,v in targets.items() }
    return preds, targets


def zip_gt_with_pred(loader, preds : Batch):
    '''
    Returns iterator over tuples of the Batch type.
    '''
    pred_iter = (dtr.to_numpy(s) for s in preds.iter_frames())
    sample_iter = (dtr.to_numpy(sample) for batch in loader for sample in dtr.undo_collate(batch))
    yield from zip(sample_iter, pred_iter)


class DrawPredictionsWithHistory:
    def __init__(self, name):
        self.index_by_individual = defaultdict(list)
        self.name = name
    
    def print_viewed(self):
        return print(self.name + ":\n" + pprint.pformat(dict(self.index_by_individual), compact=True))

    def __call__(self, gt_pred : Tuple[Batch, Dict[str,Tensor]]):
        gt, _ = gt_pred
        try:
            individual = gt['individual'].item()
        except KeyError:
            individual = "unkown"
        self.index_by_individual[individual].append(gt['index'].item())
        return vis.draw_prediction(gt_pred)


def interleaved(a,b):
    c = np.empty((a.size + b.size,), dtype=a.dtype)
    c[0::2] = a.ravel()
    c[1::2] = b.ravel()
    return c


class TableBuilder:
    data_name_table = {
        'aflw2k3d' : 'AFLW 2k 3d',
        'aflw2k3d_grimaces' : 'grimaces'
    }

    def __init__(self):
        self._header = [ 'Data', 'Pitch°', 'Yaw°', 'Roll°', 'Mean°', 'Geodesic°', 'XY%', 'S%', 'NME3d%', 'NME2d%_30', 'NME2d%_60', 'NME2d%_90', 'NME2d%_avg' ]
        self._entries_by_model = defaultdict(list)
    
    def add_row(self, model : str, data : str, euler_angles : List[float], geodesic : float, rmse_pos : float, rmse_size : float, unweighted_nme_3d, nme_2d, data_aux_string = None):
        unweighted_nme_3d = unweighted_nme_3d*100 if unweighted_nme_3d is not None else 'n/a'
        nme_2d_30, nme_2d_60, nme_2d_90, nme_2d_avg = ['/na' for _ in range(4)] if nme_2d is None else [x*100 for x in nme_2d]
        data = self.data_name_table.get(data, data) + (data_aux_string if data_aux_string is not None else '')
        self._entries_by_model[model] += [[data] + euler_angles + [ np.average(euler_angles).tolist(), geodesic, rmse_pos, rmse_size, unweighted_nme_3d, nme_2d_30, nme_2d_60, nme_2d_90, nme_2d_avg]]
    
    def build(self) -> str:
        prefix = commonprefix(list(self._entries_by_model.keys()))
        nicer_model_paths = {
            m:relpath(m,prefix) for m in self._entries_by_model.keys()
        }
        string_rows = []
        for model, rows in self._entries_by_model.items():
            string_rows += [ nicer_model_paths[model] ]
            string_rows += tabulate.tabulate(rows, self._header, tablefmt='github', floatfmt=".2f").splitlines()
        return '\n'.join(string_rows)

    def build_json(self) -> str:
        prefix = commonprefix(list(map(os.path.dirname,self._entries_by_model.keys())))
        def model_table(rows):
            by_header = defaultdict(list)
            for row in rows:
                for name, value in zip(self._header, row):
                    by_header[name].append(value)
            return by_header
        table = {
            relpath(m,prefix):model_table(rows) for m,rows in self._entries_by_model.items() }
        return json.dumps(table, indent=2)



def report(net_filename, data_name, roi_config : RoiConfig, args : argparse.Namespace, builder : TableBuilder):
    loader = trackertraincode.pipelines.make_validation_loader(data_name, use_head_roi=roi_config.use_head_roi)
    net = load_pose_network(net_filename, args.device)
    if data_name == 'biwi':
        if args.perspective_correction:
            correction_func = make_perspective_corrector(BIWI_HORIZONTAL_FOV)
        else:
            correction_func = None
        preds, targets = compute_predictions_and_targets(loader, net, ['coord','pose', 'roi'], roi_config, correction_func)

    else:
        preds, targets = compute_predictions_and_targets(loader, net, ['coord','pose', 'roi', 'pt3d_68'], roi_config, None)
    # Position and size errors are measured relative to the ROI size. Hence in percent.
    poseerrs = trackertraincode.eval.PoseErr()(preds, targets)
    eulererrs = trackertraincode.eval.EulerAngleErrors()(preds, targets)
    e_rot, e_posx, e_posy, e_size = np.array(poseerrs).T
    rmse_pos = np.sqrt(np.average(np.sum(np.square(np.vstack([e_posx, e_posy]).T), axis=1), axis=0))
    rmse_size = np.sqrt(np.average(np.square(e_size)))
    if 'pt3d_68' in preds:
        uw_nme_3d = trackertraincode.eval.UnweightedKptNME()(preds, targets)
        nme_2d = trackertraincode.eval.KptNME(dimensions=2)(preds, targets)
    else:
        uw_nme_3d = nme_2d = None
    builder.add_row(
        model=net_filename,
        data=data_name,
        euler_angles=(np.average(np.abs(eulererrs), axis=0)*utils.rad2deg).tolist(),
        geodesic=(np.average(e_rot)*utils.rad2deg).tolist(),
        rmse_pos=(rmse_pos*100.).tolist(),
        rmse_size=(rmse_size*100.).tolist(),
        data_aux_string=' / ' + str(roi_config),
        unweighted_nme_3d=np.average(uw_nme_3d) if uw_nme_3d is not None else None,
        nme_2d=nme_2d
    )

    if args.vis != 'none':
        quantity = {
            'kpts' : uw_nme_3d,
            'rot' : e_rot,
            'size' : e_size
        }[args.vis]
        if quantity is None:
            print(f"Prediction for {args.vis} is not available.")
            return []

        order = np.ascontiguousarray(np.argsort(quantity)[::-1])
        loader = trackertraincode.pipelines.make_validation_loader(data_name, order=order)
        new_preds = Batch(Metadata(0, batchsize=len(order)), **{k:v[order] for k,v in preds.items()})
        worst_rot_iter = zip_gt_with_pred(loader, new_preds)
        history = DrawPredictionsWithHistory(data_name + '/' + net_filename)
        fig, btn = vis.matplotlib_plot_iterable(worst_rot_iter, history)
        fig.suptitle(data_name + ' / ' + str(roi_config) + ' / ' + net_filename)
        return [fig, btn, history]
    else:
        return []


def run(args):
    gui = []
    table_builder = TableBuilder()
    roi_configs = comprehensive_roi_configs if args.comprehensive_roi else normal_roi_configs
    datasets = args.ds.split('+')
    for net_filename in args.filenames:
        for name in datasets:
            for roi_config in roi_configs:
                gui += report(net_filename, name, roi_config, args, table_builder)
    if args.json:
        print (f"writing {args.json}")
        with open(args.json, 'w') as f:
            f.write(table_builder.build_json())
    else:
        print (table_builder.build())
    pyplot.show()
    print ("Viewed samples per individual:")
    for thing in gui:
        if not isinstance(thing, DrawPredictionsWithHistory):
            continue
        thing.print_viewed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate pose networks")
    parser.add_argument('filenames', help='filenames of checkpoint or onnx model file', type=str, nargs='*')
    parser.add_argument('--vis', dest='vis', help='visualization of worst', default='none', choices=['none','kpts','rot','size'])
    parser.add_argument('--device', help='select device: cpu or cuda', default='cuda', type=str)
    parser.add_argument('--comprehensive-roi', action='store_true', default=False)
    parser.add_argument('--perspective-correction', action='store_true', default=False)
    parser.add_argument('--json', type=str, default=None)
    parser.add_argument('--ds', type=str, default='aflw2k3d')
    args = parser.parse_args()
    run(args)