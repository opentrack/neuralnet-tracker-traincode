#!/usr/bin/env python
# coding: utf-8

# Seems to run a bit faster than with default settings and less bugged
# See https://github.com/pytorch/pytorch/issues/67864
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from typing import Any, List, NamedTuple
import numpy as np
import argparse
import tqdm
import tabulate
from numpy.typing import NDArray
from matplotlib import pyplot
from os.path import basename
import functools
import torch

from trackertraincode.datasets.batch import Batch
import trackertraincode.datatransformation as dtr
import trackertraincode.eval
import trackertraincode.pipelines
import trackertraincode.vis as vis
import trackertraincode.utils as utils

from trackertraincode.eval import load_pose_network, predict

load_pose_network = functools.lru_cache(maxsize=1)(load_pose_network)

class RoiConfig(NamedTuple):
    expansion_factor : float = 1.2
    center_crop : bool = False

    def __str__(self):
        crop = ['ROI','CC'][self.center_crop]
        return f'{crop}{self.expansion_factor:0.1f}'

normal_roi_configs = [ RoiConfig() ]
comprehensive_roi_configs = [ RoiConfig(*x) for x in [(1.2, False), (1.0, False), (0.6, True), (0.8, True) ] ]


def determine_roi(batch : Batch, use_center_crop : bool):
    if not use_center_crop:
        return batch['roi']
    w,h = batch.meta.image_wh
    b = batch.meta.batchsize
    return torch.tensor([0,0,h,w], dtype=torch.float32).expand((b,4))


def compute_predictions_and_targets(loader, net, keys, roi_config : RoiConfig):
    preds   = []
    targets = []
    first = True
    bar = tqdm.tqdm(total = len(loader.dataset))
    for batch in utils.iter_batched(loader, 32):
        batch = Batch.collate(batch)
        pred = predict(
            net, 
            batch['image'], 
            rois=determine_roi(batch, roi_config.center_crop), 
            focus_roi_expansion_factor=roi_config.expansion_factor)
        if first:
            keys = list(frozenset(pred.keys()).intersection(frozenset(keys)))
            first = False
        pred = Batch(batch.meta, **{ k:pred[k] for k in keys })
        batch = Batch(batch.meta, **{ k:batch[k] for k in batch })
        preds.append(pred)
        targets.append(batch)
        bar.update(batch.meta.batchsize)
    preds = dtr.collate_list_of_batches(preds)
    targets = dtr.collate_list_of_batches(targets)
    return preds, targets


def iterate_predictions(loader, preds : Batch):
    pred_iter = (dtr.to_numpy(s) for s in preds.iter_frames())
    sample_iter = (dtr.to_numpy(sample) for batch in loader for sample in dtr.undo_collate(batch))
    yield from zip(sample_iter, pred_iter)


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
        self._header = [ 'Model', 'Data', 'Yaw°', 'Pitch°', 'Roll°', 'Mean°', 'Geodesic°', 'XY%', 'S%' ]
        self._entries = []
    
    def add_row(self, model : str, data : str, euler_angles : List[float], geodesic : float, rmse_pos : float, rmse_size : float, data_aux_string = None):
        maxlen = 30
        if len(model) > maxlen+3:
            model = '...'+model[-maxlen:]
        data = self.data_name_table[data] + (data_aux_string if data_aux_string is not None else '')
        self._entries.append([model, data] + euler_angles + [ np.average(euler_angles).tolist(), geodesic, rmse_pos, rmse_size] )
    
    def build(self) -> str:
        return tabulate.tabulate(self._entries, self._header, tablefmt='github', floatfmt=".2f")



def report(net_filename, data_name, roi_config, args, builder : TableBuilder):
    loader = trackertraincode.pipelines.make_validation_loader(data_name)
    net = load_pose_network(net_filename, args.device)
    preds, targets = compute_predictions_and_targets(loader, net, ['coord','pose', 'roi', 'pt3d_68'], roi_config)
    # Position and size errors are measured relative to the ROI size. Hence in percent.
    poseerrs = trackertraincode.eval.PoseErr()(preds, targets)
    eulererrs = trackertraincode.eval.EulerAngleErrors()(preds, targets)
    e_rot, e_posx, e_posy, e_size = np.array(poseerrs).T
    rmse_pos = np.sqrt(np.average(np.sum(np.square(np.vstack([e_posx, e_posy]).T), axis=1), axis=0))
    rmse_size = np.sqrt(np.average(np.square(e_size)))
    builder.add_row(
        model=basename(net_filename),
        data=data_name,
        euler_angles=(np.average(np.abs(eulererrs), axis=0)*utils.rad2deg).tolist(),
        geodesic=(np.average(e_rot)*utils.rad2deg).tolist(),
        rmse_pos=(rmse_pos*100.).tolist(),
        rmse_size=(rmse_size*100.).tolist(),
        data_aux_string=' / ' + str(roi_config)
    )

    if args.vis:
        order = interleaved(np.argsort(e_rot)[::-1], np.argsort(e_size)[::-1])
        loader = trackertraincode.pipelines.make_validation_loader(data_name, order=order)
        new_preds = Batch(preds.meta, **{k:v[order] for k,v in preds.items()})
        new_preds.meta.batchsize = len(order)
        worst_rot_iter = iterate_predictions(loader, new_preds)
        fig, btn = vis.matplotlib_plot_iterable(worst_rot_iter, vis.draw_prediction)
        fig.suptitle(data_name + ' / ' + net_filename)
        return [fig, btn]
    else:
        return []


def run(args):
    gui = []
    table_builder = TableBuilder()
    roi_configs = comprehensive_roi_configs if args.comprehensive_roi else normal_roi_configs
    for net_filename in args.filenames:
        for name in [ 'aflw2k3d', 'aflw2k3d_grimaces']:
            for roi_config in roi_configs:
                gui += report(net_filename, name, roi_config, args, table_builder)
    print (table_builder.build())
    pyplot.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate pose networks")
    parser.add_argument('filenames', help='filenames of checkpoint or onnx model file', type=str, nargs='*')
    parser.add_argument('--no-vis', dest='vis', help='disable visualization', default=True, action='store_false')
    parser.add_argument('--device', help='select device: cpu or cuda', default='cuda', type=str)
    parser.add_argument('--comprehensive-roi', action='store_true', default=False)
    args = parser.parse_args()
    run(args)