#!/usr/bin/env python
# coding: utf-8

from typing import Any, List
import numpy as np
import argparse
import tqdm
import tabulate
from numpy.typing import NDArray
from matplotlib import pyplot
from os.path import basename

from trackertraincode.datasets.batch import Batch
import trackertraincode.datatransformation as dtr
import trackertraincode.eval
import trackertraincode.pipelines
import trackertraincode.vis as vis
import trackertraincode.utils as utils

from trackertraincode.eval import load_pose_network, predict


def compute_predictions_and_targets(loader, net, keys):
    preds   = []
    targets = []
    first = True
    bar = tqdm.tqdm(total = len(loader.dataset))
    for batch in utils.iter_batched(loader, 32):
        batch = Batch.collate(batch)
        pred = predict(net, batch['image'], rois=batch['roi'])
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
    def __init__(self):
        self._header = [ 'Model', 'Data', 'Yaw°', 'Pitch°', 'Roll°', 'Mean°', 'Geodesic°', 'XY%', 'S%' ]
        self._entries = []
    
    def add_row(self, model : str, data : str, euler_angles : List[float], geodesic : float, rmse_pos : float, rmse_size : float):
        maxlen = 30
        if len(model) > maxlen+3:
            model = '...'+model[-maxlen:]
        self._entries.append([model, data] + euler_angles + [ np.average(euler_angles).tolist(), geodesic, rmse_pos, rmse_size] )
    
    def build(self) -> str:
        return tabulate.tabulate(self._entries, self._header, tablefmt='github', floatfmt=".2f")


def report(name, net, args, builder : TableBuilder):
    loader = trackertraincode.pipelines.make_validation_loader(name)
    preds, targets = compute_predictions_and_targets(loader, net, ['coord','pose', 'roi', 'pt3d_68'])
    # Position and size errors are measured relative to the ROI size. Hence in percent.
    poseerrs = trackertraincode.eval.PoseErr()(preds, targets)
    eulererrs = trackertraincode.eval.EulerAngleErrors()(preds, targets)
    e_rot, e_posx, e_posy, e_size = np.array(poseerrs).T
    rmse_pos = np.sqrt(np.average(np.sum(np.square(np.vstack([e_posx, e_posy]).T), axis=1), axis=0))
    rmse_size = np.sqrt(np.average(np.square(e_size)))
    builder.add_row(
        model=basename(args.filename),
        data=name,
        euler_angles=(np.average(np.abs(eulererrs), axis=0)*utils.rad2deg).tolist(),
        geodesic=(np.average(e_rot)*utils.rad2deg).tolist(),
        rmse_pos=(rmse_pos*100.).tolist(),
        rmse_size=(rmse_size*100.).tolist()
    )

    if args.vis:
        order = interleaved(np.argsort(e_rot)[::-1], np.argsort(e_size)[::-1])
        loader = trackertraincode.pipelines.make_validation_loader(name, order=order)
        new_preds = Batch(preds.meta, **{k:v[order] for k,v in preds.items()})
        new_preds.meta.batchsize = len(order)
        worst_rot_iter = iterate_predictions(loader, new_preds)
        fig, btn = vis.matplotlib_plot_iterable(worst_rot_iter, vis.draw_prediction)
        fig.suptitle(name)
        return [fig, btn]
    else:
        return []


def run(args):
    net = load_pose_network(args.filename, args.device)
    gui = []
    table_builder = TableBuilder()
    for name in [ 'aflw2k3d', 'aflw2k3d_grimaces']:
        gui += report(name, net, args, table_builder)
    print (table_builder.build())
    pyplot.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate pose networks")
    parser.add_argument('filename', help='filename of checkpoint or onnx model file', type=str)
    parser.add_argument('--no-vis', dest='vis', help='disable visualization', default=True, action='store_false')
    parser.add_argument('--device', help='select device: cpu or cuda', default='cuda', type=str)
    parser.add_argument('--res', dest='input_resolution', help='input resolution for loaded models where it is not clear', default=129, type=int)
    parser.add_argument('--auto-level', dest='auto_level', help='automatically adjust brightness levels for maximum contrast within the roi', default=False, action='store_true')
    args = parser.parse_args()
    run(args)