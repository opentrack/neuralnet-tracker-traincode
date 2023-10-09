#!/usr/bin/env python
# coding: utf-8

import numpy as np
import argparse
import tqdm
from typing import NamedTuple, Optional, List, Tuple
from scipy.spatial.transform import Rotation
from matplotlib import pyplot
import itertools
from collections import defaultdict
import torch

from trackertraincode.datasets.batch import Batch
import trackertraincode.datatransformation as dtr
import trackertraincode.pipelines
import trackertraincode.vis as vis
import trackertraincode.utils as utils
import trackertraincode.train as train

from trackertraincode.eval import load_pose_network, predict


# Frame ranges where eyes closed
blinks = [
    (42,85),
    (103,157),
    (189,190),
    (209,210),
    (222,223),
    (254,255),
    (298,299),
    (360,363),
    (398,404),
    (466,472),
    (504,507),
    (567,597),
]


class Poses(NamedTuple):
    hpb : np.ndarray
    xy : np.ndarray
    sz : np.ndarray
    pose_scales_tril : Optional[np.ndarray] = None
    eyeparam : Optional[np.ndarray] = None
    coord_scales : Optional[np.ndarray] = None


def convertlabels(labels : dict) -> Poses:
    hpb = utils.as_hpb(Rotation.from_quat(labels.pop('pose')))
    coord = labels.pop('coord').numpy()
    xy = coord[...,:2]
    sz = coord[...,2]
    labels = { k:v.numpy() for k,v in labels.items() if k in Poses._fields }
    return Poses(
        hpb, 
        xy, 
        sz,
        **labels)


def plot_coord(ax, xy, sz, **kwargs):
    ax.plot(xy[:1000,0] / 400 - 0.5, label='x', c='r', **kwargs)
    ax.plot(xy[:1000,1] / 400 - 0.5, label='y', c='g', **kwargs)
    ax.plot(sz[:1000] / 100 - 1., label='size', c='k', **kwargs)

def plot_hpb(ax, hpb, **kwargs):
    ax.plot(hpb[:1000,0], label='h', c='r', **kwargs)
    ax.plot(hpb[:1000,1], label='p', c='g', **kwargs)
    ax.plot(hpb[:1000,2], label='r', c='b', **kwargs)




def visualize(preds : List[Poses]):
    alphas = [0.5] * len(preds)
    alphas[0] = 1.

    fig, axes = pyplot.subplots(4,1,figsize=(18,5))  

    for i, alpha, pred in zip(itertools.count(0), alphas, preds):
        plot_hpb(axes[0], pred.hpb, alpha=alpha)
        plot_coord(axes[1], pred.xy, pred.sz, alpha=alpha)
        if pred.pose_scales_tril is not None:
            cov = pred.pose_scales_tril @ pred.pose_scales_tril.swapaxes(-1,-2)
            for i in range(3):
                for j in range(i+1):
                    axes[2].plot(cov[...,i,j], label=f'rv[{i},{j}]')
            axes[2].plot(np.square(pred.coord_scales[...,2]),label='sz')
        if pred.eyeparam is not None:
            axes[3].plot(pred.eyeparam[:,0], label='left eye')
            axes[3].plot(pred.eyeparam[:,1], label='right eye')

    for ax in axes:
        ax.patch.set_visible(False) # Hide Background
        ax2 = ax.twinx()
        ax2.set_zorder(ax.get_zorder()-1)
        for a,b in blinks:
            ax2.bar(0.5*(a+b), 1, width=b-a, bottom=0, color='yellow')
        ax.legend()
    return fig


def report_blink_stability(poses_by_parameters : List[Poses]):
    xs = np.asarray([ a for a,b in blinks] + [b for a,b in blinks ], dtype=np.int64)
    lefts = xs - 5
    rights = xs + 5

    def mse(vals):
        fuck = np.square(vals[lefts]-vals[rights])
        return np.sqrt(np.mean(fuck, axis=0))

    def param_average_mse(name):
        return np.average([mse(getattr(poses,name)) for poses in poses_by_parameters], axis=0)
    
    for name in ['hpb', 'sz', 'xy']:
        mse_val = np.atleast_1d(param_average_mse(name))
        if name == 'hpb':
            mse_val *= 180./np.pi
        print (f"\t {name:4s}: "+", ".join(f'{x:0.2f}' for x in mse_val))


def predict_dataset(net, loader, crop_size_factor) -> Poses:
    bar = tqdm.tqdm(total = len(loader.dataset))
    def predict_sample_list(samples : List[Batch]):
        images = [ s['image'] for s in samples ]
        rois = torch.stack([ s['roi'] for s in samples ])
        out = predict(net, images, rois, focus_roi_expansion_factor=crop_size_factor)
        bar.update(len(samples))
        return out
    preds = [ predict_sample_list(batch) for batch in utils.iter_batched(loader,32) ]
    preds = Batch.collate(preds)
    poses = convertlabels(preds)
    return poses


def main_open_loop(checkpoints : List[str], device):
    loader = trackertraincode.pipelines.make_validation_loader('myself')

    def process(checkpoint, crop_size_factor):
        net = load_pose_network(checkpoint, device)
        return predict_dataset(net, loader, crop_size_factor)

    poses_by_checkpoints = defaultdict(list)
    for crop_size_factor in [1., 1.2]:
        poses = [ process(fn, crop_size_factor) for fn in checkpoints ]
        fig = visualize(poses)
        fig.suptitle(f'cropsize={crop_size_factor:.1f}')
        for fn, pose in zip(checkpoints, poses):
            poses_by_checkpoints[fn].append(pose)
    pyplot.show()

    for name in checkpoints:
        print (f"Checkpoint: {name}")
        report_blink_stability(poses_by_checkpoints[name])


def closed_loop_tracking(model, loader, crop_size_factor):
    current_roi = None
    preds = []
    bar = tqdm.tqdm(total = len(loader.dataset))
    for sample in loader:
        image, roi = sample['image'], sample['roi']
        if current_roi is not None:
            roi[...] = current_roi
        pred = predict(model, image[None,...], roi[None,...], focus_roi_expansion_factor=crop_size_factor)
        x0, y0, x1, y1 = pred['roi'][0]
        w, h = sample.meta.image_wh
        current_roi = torch.tensor([max(0., x0), max(0., y0), min(x1, w), min(y1, h)])
        preds.append(pred)
        bar.update(1)
    preds = Batch.collate(preds)
    poses = convertlabels(preds)
    return poses


def main_closed_loop(checkpoints : List[str], device):
    loader = trackertraincode.pipelines.make_validation_loader('myself')

    poses_by_params = defaultdict(list)

    for checkpoint in checkpoints:
        net = load_pose_network(checkpoint, device)
        for crop_size_factor in [1., 1.2]:
            poses = closed_loop_tracking(net, loader, crop_size_factor)
            poses_by_params[crop_size_factor].append(poses)

    for crop_size_factor, poses in poses_by_params.items():
        fig = visualize(poses)
        fig.suptitle(f'closed-loop cropsize={crop_size_factor:.1f}')
    pyplot.show()


def _create_biwi_sections_loader():
    intervals = [(145, 216), (1360,1464), (3030,3120), (8020,8100), (6570,6600), (9030,9080)]
    indices = np.concatenate([
        np.arange(a,b) for a,b in intervals
    ])
    loader = trackertraincode.pipelines.make_validation_loader('biwi', order=indices)
    sequence_starts = np.cumsum([0] + [(b-a) for a,b in intervals])
    return loader, sequence_starts


def main_analyze_pitch_vs_yaw(checkpoints : List[str]):
    fig, axes = pyplot.subplots(2,1,figsize=(20,5))

    def predict_all_nets(loader):
        poses_vs_model = {}
        for checkpoint in checkpoints:
            net = load_pose_network(checkpoint, 'cuda')
            poses = predict_dataset(net, loader, crop_size_factor=1.1)
            poses.hpb[...] *= 180./np.pi
            poses_vs_model[checkpoint] = poses
        return poses_vs_model

    loader = trackertraincode.pipelines.make_validation_loader('myself-yaw')
    poses_vs_model = predict_all_nets(loader)
    del loader

    ax = axes[0]
    for name, poses in poses_vs_model.items():
        ax.scatter(poses.hpb[:,0], poses.hpb[:,1], label=name, s=5.)
    ax.set(xlabel='yaw', ylabel='pitch')
    ax.legend()
    ax.axhline(0.,color='k')
    ax.axvline(0.,color='k')

    loader, sequence_starts = _create_biwi_sections_loader()
    poses_vs_model = predict_all_nets(loader)

    ax = axes[1]
    colors = 'rgbcmy'
    alphas = [0.5] * len(checkpoints)
    alphas[0] = 1.
    for j, (name, poses) in enumerate(poses_vs_model.items()):
        assert poses.hpb.shape[0] == sequence_starts[-1]
        for i,(a,b) in enumerate(zip(sequence_starts[:-1],sequence_starts[1:])):
            hpb = poses.hpb[a:b]
            #order = np.argsort(hpb[:,0])
            ax.plot(hpb[:,0], hpb[:,1], c=colors[i], alpha=alphas[j])
    ax.set(xlabel='yaw', ylabel='pitch')
    ax.legend()
    ax.axhline(0.,color='k')
    ax.axvline(0.,color='k')

    pyplot.show()


if __name__ == '__main__':
    np.seterr(all='raise')
    parser = argparse.ArgumentParser(description="Trains the model")
    parser.add_argument('filename', nargs='+', help='filename of checkpoint or onnx model file', type=str)
    parser.add_argument('--closed-loop', action='store_true', default=False)
    parser.add_argument('--pitch-yaw', action='store_true', default=False)
    parser.add_argument('--open-loop', action='store_true', default=False)
    args = parser.parse_args()
    if not (args.closed_loop or args.pitch_yaw):
        args.open_loop = True
    if args.open_loop:
        main_open_loop(args.filename, 'cuda')
    if args.closed_loop:
        main_closed_loop(args.filename, 'cpu')
    if args.pitch_yaw:
        main_analyze_pitch_vs_yaw(args.filename)