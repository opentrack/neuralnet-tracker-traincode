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
from torchvision.transforms import Compose
from trackertraincode.neuralnets.torchquaternion import quat_average, geodesicdistance
import glob
import matplotlib
import matplotlib.lines
import pickle

from trackertraincode.datasets.batch import Batch
import trackertraincode.datatransformation as dtr
import trackertraincode.pipelines
import trackertraincode.vis as vis
import trackertraincode.utils as utils
import trackertraincode.train as train
import trackertraincode.neuralnets.torchquaternion as quat
import trackertraincode.neuralnets.math

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


def _find_models(path : str):
    if os.path.isfile(path):
        return [ path ]
    else:
        return glob.glob(path)


class Poses(NamedTuple):
    hpb : NDArray[np.float64]
    xy  : NDArray[np.float64]
    sz  : NDArray[np.float64]
    pose_scales_tril : Optional[NDArray[np.float64]] = None
    coord_scales     : Optional[NDArray[np.float64]] = None


class PosesWithStd(NamedTuple):
    hpb : NDArray[np.float64]
    xy  : NDArray[np.float64]
    sz  : NDArray[np.float64]
    hpb_std : NDArray[np.float64]
    xy_std : NDArray[np.float64]
    sz_std : NDArray[np.float64]
    pose_scales_tril : Optional[NDArray[np.float64]] = None
    coord_scales     : Optional[NDArray[np.float64]] = None
    pose_scales_tril_std : Optional[NDArray[np.float64]] = None
    coord_scales_std     : Optional[NDArray[np.float64]] = None

    @staticmethod
    def from_poses(poses : List[Poses]):
        by_field = defaultdict(list)
        for pose in poses:
            for field in Poses._fields:
                by_field[field].append(getattr(pose,field))
        items = {k+'_std':np.std(v,axis=0) for k,v in by_field.items()}
        items.update({k:np.average(v,axis=0) for k,v in by_field.items()})
        return PosesWithStd(**items)


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


def plot_coord(ax, pose : Union[Poses,PosesWithStd], **kwargs):
    line0 : matplotlib.lines.Line2D = ax[0].plot(pose.xy[:1000,0], **kwargs)[0]
    line1 : matplotlib.lines.Line2D = ax[1].plot(pose.xy[:1000,1], **kwargs)[0]
    line2 : matplotlib.lines.Line2D = ax[2].plot(pose.sz[:1000]  , **kwargs)[0]
    if isinstance(pose, PosesWithStd):
        x = np.arange(len(pose.sz))
        ax[0].fill_between(x, pose.xy[:1000,0]-pose.xy_std[:1000,0], pose.xy[:1000,0]+pose.xy_std[:1000,0], alpha=0.2, color=line0.get_color(), **kwargs)
        ax[1].fill_between(x, pose.xy[:1000,1]-pose.xy_std[:1000,1], pose.xy[:1000,1]+pose.xy_std[:1000,1], alpha=0.2, color=line1.get_color(), **kwargs)
        ax[2].fill_between(x, pose.sz[:1000]  -pose.sz_std[:1000]  , pose.sz[:1000]  +pose.sz_std[:1000]  , alpha=0.2, color=line2.get_color(), **kwargs)


def plot_hpb(ax, pose : Union[Poses,PosesWithStd], **kwargs):
    rad2deg = 180./np.pi
    line0 = ax[0].plot(pose.hpb[:1000,0]*rad2deg, **kwargs)[0]
    line1 = ax[1].plot(pose.hpb[:1000,1]*rad2deg, **kwargs)[0]
    line2 = ax[2].plot(pose.hpb[:1000,2]*rad2deg, **kwargs)[0]
    if isinstance(pose, PosesWithStd):
        x = np.arange(len(pose.sz))
        ymin = (pose.hpb - pose.hpb_std)*rad2deg
        ymax = (pose.hpb + pose.hpb_std)*rad2deg
        ax[0].fill_between(x, ymin[:1000,0], ymax[:1000,0], alpha=0.2, color=line0.get_color(), **kwargs)
        ax[1].fill_between(x, ymin[:1000,1], ymax[:1000,1], alpha=0.2, color=line1.get_color(), **kwargs)
        ax[2].fill_between(x, ymin[:1000,2], ymax[:1000,2], alpha=0.2, color=line2.get_color(), **kwargs)
    return line0


def _has_coord_cov_matrix(preds : Poses):
    return preds.coord_scales is not None and preds.coord_scales.shape[-2:]==(3,3)


def visualize(preds : List[Union[Poses,PosesWithStd]], checkpoints : List[str]):
    def make_nice(axes):
        for ax in axes[:-1]:
            ax.xaxis.set_visible(False)
        axes[0].legend()
        for i, ax in enumerate(axes):
            ax.patch.set_visible(False) # Hide Background
            ax2 = ax.twinx()
            ax2.yaxis.set_visible(False)
            ax2.set_zorder(ax.get_zorder()-1)
            for a,b in blinks:
                ax2.bar(0.5*(a+b), 1, width=b-a, bottom=0, color='yellow')
        pyplot.tight_layout()

    fig, axes = pyplot.subplots(6,1,figsize=(18,5))  

    for i, pred in enumerate(preds):
        line = plot_hpb(axes[:3], pred)
        line.set_label(checkpoints[i])
        plot_coord(axes[3:6], pred)
    
    for ax, label in zip(axes, 'yaw,pitch,roll,x,y,size'.split(',')):
        ax.set(ylabel=label)

    make_nice(axes)

    if any(p.pose_scales_tril is not None for p in preds):
        axes_needed = 12 if any(_has_coord_cov_matrix(p) for p in preds) else 7

        fig2, axes = pyplot.subplots(axes_needed,1,figsize=(18,5))
        for checkpoint, pred in zip(checkpoints, preds):
            if pred.pose_scales_tril is None:
                continue
            ylim = np.amin(pred.pose_scales_tril), np.amax(pred.pose_scales_tril)
            k = 0
            for i in range(3):
                for j in range(i+1):
                    axes[k].plot(pred.pose_scales_tril[...,i,j], label=checkpoint)
                    axes[k].set(ylabel=f'r cov[{i},{j}]')
                    axes[k].set(ylim=ylim)
                    k += 1
            if _has_coord_cov_matrix(pred):
                axes[k+0].plot(pred.coord_scales[...,2,2])
                axes[k+1].plot(pred.coord_scales[...,0,0])
                axes[k+2].plot(pred.coord_scales[...,1,1])
                axes[k+3].plot(pred.coord_scales[...,1,0])
                axes[k+4].plot(pred.coord_scales[...,2,0])
                axes[k+5].plot(pred.coord_scales[...,2,1])
                for i, label in zip(range(k,k+6),['y-sz', 'x-sz', 'sz', 'x', 'y', 'x-y']):
                    axes[i].set(ylabel=label)
            else:
                axes[k].plot(pred.coord_scales[...,2])
                axes[k].set(ylabel='sz')
        make_nice(axes)
        
        return [fig, fig2]

    return [fig]


def report_blink_stability(poses_by_parameters : List[Poses]):
    xs = np.asarray([ a for a,b in blinks] + [b for a,b in blinks ], dtype=np.int64)
    lefts = xs - 5
    rights = xs + 5

    def mse(vals):
        diffsqr = np.square(vals[lefts]-vals[rights])
        return np.sqrt(np.mean(diffsqr, axis=0))

    def param_average_mse(name):
        return np.average([mse(getattr(poses,name)) for poses in poses_by_parameters], axis=0)
    
    for name in ['hpb', 'sz', 'xy']:
        mse_val = np.atleast_1d(param_average_mse(name))
        if name == 'hpb':
            mse_val *= 180./np.pi
        print (f"\t {name:4s}: "+", ".join(f'{x:0.2f}' for x in mse_val))


def predict_dataset(net, loader, crop_size_factor, return_gt : bool = False) -> Poses:
    bar = tqdm.tqdm(total = len(loader.dataset))
    def predict_sample_list(samples : List[Batch]):
        images = [ s['image'] for s in samples ]
        rois = torch.stack([ s['roi'] for s in samples ])
        out = predict(net, images, rois, focus_roi_expansion_factor=crop_size_factor)
        bar.update(len(samples))
        return out
    if return_gt:
        def extract_gt(samples : List[Batch]):
            for s in samples:
                s = copy.copy(s)
                del s['image']
                yield s
        preds, gt = zip(*((predict_sample_list(batch),list(extract_gt(batch))) for batch in utils.iter_batched(loader,32)))
        preds = Batch.collate(preds)
        gt = Batch.collate(sum(gt,start=[]))
        return preds, gt
    else:
        preds = [ predict_sample_list(batch) for batch in utils.iter_batched(loader,32) ]
        preds = Batch.collate(preds)
        return preds


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


def _track_multiple_networks(
    path : str, 
    device : str, 
    loader : dtr.PostprocessingDataLoader, 
    prediction_func : Callable[[torch.nn.Module,dtr.PostprocessingDataLoader],Poses]) -> Union[Poses,PosesWithStd]:
    '''Predictions from a single or a group of networks'''
    checkpoints = _find_models(path)
    print (f"Evaluating {path}. Found {len(checkpoints)} checkpoints.")
    preds = []
    for checkpoint in checkpoints:
        net = load_pose_network(checkpoint, device)
        preds.append(prediction_func(net, loader))
    aggregated = PosesWithStd.from_poses(preds) if len(preds)>1 else next(iter(preds))
    return aggregated, preds


def main_open_loop(paths : List[str], device):
    loader = trackertraincode.pipelines.make_validation_loader('myself')

    def process(path, crop_size_factor):
        return _track_multiple_networks(
            path, 
            device, 
            loader, 
            lambda n,l: convertlabels(predict_dataset(n,l,crop_size_factor=crop_size_factor)))

    poses_by_checkpoints = defaultdict(list)
    for crop_size_factor in [1., 1.2]:
        poses_aggregated, poses_lists = zip(*[ process(fn, crop_size_factor) for fn in paths ])
        figs = visualize(poses_aggregated, paths)
        for fig in figs:
            fig.suptitle(f'cropsize={crop_size_factor:.1f}')
        for fn, poses_list in zip(paths, poses_lists):
            poses_by_checkpoints[fn] += poses_list
    pyplot.show()

    for name in paths:
        print (f"Checkpoint: {name}")
        report_blink_stability(poses_by_checkpoints[name])



def main_closed_loop(paths : List[str], device):
    loader = trackertraincode.pipelines.make_validation_loader('myself')

    def process(path, crop_size_factor):
        return _track_multiple_networks(path, device, loader, lambda n,l: closed_loop_tracking(n,l,crop_size_factor=crop_size_factor))

    for crop_size_factor in [1., 1.2]:
        poses_aggregated, poses_lists = zip(*[ process(fn, crop_size_factor) for fn in paths ])
        figs = visualize(poses_aggregated, paths)
        for fig in figs:
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
            poses = convertlabels(predict_dataset(net, loader, crop_size_factor=1.1))
            poses.hpb[...] *= 180./np.pi
            poses_vs_model[checkpoint] = poses
        return poses_vs_model

    loader = trackertraincode.pipelines.make_validation_loader('myself_yaw')
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


def noisify(img : torch.Tensor, noiselevel : Union[torch.Tensor,float]):
    # For the case of actual batch, add image dimensions for correct broadcasting
    noiselevel = torch.as_tensor(noiselevel)[...,None,None,None]
    return (img+noiselevel*torch.randn_like(img, dtype=torch.float32)).clip(0., 255.).to(torch.uint8)


def main_analyze_noise_resist(paths : List[str]):
    rng = np.random.RandomState(seed = 12345678)
    noise_samples = 16
    data_samples = None
    noiselevels = [ 0., 2., 8., 16., 32., 48., 64. ]
    
    # noise_samples = 8
    # data_samples = 10
    # noiselevels = [ 1., 8., 16.,  ]

    def predict_noisy_dataset(net, loader, noiselevel, quantity_names, rounds = noise_samples) -> Tuple[Dict[str,torch.Tensor],Dict[str,torch.Tensor]]:
        '''Predicts given noisy inputs.

        Return:
            Dicts indexed by quantity name, returning each:
            Predictions (B x Noise x Feature) or GT (B x Feature).
        '''
        bar = tqdm.tqdm(total = len(loader.dataset))
        def predict_sample_list(samples : List[Batch]):
            rois = torch.stack([ s['roi'] for s in samples ])
            outputs = defaultdict(list) # predictions from different noised runs
            for _ in range(rounds):
                images = [ noisify(s['image'], noiselevel) for s in samples ]
                out = predict(net, images, rois, focus_roi_expansion_factor=1.1)
                for k in quantity_names:
                    outputs[k].append(out[k])
            # Stack noised runs in dimension 1
            preds_out = Batch(meta=out.meta, **{
                k:torch.stack(v,dim=1) for k,v in outputs.items()
            })
            # Collate ground truths
            gt_outputs = { k:torch.stack([s[k] for s in samples]) for k in quantity_names }
            bar.update(len(samples))
            return preds_out, gt_outputs
        preds, gt = zip(*(predict_sample_list(batch) for batch in utils.iter_batched(loader,128)))
        preds = utils.list_of_dicts_to_dict_of_lists(preds)
        for k,v in preds.items():
            preds[k] = torch.concat(v)
        gt = utils.list_of_dicts_to_dict_of_lists(gt)
        for k,v in gt.items():
            gt[k] = torch.concat(v)
        return preds, gt

    def compute_metrics_for_quats(gt : torch.Tensor, preds : torch.Tensor):
        preds = preds.swapaxes(0,1) # batch <-> noise sample (Noise,B,Features)
        mean_preds = torch.from_numpy(quat_average(preds))
        preds_spread = geodesicdistance(mean_preds[None,...].expand_as(preds), preds).square().mean(dim=0).sqrt().mean(dim=0)
        preds_error  = geodesicdistance(mean_preds, gt).mean(dim=0)
        preds_error2 = geodesicdistance(preds, gt[None,:,:]).mean()
        return preds_error, preds_spread, preds_error2

    def bbox_size(roi : torch.Tensor):
        x0,y0,x1,y1 = roi.moveaxis(-1,0)
        return torch.sqrt((x1-x0)*(y1-y0))

    def compute_metrics_for_roi(gt : torch.Tensor, preds : torch.Tensor):
        # Prediction shape is (B,Noise,len([x0,y0,x1,y1]))
        scale = bbox_size(gt)
        mean_preds = torch.mean(preds, dim=1, keepdim=True)
        preds_spread = (preds - mean_preds).square().mean(dim=1).sqrt().div(scale[:,None]).mean()
        preds_error = (mean_preds - gt).abs().div(scale[:,None]).mean()
        preds_error2 = (preds - gt[:,None,:]).abs().div(scale[:,None,None]).mean()
        return preds_error, preds_spread, preds_error2

    sample_indices = rng.choice(1900,size=data_samples) if data_samples is not None else None
    
    loader = trackertraincode.pipelines.make_validation_loader('aflw2k3d', order = sample_indices)

    rad2deg = 180./np.pi

    assert all(bool(_find_models(path)) for path in paths)

    metrics_by_network_and_noise_and_quantity = defaultdict(list)
    for path in paths:
        checkpoint = _find_models(path)
        for checkpoint in checkpoint:
            net = load_pose_network(checkpoint, 'cuda')
            for noiselevel in noiselevels:
                preds, gt  = predict_noisy_dataset(net, loader, noiselevel, ('pose','roi'))
                err_rot, spread, err_rot2 = compute_metrics_for_quats(gt['pose'], preds['pose'])
                err_roi, spread_roi, err_roi2 = compute_metrics_for_roi(gt['roi'], preds['roi'])
                metrics_by_network_and_noise_and_quantity[path, noiselevel,'pose'].append([
                    err_rot*rad2deg, 
                    spread*rad2deg,
                    err_rot2*rad2deg ])
                metrics_by_network_and_noise_and_quantity[path, noiselevel,'roi'].append([
                    err_roi, 
                    spread_roi,
                    err_roi2 ])

    prefix = os.path.commonprefix([p for p,_,_ in metrics_by_network_and_noise_and_quantity.keys()])
    metrics_by_network_and_noise_and_quantity = { (os.path.relpath(k,prefix),n,q):v for (k,n,q),v in metrics_by_network_and_noise_and_quantity.items() }

    # Mean and average over ensemble
    metrics_by_network_and_quantity = defaultdict(list)
    for (cp, noise, quantity), results in metrics_by_network_and_noise_and_quantity.items():
        metrics_by_network_and_quantity[cp,quantity].append((
            noise,
            np.average(results,axis=0),
            np.std(results,axis=0) if len(results) > 1 else np.full((2,),np.nan)
        ))

    for (checkpoint, quanitity), rows in metrics_by_network_and_quantity.items():
        print (f"Checkpoint: {checkpoint} / Quantity: {quanitity}")
        noise, avg, maybe_std = map(np.asarray,zip(*rows))
        rows = zip(noise, avg[:,0], maybe_std[:,0], avg[:,1], maybe_std[:,1])
        print (tabulate.tabulate(rows, [ 'noise', 'center', '+/-', 'spread', '+/-', 'err', '+/-'  ], tablefmt='github', floatfmt=".2f"))

    with open('/tmp/noise_resist_result.pkl','wb') as f:
        pickle.dump(metrics_by_network_and_quantity, f)

    if 1: # vis
        checkpoints = frozenset([cp for cp,_ in metrics_by_network_and_quantity.keys()])
        for quantity in ['pose','roi']:
            fig, ax = pyplot.subplots(3,1)
            for checkpoint in checkpoints:
                metrics = metrics_by_network_and_quantity[checkpoint,quantity]
                noise, avg, maybe_std = map(np.asarray,zip(*metrics))
                ycenter, spread, yerr  = avg.T
                if np.any(np.isnan(maybe_std)):
                    ax[0].plot(noise, spread, label=checkpoint)
                    ax[1].plot(noise, ycenter, label=checkpoint)
                    ax[2].plot(noise, yerr, label=checkpoint)
                else:
                    ax[0].errorbar(noise, spread, yerr=maybe_std[:,0], label=checkpoint, capsize=10.)
                    ax[1].errorbar(noise, ycenter, yerr=maybe_std[:,1], label=checkpoint, capsize=10.)
                    ax[2].errorbar(noise, yerr, yerr=maybe_std[:,2], label=checkpoint, capsize=10.)
                ax[0].legend()
                if quantity == 'pose':
                    ax[0].set(xlim=(0.,64), ylim=(0.,2.), ylabel='rot spread [deg]')
                    ax[1].set(xlim=(0.,64), ylim=(4.,6.), ylabel='rot loc')
                    ax[2].set(xlim=(0.,64), ylim=(4.,6.), xlabel='input noise std', ylabel='rot err')
                elif quantity == 'roi':
                    ax[0].set(xlim=(0.,64), ylim=(0.,0.02), ylabel='coordinate spread [%]')
                    ax[1].set(xlim=(0.,64), ylim=(0.,0.2), ylabel='coordinate loc error [%]')
                    ax[2].set(xlim=(0.,64), ylim=(0.,0.2), xlabel='input noise std', ylabel='roi coordinate error [%]')
        pyplot.show()


def main_analyze_uncertainty_error_correlation(paths : List[str], noised : bool = False):
    def noisify_batch(batch : Batch):
        if noised:
            levels = trackertraincode.neuralnets.math.random_uniform(batch.meta.prefixshape,0.,128.)
            batch['image'] = noisify(batch['image'], levels)
        else:
            levels = torch.zeros(batch.meta.prefixshape)
        batch['noise_level'] = levels
        return batch

    loader = trackertraincode.pipelines.make_validation_loader('aflw2k3d', num_workers=0)
    loader.postprocess = noisify_batch

    results_by_paths = {}

    for path in paths:
        checkpoints = _find_models(path)
        rot_err = []
        uncertainty = []
        noiselevel = []
        for checkpoint in checkpoints:
            net = load_pose_network(checkpoint, 'cuda')
            pred, gt = predict_dataset(net, loader, crop_size_factor=1.1, return_gt=True)
            rot_err += [ quat.geodesicdistance(pred['pose'],gt['pose']) ]
            uncertainty += [ torch.norm(torch.matmul(pred['pose_scales_tril'],pred['pose_scales_tril'].mT),dim=(-1,-2)) ]
            noiselevel += [ gt['noise_level'] ]
        results_by_paths[path] = (
            torch.cat(rot_err).cpu().numpy(),
            torch.cat(uncertainty).cpu().numpy(),
            torch.cat(noiselevel).cpu().numpy()
        )

    fig, ax = pyplot.subplots(1,3,dpi=120,figsize=(10,2))
    for path, (rot_err, uncertainty, noiselevel) in results_by_paths.items():
        ax[0].set_axisbelow(True)
        ax[0].grid()
        ax[0].scatter(
            rot_err*180./np.pi, 
            np.sqrt(uncertainty)*180./np.pi,
            rasterized=True, s=3., edgecolor='none', alpha=0.3)
        ax[0].set(xlabel='geo. err. °',ylabel='uncertainty °')
        ax[1].set_axisbelow(True)
        ax[1].grid()
        ax[1].scatter(
            noiselevel, 
            np.sqrt(uncertainty)*180./np.pi,
            rasterized=True, s=3., edgecolor='none', alpha=0.3)
        ax[1].set(xlabel='noise stddev.', ylabel='uncertainty °')
        ax[2].set_axisbelow(True)
        ax[2].grid()
        ax[2].scatter(
            noiselevel, 
            rot_err*180./np.pi,
            rasterized=True, s=3., edgecolor='none', alpha=0.3)
        ax[2].set(xlabel='noise stddev.', ylabel='geo. err. °')
    pyplot.tight_layout
    fig.savefig('/tmp/uncertainty_vs_err.svg')
    pyplot.show()



if __name__ == '__main__':
    np.seterr(all='raise')
    parser = argparse.ArgumentParser(description="Evaluates the model")
    parser.add_argument('filename', nargs='+', help='filenames or folders of checkpoint or onnx files', type=str)
    parser.add_argument('--closed-loop', action='store_true', default=False)
    parser.add_argument('--pitch-yaw', action='store_true', default=False)
    parser.add_argument('--open-loop', action='store_true', default=False)
    parser.add_argument('--noise-resist', action='store_true', default=False)
    parser.add_argument('--uncertainty-correlation', action='store_true', default=False)
    args = parser.parse_args()
    if not (args.closed_loop or args.pitch_yaw or args.noise_resist or args.uncertainty_correlation):
        args.open_loop = True
    if args.open_loop:
        main_open_loop(args.filename, 'cuda')
    if args.closed_loop:
        main_closed_loop(args.filename, 'cuda')
    if args.pitch_yaw:
        main_analyze_pitch_vs_yaw(args.filename)
    if args.noise_resist:
        main_analyze_noise_resist(args.filename)
    if args.uncertainty_correlation:
        main_analyze_uncertainty_error_correlation(args.filename)