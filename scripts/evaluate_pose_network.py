#!/usr/bin/env python
# coding: utf-8

from os.path import join, dirname
import numpy as np
import cv2
cv2.setNumThreads(1)
import os
import argparse
import h5py
from scipy.spatial.transform import Rotation

from matplotlib import pyplot

import datasets.datasets
import vis
import utils
import train

import torch


DEFAULT_FILENAME = join(dirname(__file__),'..','model_files','best_rot_NetworkWithPointHead.ckpt')

def load_network(args):
    filename = args.filename
    if filename.endswith('.onnx'):
        import onnxruntime

        # Construct a wrapper right here
        class OnnxPoseNetwork(object):
            input_resolution = 129
            
            def __init__(self, modelfile):
                self.session = onnxruntime.InferenceSession(modelfile)
            
            @property
            def training(self):
                return False
            
            def _inference_iter(self, batch):
                for img in batch.numpy():
                    yield self.session.run(None, {
                        'x': img[None,...]
                    })
                
            def __call__(self, batch):
                coords, quats, boxes = zip(*self._inference_iter(batch))
                return {
                    'coord' : torch.from_numpy(np.vstack(coords)),
                    'pose'  : torch.from_numpy(np.vstack(quats)),
                    'roi'   : torch.from_numpy(np.vstack(boxes))
                }
        return OnnxPoseNetwork(filename)
    else:
        import neuralnets.models
        import neuralnets.modelcomponents

        # The checkpoints are not self-describing. So we must have a matching network in code.
        #net = neuralnets.models.FinetuningNetworkWrapper(neuralnets.models.NetworkWithPointHead(enable_point_head=False, enable_full_head_box=True, enable_face_detector=True))
        #net = neuralnets.models.NetworkWithPointHead(enable_point_head=False, enable_full_head_box=True, enable_face_detector=True)
        #net = neuralnets.models.LocalAttentionNetwork(enable_face_detector=True, enable_full_head_box=True)
        net = neuralnets.models.FinetuningNetworkWrapper(neuralnets.models.LocalAttentionNetwork(enable_face_detector=True, enable_full_head_box=True))
        state_dict = torch.load(filename)
        neuralnets.modelcomponents.clear_denormals_inplace(state_dict)
        net.load_state_dict(state_dict)
        net.eval()
        return net


def create_dataloaders(args, inputsize):
    loaders = datasets.datasets.make_validation_loaders(
        'aflw' in args.dataset,
        'vggface' in args.dataset,
        inputsize = inputsize,
    )
    return loaders


def iterate_predictions(loader, net):
    for batch in loader:
        with torch.no_grad():
            preds = net(batch['image'])
        for sample, pred in zip(utils.undo_collate(batch), utils.undo_collate(preds)):
                yield vis.unnormalize_sample_to_numpy(sample, pred)


def display_worst_list(items):
    def iterate():
        for value, sample, pred in items:
            yield vis.unnormalize_sample_to_numpy(sample, pred)
    return vis.matplotlib_plot_iterable(iterate(), vis.draw_prediction)


def report(net, loader):
    poseerrs, eulererrs, enable = train.metrics_over_full_dataset(
        net, 
        [ train.PoseErr(), train.EulerAngleErrors(), train.PoseEnableWeights() ], 
        loader)
    enable = (enable>0.).astype(np.float32)
    print (f"   Euler MAE (p,y,r): {np.average(np.abs(eulererrs), weights=enable, axis=0)*utils.rad2deg}")
    print (f"   Euler MAE (p+y+r)/3: {np.average(np.average(np.abs(eulererrs), weights=enable, axis=0))*utils.rad2deg}")
    e_rot, e_posx, e_posy, e_size = np.array(poseerrs).T
    rmse_pos = np.sqrt(np.average(np.sum(np.square(np.vstack([e_posx, e_posy]).T), axis=1), weights=enable, axis=0))
    rmse_size = np.sqrt(np.average(np.square(e_size), weights=enable))
    print (f"   Average angular error: {np.average(e_rot)*180/np.pi:.03f}°")
    print (f"   Position RMSE: {rmse_pos*100:.03f}%")
    print (f"   Size RMSE: {rmse_size*100:.03f}%")
    

def rot_err(preds, batch):
    errs = train.PoseErr()(preds, batch)
    return errs[:,0] # Rotation


def pose_err(preds, batch):
    errs = train.PoseErr()(preds, batch)
    return np.amax(errs[:,1:], axis=1)


def run(args):
    net = load_network(args)
    gui = []
    loaders = create_dataloaders(args, net.input_resolution)
    for name, loader in loaders.items():
        print (f"--- On {name} ---")
        report(net, loader)
        if args.vis:
            fig, btn = vis.matplotlib_plot_iterable(iterate_predictions(loader, net), vis.draw_prediction)
            fig.suptitle(name)
            gui.append((fig,btn))
            worst = train.k_worst_over_dataset(net, loader, rot_err, 9)
            fig, btn = display_worst_list(worst)
            fig.suptitle(name+' worst rot')
            gui.append((fig,btn))
            worst = train.k_worst_over_dataset(net, loader, pose_err, 9)
            fig, btn = display_worst_list(worst)
            fig.suptitle(name+' worst pose')
            gui.append((fig,btn))

    if 'aflw2k3d' in loaders.keys():
        print ('''
Previously with ClearDusk's MobileNet Implementation:
Trained on 300WLP + YTFaces from Kaggle. Commit 1629ad.
--- On aflw2k3d ---
   Euler MAE (p,y,r): [5.22213843 3.37474653 3.56001 ]
   Euler MAE (p+y+r)/3: 4.052298319124914
   Average angular error: 6.247°
   Position RMSE: 4.342%
   Size RMSE: 5.805%
--- On aflw2k3d_grimaces ---
   Euler MAE (p,y,r): [5.52069662 4.15235869 3.82043717]
   Euler MAE (p+y+r)/3: 4.49783082554956
   Average angular error: 7.285°
   Position RMSE: 4.662%
   Size RMSE: 8.757%
Note: ResNet50-based model is a few tenths of degrees worse due to overfitting apparently.
''')
    if 'vggface2' in loaders.keys():
        print ('''
--- BASELINE MobilnetV1 on vggface2 ---
   Euler MAE (p,y,r): [1.67117337 1.56536549 1.21603189]
   Euler MAE (p+y+r)/3: 1.48419024685242
   Average angular error: 2.750°
   Position RMSE: 5.232%
   Size RMSE: 2.413%
        ''')

    pyplot.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate pose networks")
    parser.add_argument('filename', help='filename of checkpoint or onnx model file', nargs='?', type=str, default=DEFAULT_FILENAME)
    parser.add_argument('--no-vis', dest='vis', help='disable visualization', default=True, action='store_false')
    parser.add_argument('--ds', dest='dataset', help='the dataset [aflw,vggface]', default='aflw', type=str)
    args = parser.parse_args()
    run(args)