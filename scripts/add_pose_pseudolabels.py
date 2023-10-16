import h5py
from os.path import join, dirname, isfile
import numpy as np
import tqdm
import argparse
import copy
from typing import NamedTuple, Optional, List, Tuple
from collections import defaultdict
import torch
from torch import Tensor
from matplotlib import pyplot
import gc

from trackertraincode.datasets.dshdf5pose import Hdf5PoseDataset
from trackertraincode.neuralnets.torchquaternion import quat_average
import trackertraincode.vis as vis
import trackertraincode.datatransformation as dtr
import trackertraincode.utils as utils
from torchvision.transforms import Compose
from trackertraincode.datasets.batch import Batch
from torch.utils.data import Subset

from trackertraincode.eval import load_pose_network, predict, InferenceNetwork

from trackertraincode.datasets.dshdf5pose import create_pose_dataset, FieldCategory

C = FieldCategory


def setup_loader(args):
    ds = Hdf5PoseDataset(
        args.filename, 
        transform=Compose([
            # dtr.batch_to_torch_nchw,
            #dtr.to_tensor,
            dtr.offset_points_by_half_pixel, # For when pixels are considered grid cell centers
        ]),
        monochrome=True)
    if args.dryrun:
        ds = Subset(ds, np.arange(10))
    N = len(ds)
    loader = dtr.PostprocessingDataLoader(ds, args.batchsize,
        shuffle=False,
        num_workers=0, #utils.num_workers(),
        postprocess=None,
        collate_fn= lambda samples : samples,
        unroll_list_of_batches = False
    )
    return loader, ds


def fit_batch(net : InferenceNetwork, batch : List[Batch]):
    images = [ s['image'] for s in batch ]
    rois = torch.stack([ s['roi'] for s in batch ])
    indices = torch.stack([ s['index'] for s in batch ])
    out = predict(net, images, rois, focus_roi_expansion_factor=1.2)
    out = {
        k:out[k] for k in 'unnormalized_quat coord pt3d_68 shapeparam'.split()
    }
    out.update(index = indices)
    return out


def test_quats_average():
    def positivereal(q):
        s = np.sign(q[...,3])
        return q*s[...,None]
    from scipy.spatial.transform import Rotation
    expected_quats = Rotation.random(10).as_quat()
    quats = Rotation.from_quat(np.repeat(expected_quats, 10, axis=0))
    offsets = Rotation.random(10*10).as_rotvec()*0.01
    quats = quats * Rotation.from_rotvec(offsets)
    quats = quats.as_quat().reshape((10,10,4)).transpose(1,0,2)
    out = quat_average(quats)
    #print (positivereal(out) - positivereal(expected_quats))
    assert np.allclose(positivereal(out) , positivereal(expected_quats), atol=0.02)


@torch.no_grad()
def fitall(args):
    assert all(isfile(f) for f in args.checkpoints)
    print ("Inferring from networks:", args.checkpoints)

    with h5py.File(args.filename, 'r+') as f:
        g = f.require_group(args.hdfgroupname) if args.hdfgroupname else f
        for key in 'coords quats pt3d_68 shapeparams':
            try:
                del g[key]
            except KeyError:
                pass

    loader, ds = setup_loader(args)
    num_samples = len(ds)

    outputs_per_net = defaultdict(list)
    for modelfile in tqdm.tqdm(args.checkpoints, desc="Network"):
        net = load_pose_network(modelfile, 'cuda')
        outputs = [ fit_batch(net, batch) for batch in tqdm.tqdm(loader, "Batch") ]
        outputs = utils.list_of_dicts_to_dict_of_lists(outputs)
        outputs = {k:np.concatenate(v,axis=0) for k,v in outputs.items() }
        ordering = np.argsort(outputs.pop('index'))
        outputs = { k:v[ordering] for k,v in outputs.items() }
        for k,v in outputs.items():
            outputs_per_net[k].append(v)
        del outputs
    outputs_per_net = {
        k:np.stack(v) for k,v in outputs_per_net.items()
    }

    del loader
    del ds
    gc.collect() # Ensure the hdf5 file in the data was really closed.
    # There is no way to enforce it. We can only hope the garbage
    # collector will destroy the objects. If there is still a reference
    # left it will be read-only and lead to failure when trying to write
    # below.

    # FIXME: final quats output is busted. Values are more or less garbage.
    #        unnormalized_quat looks fine!

    quats = quat_average(outputs_per_net.pop('unnormalized_quat'))
    coords = np.average(outputs_per_net.pop('coord'), axis=0)
    pt3d_68 = np.average(outputs_per_net.pop('pt3d_68'), axis=0)
    shapeparams = np.average(outputs_per_net.pop('shapeparam'), axis=0)

    assert len(quats) == num_samples

    with h5py.File(args.filename, 'r+') as f:
        g = f.require_group(args.hdfgroupname) if args.hdfgroupname else f
        ds_quats = create_pose_dataset(g, C.quat, count=num_samples, data=quats)
        ds_coords = create_pose_dataset(g, C.xys, count=num_samples, data=coords)
        ds_pt3d_68 = create_pose_dataset(g, C.points, name='pt3d_68', count=num_samples, shape_wo_batch_dim=(68,3), data=pt3d_68)
        ds_shapeparams = create_pose_dataset(g,C.general, name='shapeparams', count=num_samples, shape_wo_batch_dim=(50,), data = shapeparams)


if __name__== '__main__':
    test_quats_average()

    defaultcheckpoint = join(dirname(__file__),'..','model_files','best_rot_NetworkWithPointHead.pt')
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='the dataset to label')
    parser.add_argument('-c','--checkpoints', help='model checkpoint', nargs='*', type=str, default=defaultcheckpoint)
    parser.add_argument('-b','--batchsize', help="The batch size", type=int, default=512)
    parser.add_argument('--hdf-group-name', help="Group to store the annotations in", type=str, default='', dest='hdfgroupname')
    parser.add_argument('--dryrun', default=False, action='store_true')
    #parser.add_argument('-d','--device', type=str, default='cuda')
    args = parser.parse_args()
    args.device = 'cuda'
    fitall(args)