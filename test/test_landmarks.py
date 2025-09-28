from os.path import join, dirname
import os
from matplotlib import projections, pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
from functools import partial

from torchvision.transforms import Compose
from torch import nn
import torch

from trackertraincode.neuralnets.modelcomponents import DeformableHeadKeypoints, PosedDeformableHead
from trackertraincode.neuralnets.rotrepr import QuatRepr
from trackertraincode.datasets.batch import Batch
import trackertraincode.datatransformation as dtr
from trackertraincode.datasets.dshdf5pose import Hdf5PoseDataset
import trackertraincode.vis as vis

# TODO: rename to test_modelcomponents.py


def test_landmarks():
    headmodel = PosedDeformableHead(DeformableHeadKeypoints())

    ds = Hdf5PoseDataset(
        join(dirname(__file__),'..','aflw2kmini.h5'),
        transform = Compose([
            dtr.batch.offset_points_by_half_pixel,
            dtr.batch.normalize_batch
        ]))
    batch = Batch.collate([smpl for smpl in ds])
    with torch.no_grad():
        pred = headmodel(batch['coord'],QuatRepr(batch['pose']),batch['shapeparam'])
    target = batch['pt3d_68']
    diff = torch.mean(torch.norm(pred-target, dim=-1),axis=-1)

    worst_to_best = np.flip(np.argsort(diff.numpy()))

    if 0:
        for i in worst_to_best:
            fig, axes = pyplot.subplots(1,2)
            axes[0].scatter(pred[i,:,0],-pred[i,:,1], c='r')
            axes[0].scatter(target[i,:,0],-target[i,:,1], c='g')
            axes[1].scatter(pred[i,:,0],-pred[i,:,2], c='r')
            axes[1].scatter(target[i,:,0],-target[i,:,2], c='g')
            for j, p in enumerate(pred[i]):
                axes[0].text(p[0], -p[1], s=str(j), size=9, color='r')
            for j, p in enumerate(target[i]):
                axes[0].text(p[0], -p[1], s=str(j), size=9, color='g')
            pyplot.show()

    assert torch.max(diff) < 0.01, f"Landmark reconstruction error too large: {torch.max(diff)}"



if __name__ == '__main__':
    raise RuntimeError("Run pytest")