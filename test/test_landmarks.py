import copy
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
from scipy.spatial.transform import Rotation

from trackertraincode.neuralnets.modelcomponents import DeformableHeadKeypoints, PosedDeformableHead
from trackertraincode.neuralnets.rotrepr import QuatRepr
from trackertraincode.datasets.batch import Batch
import trackertraincode.datatransformation as dtr
from trackertraincode.datasets.dshdf5pose import Hdf5PoseDataset
import trackertraincode.vis as vis
from trackertraincode.facemodel.bfm import BFMModel, ScaledBfmModule

# TODO: rename to test_modelcomponents.py


def test_landmarks():
    headmodel = PosedDeformableHead(DeformableHeadKeypoints())

    ds = Hdf5PoseDataset(
        join(dirname(__file__), "..", "aflw2kmini.h5"),
        transform=Compose([dtr.batch.offset_points_by_half_pixel, dtr.batch.normalize_batch]),
    )
    batch = Batch.collate([smpl for smpl in ds])
    with torch.no_grad():
        pred = headmodel(batch["coord"], QuatRepr(batch["pose"]), batch["shapeparam"])
    target = batch["pt3d_68"]
    diff = torch.mean(torch.norm(pred - target, dim=-1), axis=-1)

    worst_to_best = np.flip(np.argsort(diff.numpy()))

    if 0:
        for i in worst_to_best:
            fig, axes = pyplot.subplots(1, 2)
            axes[0].scatter(pred[i, :, 0], -pred[i, :, 1], c="r")
            axes[0].scatter(target[i, :, 0], -target[i, :, 1], c="g")
            axes[1].scatter(pred[i, :, 0], -pred[i, :, 2], c="r")
            axes[1].scatter(target[i, :, 0], -target[i, :, 2], c="g")
            for j, p in enumerate(pred[i]):
                axes[0].text(p[0], -p[1], s=str(j), size=9, color="r")
            for j, p in enumerate(target[i]):
                axes[0].text(p[0], -p[1], s=str(j), size=9, color="g")
            pyplot.show()

    assert torch.max(diff) < 0.01, f"Landmark reconstruction error too large: {torch.max(diff)}"


def _roi(x):
    min_ = torch.amin(x[..., :2], dim=-2)
    max_ = torch.amax(x[..., :2], dim=-2)
    return torch.cat([min_, max_], dim=0)


@torch.no_grad()
def test_subsampled_face():
    rng = np.random.default_rng(seed=1337)
    coord = torch.as_tensor([0.0, 0.0, 1.0])
    shapeparam = torch.zeros((50,))
    quat = QuatRepr(
        torch.from_numpy(Rotation.random(num=100, random_state=rng).as_quat()).to(torch.float32)
    )

    rng = np.random.default_rng(seed=29387)
    headmodel = PosedDeformableHead(ScaledBfmModule(BFMModel()))
    downsampled = PosedDeformableHead(headmodel.deformable_head.downsampled(2048, rng))

    assert headmodel.deformable_head.vertices.size(0) > 10000
    assert downsampled.deformable_head.vertices.size(0) == 2048
    assert downsampled.deformable_head is not headmodel.deformable_head

    reference_points = headmodel(coord, quat, shapeparam)
    downsampled_points = downsampled(coord, quat, shapeparam)

    headsize = reference_points.abs().max().item()
    assert 0.9 <= headsize <= 1.2

    reference_rois = _roi(reference_points)
    downsampled_rois = _roi(downsampled_points)
    np.testing.assert_allclose(downsampled_rois, reference_rois, atol=0.035)


if __name__ == "__main__":
    raise RuntimeError("Run pytest")
