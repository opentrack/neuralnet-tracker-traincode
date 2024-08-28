from os.path import join, dirname
import os
from matplotlib import projections, pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation
import random
import math
from functools import partial
import pytest

from torchvision.transforms import Compose
from torch import nn
import torch

from trackertraincode.neuralnets.modelcomponents import DeformableHeadKeypoints, rigid_transformation_25d, PosedDeformableHead, PerspectiveCorrector
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
            dtr.offset_points_by_half_pixel,
            partial(dtr.normalize_batch)
        ]))
    batch = Batch.collate([smpl for smpl in ds])
    with torch.no_grad():
        pred = headmodel(batch['coord'],batch['pose'],batch['shapeparam'])
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



def test_make_look_at_matrix():
    m = PerspectiveCorrector.make_look_at_matrix(torch.as_tensor([0.,0.,1.])).numpy()
    np.testing.assert_allclose(m, np.eye(3))
    
    SQRT3 = math.sqrt(3.)
    m = PerspectiveCorrector.make_look_at_matrix(torch.as_tensor([1.,1.,1.])).numpy()
    np.testing.assert_allclose(m[:,2], np.asarray([1./SQRT3,1./SQRT3,1./SQRT3]))
    assert np.abs(np.dot(m[:,0],np.asarray([0.,1.,0.]))) < 1.e-6
    assert m[0,0] > 0.1
    assert m[1,1] > 0.1


def fov_h(fov, aspect):
    # w/h = aspect
    # w/f = 2*tan(fov_w/2)
    # h/f = 2*tan(a)
    # aspect = tan(fov_w/2)/tan(a)
    # -> a = atan(1/aspect * tan(fov_w/2))
    return 2.*math.atan(1./aspect*math.tan(fov/2.*math.pi/180.))*180./math.pi


@pytest.mark.parametrize("fov, image_size, coord, pose, expected", [
    # Rotation matches the fov angle when position is at the edge of the screen (horizontal)
    (90.0, [200,100], [200.,50.,1.], Rotation.identity(), Rotation.from_rotvec([0.,45.,0.],degrees=True)),
    # Rotation matches the fov angle when position is at the edge of the screen (vertial)
    (90.0, [200,100], [100.,100.,1.], Rotation.identity(), Rotation.from_rotvec([-fov_h(90.,2.)/2.,0.,0.],degrees=True)),
    # Returns identity for position in the center
    (90.0, [200,100], [100.,50.,1.], Rotation.identity(), Rotation.identity()),
    # Test if original rotation is considered
    (90.0, [200,100], [100.,50.,1.], Rotation.from_rotvec([10.,20.,30.],degrees=True), Rotation.from_rotvec([10.,20.,30.],degrees=True)),
])
def test_perspective_corrector(fov, image_size, coord, pose, expected):
    corrector = PerspectiveCorrector(fov)
    result = Rotation.from_quat(corrector.corrected_rotation(
        image_sizes = torch.as_tensor(image_size,dtype=torch.long),
        coord = torch.as_tensor(coord, dtype=torch.float32),
        pose = torch.from_numpy(pose.as_quat()).to(dtype=torch.float32)
    ).numpy())
    assert Rotation.approx_equal(expected, result, atol=0.01, degrees=True), f"Converted to quats: expected = {expected.as_quat()} vs result = {result.as_quat()}"



if __name__ == '__main__':
    raise RuntimeError("Run pytest")