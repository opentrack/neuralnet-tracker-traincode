import numpy as np
from typing import Callable, Set, Sequence, Union, List, Tuple, Dict, Optional, NamedTuple
from matplotlib import pyplot
import pytest
import numpy.testing
from functools import partial

import torch
import torch.nn.functional as F

import trackertraincode.datatransformation as  dtr
from trackertraincode.datatransformation.affinetrafo import (
    transform_image_torch, 
    transform_image_pil, 
    transform_image_opencv)

from trackertraincode.datasets.batch import Batch, Metadata
from trackertraincode.neuralnets.affine2d import Affine2d

from kornia.geometry.subpix import spatial_expectation2d


'''
Test consistency between scaling of an image and scaling of landmarks.

We have to think of landmarks as slightly off-center, namely to be located at the
top left of the respective pixel area. That is, assuming the landmarks happen to be
integral values.

Then reducing the image size by factor of x simply amounts to multiplying the landmark
coordinates by x.
'''


def no_randomization(B, scaling_mode) -> dtr.RoiFocusRandomizationParameters:
    return dtr.RoiFocusRandomizationParameters(
        scales = torch.tensor(1.),
        angles = torch.tensor(0.),
        translations  = torch.tensor([0.,0.]),
        scaling_mode = scaling_mode)

def with_some_similarity_trafo(B, scaling_mode) -> dtr.RoiFocusRandomizationParameters:
    return dtr.RoiFocusRandomizationParameters(
        scales = torch.tensor(1.5),
        angles = torch.tensor(20.*np.pi/180.),
        translations  = torch.tensor([-0.30,-0.05]),
        scaling_mode = scaling_mode)


class Case(NamedTuple):
    S : int # size
    R : int # new size
    X : int # point position
    Y : int # point position
    batch : Batch
    tol : List[float]


def make_batch(scale_up_or_down, aligned_corners):
    tolerances = [ 1.5, 0.1, 1.5]
    if aligned_corners:
        if scale_up_or_down == 'down':
            S = 101
            R = (S-1)//10+1
            X, Y = 30, 5
        else:
            S = 10  # S-1 segments. Every segment will have S-1 more points
            R = (S-1)*10+1
            X, Y = 3, 2
            tolerances = [ 4., 1., 4. ]
    else:
        if scale_up_or_down == 'down':
            S = 100
            R = 10
            X, Y = 30, 10
        else:
            S = 10
            R = 100
            X, Y = 3, 2

    img = torch.zeros((3,S,S), dtype=torch.float32)
    img[0,0,0] = 255.
    img[1,Y,X] = 255.
    img[2,S-1,S-1] = 255.
    points = torch.tensor([[0.,0.,0.],[X,Y,0.],[S-1,S-1,0.]], dtype=torch.float32)
    roi = torch.tensor([0.,0.,S-1,S-1])
    if not aligned_corners:
        points += 0.5
        roi = torch.tensor([0.,0.,S,S])
    batch = Batch(
        Metadata(_imagesize = S, batchsize=0, categories=
                 {'image' : dtr.FieldCategory.image, 
                  'pt3d_68' : dtr.FieldCategory.points, 
                  'roi' : dtr.FieldCategory.roi}), {
        'image' : img,
        'pt3d_68' : points,
        'roi' : roi
    })
    return Case(S, R, X, Y, batch, tolerances)


def make_batch_with_room_around_roi(scale_up_or_down, aligned_corners):
    tolerances = [ 1.5, 0.1, 1.5]
    if aligned_corners:
        if scale_up_or_down == 'down':
            pad = 50
            S = 101
            R = (S-1)//10+1
            X, Y = 30, 5
        else:
            pad = 2
            S = 10  # S-1 segments. Every segment will have S-1 more points
            R = (S-1)*10+1
            X, Y = 3, 2
            tolerances = [ 4., 1., 4. ]
    else:
        if scale_up_or_down == 'down':
            pad = 50
            S = 100
            R = 10
            X, Y = 30, 10
        else:
            pad = 2
            S = 10
            R = 100
            X, Y = 3, 2

    img = torch.zeros((3,S+2*pad,S+2*pad), dtype=torch.float32)
    img[0,pad,pad] = 255.
    img[1,Y+pad,X+pad] = 255.
    img[2,S+pad-1,S+pad-1] = 255.
    points = torch.tensor([[pad,pad,0.],[X+pad,Y+pad,0.],[S+pad-1,S+pad-1,0.]], dtype=torch.float32)
    roi = torch.tensor([pad,pad,S+pad-1,S+pad-1], dtype=torch.float32)
    if not aligned_corners:
        points += 0.5
        roi = torch.tensor([pad,pad,S+pad,S+pad], dtype=torch.float32)
    batch = Batch(
        Metadata(_imagesize = S, batchsize=0, categories=
                 {'image' : dtr.FieldCategory.image, 
                  'pt3d_68' : dtr.FieldCategory.points, 
                  'roi' : dtr.FieldCategory.roi}), {
        'image' : img,
        'pt3d_68' : points,
        'roi' : roi
    })
    return Case(S, R, X, Y, batch, tolerances)



def check(result : Batch, td : Case, align_corners : bool):
    hm = result['image'][None,...]
    assert hm.shape == (1,len(td.batch['pt3d_68']),td.R,td.R)
    hm /= hm.sum(dim=[-1,-2],keepdims=True)
    heatmap_points = spatial_expectation2d(hm, normalized_coordinates=False)[0]
    if not align_corners:
        # Because the vertices of the heatmap are centered in the middle of the pixel areas
        heatmap_points += 0.5
    coord_points = result['pt3d_68'][:,:2]
    for i, (a, b, tol) in enumerate(zip(heatmap_points, coord_points, td.tol)):
        numpy.testing.assert_allclose(a, b, atol = tol, err_msg=f'Mismatch at point {i}')


def vis(td, result, align_corners):
    fig, ax = pyplot.subplots(1,1)
    extent = [-0.5,td.R-1+0.5,td.R-1+0.5,-0.5] if align_corners else [0.,td.R,td.R,0.]
    img = ax.imshow(
        dtr._ensure_image_nhwc(result['image']).mean(dim=-1), 
        interpolation='bilinear' if align_corners else 'nearest',
        vmin=0., 
        extent=extent) 
    ax.scatter(*result['pt3d_68'].T[:2])
    fig.colorbar(img, ax=ax)
    pyplot.show()


@pytest.mark.parametrize('scaling_way', [ 'up','down'])
@pytest.mark.parametrize('sampling_method', [
    dtr.ScalingMode.TORCH_GRID_SAMPLE_NO_ALIGN_CORNERS,
    dtr.ScalingMode.TORCH_GRID_SAMPLE_ALIGN_CORNERS,
    dtr.ScalingMode.PIL_HAMMING_WINDOW,
    dtr.ScalingMode.OPENCV_AREA
])
def test_scalingtrafo(scaling_way, sampling_method):
    align_corners = sampling_method==dtr.ScalingMode.TORCH_GRID_SAMPLE_ALIGN_CORNERS
    td = make_batch(scaling_way, align_corners)
    augmentation = dtr.RandomFocusRoi(new_size = td.R)
    augmentation.make_randomization_parameters = partial(no_randomization, scaling_mode=sampling_method)
    result = augmentation(td.batch)
    #vis(td, result, align_corners)
    if sampling_method in (dtr.ScalingMode.PIL_HAMMING_WINDOW, dtr.ScalingMode.OPENCV_AREA)  and scaling_way == 'down':
        # At least one pixel tolerance, due to aliasing.
        td = td._replace(tol = [max(1.,t) for t in td.tol])
    check(result,td,align_corners)


@pytest.mark.parametrize('scaling_way', ['down', 'up'])
@pytest.mark.parametrize('sampling_method', [
    dtr.ScalingMode.TORCH_GRID_SAMPLE_NO_ALIGN_CORNERS,
    dtr.ScalingMode.TORCH_GRID_SAMPLE_ALIGN_CORNERS,
    dtr.ScalingMode.PIL_HAMMING_WINDOW,
    dtr.ScalingMode.OPENCV_AREA
])
def test_scalingtrafo_with_randomizer(scaling_way, sampling_method):
    align_corners = sampling_method==dtr.ScalingMode.TORCH_GRID_SAMPLE_ALIGN_CORNERS
    td = make_batch_with_room_around_roi(scaling_way, align_corners)
    augmentation = dtr.RandomFocusRoi(new_size = td.R)
    augmentation.make_randomization_parameters = partial(with_some_similarity_trafo, scaling_mode=sampling_method)
    result = augmentation(td.batch)
    #vis(td, result, align_corners)
    if sampling_method in (dtr.ScalingMode.PIL_HAMMING_WINDOW, dtr.ScalingMode.OPENCV_AREA)  and scaling_way == 'down':
        # At least one pixel tolerance, due to aliasing.
        td = td._replace(tol = [max(1.,t) for t in td.tol])
    check(result,td,align_corners)


@pytest.mark.parametrize('scaling_mode', [
    dtr.ScalingMode.PIL_HAMMING_WINDOW,
    dtr.ScalingMode.OPENCV_AREA,
    dtr.ScalingMode.TORCH_GRID_SAMPLE_NO_ALIGN_CORNERS,
    dtr.ScalingMode.TORCH_GRID_SAMPLE_ALIGN_CORNERS,
])
@pytest.mark.parametrize('dt',[torch.float32, torch.uint8])
def test_transform_image_only(scaling_mode, dt):
    img = 1.*torch.ones((1,8,18),dtype=torch.float32)
    img = torch.nn.functional.pad(img, [1,1,1,1], mode='constant', value=1.)
    if dt == torch.uint8:
        img = (img * 255).to(dt)

    if scaling_mode == dtr.ScalingMode.TORCH_GRID_SAMPLE_ALIGN_CORNERS:
        tr = Affine2d.range_remap_2d([15.,5.],[24.,14.], [0., 0.], [99.,99.])
    else:
        roi = torch.tensor([15, 5, 25, 15], dtype=torch.float32)
        tr = Affine2d.range_remap_2d([15,5],[25.,15.], [0., 0.], [100.,100.])
    #  x x x x o o  corner = pixel
    #      |     |
    #
    #  x x x x o o  cell-center = pixel
    #     |   |   |
    if scaling_mode in (dtr.ScalingMode.TORCH_GRID_SAMPLE_NO_ALIGN_CORNERS, dtr.ScalingMode.TORCH_GRID_SAMPLE_ALIGN_CORNERS):
        new_img = transform_image_torch(img, tr, 100, scaling_mode==dtr.ScalingMode.TORCH_GRID_SAMPLE_ALIGN_CORNERS, dtr.FieldCategory.image)
    else:
        new_img = {
            dtr.ScalingMode.PIL_HAMMING_WINDOW : transform_image_pil,
            dtr.ScalingMode.OPENCV_AREA : transform_image_opencv
        }[scaling_mode](img, roi, 100, dtr.FieldCategory.image)

    nonzeroval, zero, tol = {
        torch.float32 : (1., 0., 1.e-3),
        torch.uint8 : (255, 0, 1)
    }[dt]

    if 0:
        print (scaling_mode)
        pyplot.imshow(new_img[0])
        pyplot.show()
    assert new_img.shape==(1,100,100)
    numpy.testing.assert_allclose(new_img[:,:40,:40], np.array(nonzeroval), atol=tol)
    numpy.testing.assert_allclose(new_img[:,60:,:], np.array(zero), atol=tol)
    numpy.testing.assert_allclose(new_img[:,:,60:], np.array(zero), atol=tol)


# def test_transform_image_pil_aliasing():
#     img = 255.*torch.eye(100,dtype=torch.float32).expand(1,-1,-1)
#     roi = torch.tensor([0., 0., 99., 99.], dtype=torch.float32)

#     new_img = dtr.transform_image_opencv(img, roi, 10, dtr.FieldCategoryPoseDataset.image)

#     pyplot.imshow(new_img[0])
#     pyplot.show()


if __name__ == '__main__':
    #pytest.main(["-s","-x",__file__, "-k", "test_transform_image_only"])
    pytest.main(["-s","-x",__file__])