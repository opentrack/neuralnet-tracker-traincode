import numpy as np
from typing import Callable, Set, Sequence, Union, List, Optional, NamedTuple, Literal

import torch

from trackertraincode.datasets.dshdf5pose import FieldCategory, imagelike_categories
from trackertraincode.neuralnets.affine2d import Affine2d
from trackertraincode.neuralnets.math import affinevecmul


def position_normalization(w : int ,h : int):
    return Affine2d.range_remap_2d([0.,0.], [w,h], [-1., -1.], [1., 1.])


def position_unnormalization(w : int, h : int):
    return Affine2d.range_remap_2d([-1.,-1.], [1.,1.], [0., 0.], [w, h])


def handle_backtransform_insertion(sample : dict, W : int, H : int, tr : Affine2d, type : str = 'tensor'):
    assert type in ('tensor','ndarray')
    if (prev_tr := sample.get('image_backtransform', None)) is not None:
        pass
    else:
        new_tr = sample['image_backtransform'] = tr.inv().tensor()
        if type == 'ndarray':
            new_tr = new_tr.numpy()
        sample['image_backtransform'] = new_tr
    if 'image_original_size' not in sample:
        img_size = torch.tensor((W, H), dtype=torch.int32)
        if type == 'ndarray':
            img_size = img_size.numpy()
        sample['image_original_size'] = img_size


def transform_points(tr : Affine2d, points : torch.Tensor):
    assert points.size(-1) in (2,3), f"Bad point array shape: {points.shape}"
    m = tr.tensor()
    batch_dimensions = m.shape[:-2]
    # First create a new shape for the transform so that it can be broadcasted.
    # The problem is that I want to support point shapes like B x N x F, where the transform may come as B x 2 x 3.
    # So I have to inject the N in the middle, or actually a 1 and let broadcasting handle the rest
    assert points.shape[:len(batch_dimensions)] == batch_dimensions
    new_shape =batch_dimensions + tuple(1 for _ in range(len(points.shape)-len(batch_dimensions)-1)) + (2,3)
    m = m.view(*new_shape)
    if points.size(-1) == 2:
        return affinevecmul(m, points)
    else:
        # Transform x,y. Scale z like x and y. Don't invert z in case of reflections.
        out = torch.empty_like(points)
        out[...,:2] = affinevecmul(m, points[...,:2])
        out[...,2] = torch.sqrt(torch.abs(tr.det))[...,None] * points[...,2]
        return out


def transform_keypoints(tr : Affine2d, points : torch.Tensor):
    from trackertraincode.facemodel.keypoints68 import flip_map
    out = transform_points(tr, points)
    det = tr.det
    if points.shape[-1]==3 and torch.any(det < 0.):
        mask = det<0.
        flipped = out[mask,...][:,flip_map,:].clone(memory_format=torch.contiguous_format)
        out[det<0.] = flipped
    elif points.shape[-1]==2 and det<0.:
        out = out[flip_map,:].clone(memory_format=torch.contiguous_format)
    return out


def transform_roi(tr : Affine2d, roi : torch.Tensor):
    x0, y0, x1, y1 = roi.moveaxis(-1,0)
    pointvec = roi.new_empty(roi.shape[:-1]+ (4, 2))
    points = [
        (x0,y0),
        (x0,y1),
        (x1,y0),
        (x1,y1)
    ]
    for i, (x,y) in enumerate(points):
        pointvec[...,i,0] = x
        pointvec[...,i,1] = y
    pointvec = transform_points(tr, pointvec)
    out = roi.new_empty(roi.shape[:-1]+(4,))
    out[...,:2] = torch.amin(pointvec, dim=-2)
    out[...,2:] = torch.amax(pointvec, dim=-2)
    return out


def transform_coord(tr : Affine2d, coord : torch.Tensor):
    out_coord = torch.empty_like(coord)
    # Position
    out_coord[...,:2] = affinevecmul(tr.tensor(), coord[...,:2])
    # Size
    out_coord[...,2] = tr.scales*coord[...,2]
    return out_coord


def transform_rot(tr : Affine2d, quat : torch.Tensor):
    from trackertraincode.neuralnets.torchquaternion import mult
    m = tr.tensor()
    # Use the "y"-vector to determine the rotation angle because we want
    # zero rotation when the transform constitutes horizontal flipping.
    sn = -m[...,0,1]
    cs =  m[...,1,1]
    # Recover cos(t/2) + uz k sin(t/2)?
    # Use sign(det) to handle horizontal reflections. If reflected the rotation
    # angle must be reversed.
    detsign = torch.sign(tr.det)
    alpha = torch.atan2(sn,cs)
    qw = torch.cos(alpha*0.5)
    qk = torch.sin(alpha*0.5)*detsign
    qi = qj = torch.zeros_like(qw)
    # Premultiply quat
    zrot = torch.stack([qi,qj,qk,qw],dim=-1)
    zrot = zrot.expand_as(quat)

    out = mult(zrot, quat)
    # Another manipulation due to potential mirroring. A rotation matrix R transforms like
    # R' = T R T^-1, where T is the matrix containing the reflection of one axis.
    # It is easy to see that this negates the off-diagonals pertaining to row and columns
    # of the flipped axis. Further it is easy to see using the quaternion derived rotation
    # matrix that the reflection amounts to negating the imaginary components which pertain
    # to the non-reflected axes.
    out[...,1] = detsign*out[...,1]
    out[...,2] = detsign*out[...,2]
    return out


__affine2d_transform_table_by_category = {
    FieldCategory.xys : transform_coord,
    FieldCategory.quat : transform_rot,
    FieldCategory.roi : transform_roi,
    FieldCategory.points : transform_keypoints,
}


def apply_affine2d(trafo : Affine2d, key : str, value : torch.Tensor, category : FieldCategory):
    assert category not in imagelike_categories
    if key == 'image_backtransform':
        # Backtransform applied to local points P' : BT @ P' = P, yields points in original image
        # When applying the trafo argument to points (to apply augmentation)
        # Q = trafo @ P'
        # What is the trafo to map from Q back to the original?
        # P' = trafo^-1 @ Q
        # P = (BT @ trafo^-1) @ Q
        return (Affine2d(value) @ trafo.inv()).tensor()
    return __affine2d_transform_table_by_category.get(category,lambda tr, v: v)(trafo, value)