import numpy as np
from copy import copy
from typing import Callable, Set, Sequence, Union, List, Tuple, Dict, Optional, NamedTuple, Any
from PIL import Image
from numpy.typing import NDArray
import enum
import cv2

import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.transforms.functional import crop, resize
import kornia.filters

from trackertraincode.datasets.dshdf5pose import FieldCategory, imagelike_categories
from trackertraincode.neuralnets.affine2d import Affine2d
from trackertraincode.neuralnets.math import affinevecmul
from trackertraincode.datatransformation.core import _ensure_image_nchw, _ensure_image_nhwc


def position_normalization(w : int ,h : int):
    return Affine2d.range_remap_2d([0.,0.], [w,h], [-1., -1.], [1., 1.])


def position_unnormalization(w : int, h : int):
    return Affine2d.range_remap_2d([-1.,-1.], [1.,1.], [0., 0.], [w, h])


def _extract_size_tuple(new_size : Union[int, Tuple[int,int]]):
    try:
        new_w, new_h = new_size
    except TypeError:
        assert int(new_size), "Must be convertible to single integer"
        new_w = new_h = new_size
    return new_w, new_h


def _fixup_image_format_for_resample(img : Tensor):
    original_dtype = img.dtype
    assert original_dtype in (torch.uint8, torch.float32, torch.float16)
    if img.device == torch.device('cpu'):
        new_dtype = torch.float32 # Float32 and Uint8 are not supported
    else:
        new_dtype = torch.float16 if original_dtype==torch.uint8 else original_dtype
    img = img.to(new_dtype)
    def restore(x : Tensor):
        if original_dtype == torch.uint8:
            x = x.clip(0., 255.).to(dtype=torch.uint8)
        return x
    return img, restore


def _numpy_extract_roi(img : np.ndarray, roi : Tensor):
    # TODO: use OpenCV's copy border function?
    h, w, c = img.shape
    x0, y0, x1, y1 = tuple(map(int,roi))
    xmin = min(x0,0)
    ymin = min(y0,0)
    xmax = max(x1,w)
    ymax = max(y1,h)
    canvas_size = (ymax-ymin,xmax-xmin,c)
    canvas = np.zeros(canvas_size, dtype=img.dtype)
    x0 -= xmin
    x1 -= xmin
    y0 -= ymin
    y1 -= ymin
    canvas[0-ymin:h-ymin,0-xmin:w-xmin,:] = img
    img = np.ascontiguousarray(canvas[y0:y1,x0:x1,:])
    return img


def croprescale_image_cv2(img : Tensor, roi : Tensor, new_size):
    new_w,new_h = _extract_size_tuple(new_size)
    img = _ensure_image_nhwc(img)
    img = _numpy_extract_roi(img.numpy(), roi)
    interpolation = cv2.INTER_AREA if (img.shape[-2] >= new_size) else cv2.INTER_LINEAR
    img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    img = torch.from_numpy(img)
    if img.ndim == 2: # Add back channel dimension which might have been removed by opencv
        img = img[...,None]
    img = _ensure_image_nchw(img)
    return img


def affine_transform_image_cv2(img : Tensor, tr : Affine2d, new_size : Union[int, Tuple[int,int]]):
    new_w,new_h = _extract_size_tuple(new_size)
    img = _ensure_image_nhwc(img)
    scale_factor = float(tr.scales.numpy())
    if scale_factor > 1.:
        img = cv2.warpAffine(
            img.numpy(), 
            M=tr.tensor().numpy(), 
            dsize=(new_w,new_h), 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=None)
    else:
        rot_w, rot_h = round(new_w/scale_factor), round(new_h/scale_factor)
        scale_compensation = rot_h / new_h
        scaletr = Affine2d.trs(scales=torch.tensor(scale_compensation))
        rotated = cv2.warpAffine(
            img.numpy(), 
            M=(scaletr @ tr).tensor().numpy(), 
            dsize=(rot_w,rot_h), 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=None)
        img = cv2.resize(rotated, dsize=(new_w, new_h), interpolation=cv2.INTER_AREA)
    if img.ndim == 2: # Add back channel dimension which might have been removed by opencv
        img = img[...,None]
    return torch.from_numpy(_ensure_image_nchw(img))


def _normalize_transform(tr : Affine2d, wh : Tuple[int,int], new_wh : Tuple[int,int]):
    '''Normalize an affine image transform so that the input/output domain is [-1,1] instead of pixel ranges.
    Image size is given as (width,height) tuples.
    '''
    w, h = wh
    new_w, new_h = new_wh
    m1= Affine2d.range_remap_2d([-1,-1], [1,1], [0, 0], [w, h])[None,...]
    m2 = Affine2d.range_remap_2d([0, 0], [new_w, new_h], [-1,-1], [1, 1])[None,...]
    return m2 @ tr @ m1


def croprescale_image_torch(img : Tensor, roi : Tensor, new_size : Union[int, Tuple[int,int]]):
    assert roi.dtype == torch.int32
    assert img.ndim == 3
    assert roi.ndim == 1
    img, restore_dtype = _fixup_image_format_for_resample(img)
    new_w, new_h = _extract_size_tuple(new_size)
    lt = roi[:2]
    wh = roi[2:]-roi[:2]
    img = crop(img, lt[1], lt[0], wh[1], wh[0])
    img = resize(img, (new_h, new_w), antialias=True)
    assert img.shape[-2:] == (new_h, new_w), f"Torchvision resize failed. Expected shape ({new_h,new_w}). Got {img.shape[-2:]}."
    img = restore_dtype(img)
    return img


def affine_transform_image_torch(tmp : Tensor, tr : Affine2d, new_size : Union[int, Tuple[int,int]], antialias=False):
    '''Basically torch.grid_sample.

    WARNING: tr is defined w.r.t. pixel ranges, not [-1,1]
    '''
    # For regular images. TODO: semseg
    tmp, restore_dtype = _fixup_image_format_for_resample(tmp)
    new_w, new_h = _extract_size_tuple(new_size)
    C, H, W = tmp.shape
    tmp = tmp[None,...] # Add batch dim
    tr = tr[None,...]
    tr_normalized = _normalize_transform(tr, (W,H), (new_w, new_h))
    tr_normalized = tr_normalized.inv().tensor()
    if antialias:
        # A little bit of Anti-aliasing
        # WARNING: this is inefficient as the filter size gets larger
        scaling = tr.scales
        sampling_distance = 1./scaling
        ks = 0.4*sampling_distance
        intks = max(3,int(ks*3))
        intks = intks if (intks&1)==1 else (intks+1)
        tmp = kornia.filters.gaussian_blur2d(tmp, (intks,intks), (ks,ks), border_type='constant', separable=True)
    grid = F.affine_grid(
        tr_normalized.to(device=tmp.device), 
        [1, C, new_h, new_w],
        align_corners=False)
    if 0: # Debugging
        from matplotlib import pyplot
        pyplot.imshow(tmp[0,0], extent=[-1.,1.,1.,-1.])
        pyplot.scatter(grid[0,:,:,0].ravel(), grid[0,:,:,1].ravel())
        pyplot.show()
    tmp = F.grid_sample(tmp, grid, align_corners=False, mode='bilinear', padding_mode='zeros')
    tmp = tmp[0] # Remove batch dim
    tmp = restore_dtype(tmp)
    return tmp


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
    if key == 'image_backtransform': # TODO: should be metadata?
        return (Affine2d(value) @ trafo.inv()).tensor()
    return __affine2d_transform_table_by_category.get(category,lambda tr, v: v)(trafo, value)