import numpy as np
from copy import copy
from typing import Callable, Set, Sequence, Union, List, Tuple, Dict, Optional, NamedTuple
from PIL import Image
import enum
import cv2

import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
import kornia.filters

from trackertraincode.datasets.dshdf5pose import FieldCategory, imagelike_categories
from trackertraincode.neuralnets.affine2d import Affine2d
from trackertraincode.neuralnets.math import affinevecmul, matvecmul
from trackertraincode.datatransformation.core import _ensure_image_nchw, _ensure_image_nhwc



def position_normalization(w : int ,h : int, align_corners : bool):
    if align_corners:
        # Pixels are considered vertices of the image grid
        # Thus the maximum coordinate is w-1, or h-1 respectively.
        # E.g. a 1 pixel image is not possible as it would only be a singular point.
        #      a 2 pixel image is defined by the colors at the 4 corner points and thus has also finite area.
        w, h = w-1, h-1
    return Affine2d.range_remap_2d([0.,0.], [w,h], [-1., -1.], [1., 1.])


def position_unnormalization(w : int, h : int, align_corners : bool):
    if align_corners:
        w, h = w-1, h-1
    return Affine2d.range_remap_2d([-1.,-1.], [1.,1.], [0., 0.], [w, h])


def _to_nhw_batched(img : Tensor, view_roi : Tensor):
    shape_prefix = img.shape[:-3]
    assert shape_prefix == view_roi.shape[:-1]
    C, H, W = img.shape[-3:]
    img = img.view(-1, H, W)
    view_roi = view_roi[...,None,:].expand(*(-1 for _ in shape_prefix),C,-1).clone().view(-1,4)
    def restore_image_format(new_img):
        return new_img.view(*shape_prefix,C,new_img.shape[-2],new_img.shape[-1])
    return img, view_roi, restore_image_format


def _to_nhwc_batched(imgs : Tensor, view_roi : Tensor):
    imgs = _ensure_image_nhwc(imgs)
    shape_prefix = imgs.shape[:-3]
    imgs = imgs.view(-1,*imgs.shape[-3:])
    view_roi = view_roi.view(-1,4)
    def restore_image_format(new_img):
        return _ensure_image_nchw(new_img.view(*shape_prefix,*new_img.shape[-3:]))
    return imgs, view_roi, restore_image_format


def _pil_extract_roi(img : Image.Image, roi : Tuple[float]):
    w, h = img.width, img.height
    x0, y0, x1, y1 = tuple(map(int,roi))
    xmin = min(x0,0)
    ymin = min(y0,0)
    xmax = max(x1,w)
    ymax = max(y1,h)
    canvas_size = (xmax-xmin,ymax-ymin)
    canvas = Image.new(img.mode, canvas_size)
    x0 -= xmin
    x1 -= xmin
    y0 -= ymin
    y1 -= ymin
    canvas.paste(img, (0-xmin,0-ymin,w-xmin,h-ymin))
    img = canvas.crop((x0,y0,x1,y1))
    return img


def _numpy_extract_roi(img : np.ndarray, roi : Tuple[float]):
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


def transform_image_opencv(imgs : Tensor, view_roi : Tensor, new_size : int, category):
    assert category == FieldCategory.image, "Not Implemented"
    imgs, view_roi, restore_image_format = _to_nhwc_batched(imgs, view_roi)
    roi_for_pil = [ tuple(float(x) for x in roi) for roi in view_roi]
    def _do_crop_and_resample(img, roi):
        img = _numpy_extract_roi(img.numpy(), roi)
        interpolation = cv2.INTER_AREA if (img.shape[-2] >= new_size) else cv2.INTER_LINEAR
        img = cv2.resize(img, (new_size, new_size), interpolation=interpolation)
        return torch.from_numpy(img)
    imgs = [ _do_crop_and_resample(img,roi) for img, roi in zip(imgs,roi_for_pil) ]
    imgs = torch.stack(imgs,axis=0)
    imgs = restore_image_format(imgs)
    return imgs


def transform_image_pil(imgs : Tensor, view_roi : Tensor, new_size : int, category):
    assert category == FieldCategory.image, "Not Implemented"
    imgs, view_roi, restore_image_format = _to_nhw_batched(imgs, view_roi)
    roi_for_pil = [ tuple(float(x) for x in roi) for roi in view_roi]
    def _do_crop_and_resample(img, roi):
        img = to_pil_image(img, mode='F' if img.dtype==torch.float32 else 'L')
        img = _pil_extract_roi(img, roi)
        img = img.resize((new_size,new_size),resample=Image.HAMMING, reducing_gap=3.)
        img = pil_to_tensor(img)
        return img
    imgs = [ _do_crop_and_resample(img,roi) for img, roi in zip(imgs,roi_for_pil) ]
    imgs = torch.stack(imgs, dim=0)
    imgs = restore_image_format(imgs)
    return imgs


def transform_image_torch(tmp : Tensor, tr : Affine2d, new_size : int, align_corners : bool, category):
    assert category == FieldCategory.image, "Not Implemented"
    # For regular images. TODO: semseg
    original_dtype = tmp.dtype
    assert original_dtype in (torch.uint8, torch.float32)
    tmp = tmp.to(torch.float32)
    shape_prefix = tmp.shape[:-3]
    if tmp.ndim==3:
        # Add batch dim
        tr = tr[None,...]
        tmp = tmp[None,...]
    C, H, W = tmp.shape[-3:]
    tmp = tmp.view(-1, C, H, W)
    B = tmp.size(0)
    try:
        new_w, new_h = new_size
    except TypeError:
        assert int(new_size), "Must be convertible to single integer"
        new_w = new_h = new_size
    if align_corners:
        m1= Affine2d.range_remap_2d([-1,-1], [1,1], [0, 0], [W-1, H-1])[None,...]
        m2 = Affine2d.range_remap_2d([0, 0], [new_w-1, new_h-1], [-1,-1], [1, 1])[None,...]
    else:
        m1= Affine2d.range_remap_2d([-1,-1], [1,1], [0, 0], [W, H])[None,...]
        m2 = Affine2d.range_remap_2d([0, 0], [new_w, new_h], [-1,-1], [1, 1])[None,...]
    tr_normalized = m2 @ tr @ m1
    tr_normalized = tr_normalized.inv().tensor()
    if 1: # Anti-aliasing
        scaling = tr.scales
        sampling_distance = 1./scaling
        #if sampling_distance > 2.:
        #    tmp = trackertraincode.fftblur.fftblur(tmp, 0.5*sampling_distance)
        #elif sampling_distance > 1.:
        if 1:
            ks = 0.5*sampling_distance
            intks = max(3,int(ks*10))
            intks = intks if (intks&1)==1 else (intks+1)
            tmp = kornia.filters.gaussian_blur2d(tmp, (intks,intks), (ks,ks), border_type='constant', separable=True)
    grid = F.affine_grid(
        tr_normalized, 
        [B, C, new_h, new_w],
        align_corners=align_corners)
    if 0: # Debugging
        from matplotlib import pyplot
        pyplot.imshow(tmp[0,0], extent=[-1.,1.,1.,-1.])
        pyplot.scatter(grid[0,:,:,0].ravel(), grid[0,:,:,1].ravel())
        pyplot.show()
    tmp = F.grid_sample(tmp, grid, align_corners=align_corners, mode='bilinear', padding_mode='zeros') # if align_corners else 'border')
    tmp = tmp.view(*shape_prefix, C, new_h, new_w)
    if original_dtype != torch.float32:
        tmp = tmp.clip(0., 255.).to(dtype=torch.uint8)
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