from typing import Tuple, Union

import torch.nn.functional as F
import torch
from torch import Tensor
from torchvision.transforms.functional import crop, resize

import kornia.filters

from trackertraincode.datatransformation.image_geometric_cv2 import _extract_size_tuple
from trackertraincode.neuralnets.affine2d import Affine2d


def _fixup_image_format_for_resample(img : Tensor):
    original_dtype = img.dtype
    assert original_dtype in (torch.uint8, torch.float32, torch.float16)
    if img.device == torch.device('cpu'):
        new_dtype = torch.float32 # Float16 and Uint8 are not supported
    else:
        new_dtype = torch.float16 if original_dtype==torch.uint8 else original_dtype
    img = img.to(new_dtype)
    def restore(x : Tensor):
        if original_dtype == torch.uint8:
            x = x.clip(0., 255.).to(dtype=torch.uint8)
        return x
    return img, restore


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


def _normalize_transform(tr : Affine2d, wh : Tuple[int,int], new_wh : Tuple[int,int]):
    '''Normalize an affine image transform so that the input/output domain is [-1,1] instead of pixel ranges.
    Image size is given as (width,height) tuples.
    '''
    w, h = wh
    new_w, new_h = new_wh
    m1= Affine2d.range_remap_2d([-1,-1], [1,1], [0, 0], [w, h])[None,...]
    m2 = Affine2d.range_remap_2d([0, 0], [new_w, new_h], [-1,-1], [1, 1])[None,...]
    return m2 @ tr @ m1


def affine_transform_image_torch(tmp : Tensor, tr : Affine2d, new_size : Union[int, Tuple[int,int]], antialias=False):
    '''Basically torch.grid_sample.

    Args: 
        tr The image transform. It is defined w.r.t. pixel ranges, not [-1,1]
    '''
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