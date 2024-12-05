import numpy as np
from numpy.typing import NDArray
from typing import Literal, Any, Tuple, Union, Optional

from trackertraincode.datatransformation.tensors.representation import ensure_image_nhwc
from trackertraincode.datatransformation.tensors.representation import ensure_image_nchw
from trackertraincode.neuralnets.affine2d import Affine2d

import scipy.signal.windows
import torch
from torch import Tensor
import cv2


DownFilters = Literal['gaussian','hamming','area']
UpFilters = Literal['linear','cubic','lanczos']


def _extract_size_tuple(new_size : Union[int, Tuple[int,int]]):
    try:
        new_w, new_h = new_size
    except TypeError:
        assert int(new_size), "Must be convertible to single integer"
        new_w = new_h = new_size
    return new_w, new_h


def _numpy_extract_roi(img : NDArray, roi : Tensor):
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


def _apply_antialias_filter(img : NDArray[Any], scale_factor : float, filter : str) -> NDArray[Any]:
    if filter == 'gaussian':
        ks = 0.5 / scale_factor
        return cv2.GaussianBlur(img, (0,0), ks, ks, cv2.BORDER_REPLICATE)
    elif filter == 'hamming':
        ks = 1.0 / scale_factor
        intks = max(1,round(ks*2+1))
        intks = intks if (intks&1) else (intks+1) # Make it odd
        # kern becomes a 1d array containing the hamming window
        kern = scipy.signal.windows.hamming(intks)
        kern /= np.sum(kern)
        # Pretend it was seperable. But it actually isn't. This function applies the filter
        # first along rows then along columns.
        return cv2.sepFilter2D(img, -1, kern, kern)
    else:
        raise NotImplementedError(f"Filter: {filter}")


def _resize(img, new_w, new_h, downfilter : DownFilters, upfilter : UpFilters) -> NDArray[Any]:
    old_h, old_w = img.shape if len(img.shape)==2 else img.shape[-3:-1]
    scale_factor = 0.5*(new_w/old_w + new_h/old_h)
    filter = downfilter if scale_factor<1. else upfilter
    if filter not in ('gaussian','hamming'):
        interp = {
            'linear' : cv2.INTER_LINEAR,
            'cubic' : cv2.INTER_CUBIC,
            'lanczos' : cv2.INTER_LANCZOS4,
            'area' : cv2.INTER_AREA
        }[filter]
        return cv2.resize(img, dsize=(new_w, new_h), interpolation=interp)
    else:
        return cv2.resize(_apply_antialias_filter(img, scale_factor, filter), dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR)


def affine_transform_image_cv2(img : Tensor, tr : Affine2d, new_size : Union[int, Tuple[int,int]], downfilter : Optional[DownFilters] = None, upfilter : Optional[UpFilters] = None):
    '''Anti-aliased warpAffine.
    Args:
        img: Input in hwc format
        tr: Affine transformation that can be provided to cv2.warpAffine
        new_size: Output size that can be provided to cv2.warpAffine
        downfilter: Filter when downsampling
        upfilter: Filter when upsampling
    '''
    upfilter  = 'linear' if upfilter is None else upfilter
    downfilter = 'area' if downfilter is None else downfilter
    new_w,new_h = _extract_size_tuple(new_size)
    img = ensure_image_nhwc(img)
    scale_factor = float(tr.scales.numpy())
    if scale_factor > 1.:
        upscale_warp_interp = {
            'linear' : cv2.INTER_LINEAR,
            'cubic' : cv2.INTER_CUBIC,
            'lanczos' : cv2.INTER_LANCZOS4
        }[upfilter]
        tr = tr @ Affine2d.trs(translations=torch.tensor([0.5,0.5]))
        img = cv2.warpAffine(
            img.numpy(),
            M=tr.tensor().numpy(),
            dsize=(new_w,new_h),
            flags=upscale_warp_interp,
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
        img = _resize(rotated, new_w, new_h, downfilter, upfilter)
    if img.ndim == 2: # Add back channel dimension which might have been removed by opencv
        img = img[...,None]
    return torch.from_numpy(ensure_image_nchw(img))


def croprescale_image_cv2(img : Tensor, roi : Tensor, new_size : Union[int,Tuple[int,int]], downfilter : Optional[DownFilters] = None, upfilter : Optional[UpFilters] = None):
    upfilter  = 'linear' if upfilter is None else upfilter
    downfilter = 'area' if downfilter is None else downfilter
    new_w,new_h = _extract_size_tuple(new_size)
    img = ensure_image_nhwc(img)
    img = _numpy_extract_roi(img.numpy(), roi)
    img = _resize(img, new_w, new_h, downfilter, upfilter)
    img = torch.from_numpy(img)
    if img.ndim == 2: # Add back channel dimension which might have been removed by opencv
        img = img[...,None]
    img = ensure_image_nchw(img)
    return img