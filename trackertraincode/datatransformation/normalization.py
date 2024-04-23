import numpy as np
from copy import copy
from typing import Callable, Set, Sequence, Union, List, Tuple, Dict, Optional, NamedTuple

import torch
from torch import Tensor

from trackertraincode.pipelines import Batch
from trackertraincode.datasets.dshdf5pose import FieldCategory, imagelike_categories
from trackertraincode.neuralnets.affine2d import Affine2d

from trackertraincode.datatransformation.affinetrafo import (
    apply_affine2d,
    position_normalization, 
    position_unnormalization
)

from trackertraincode.datatransformation.core import (
    _ensure_image_nchw, get_category, from_numpy_or_tensor
)


def normalize_batch(sample : Batch):
    """
        Normalize coordinates to [-1.,1.]
        Normalize the color range to [-0.5,0.5]. 
    """
    def _normalize_bool(x : torch.Tensor, smooth=0.1):
        # Label smoothing!
        out = x.new_full(x.shape, smooth, dtype=torch.float32)
        out[x] = 1.0-smooth
        return out

    def _normalize_image(x: torch.Tensor):
        x = x.to(torch.float32).add(0.5).mul(1./256)
        return x

    def _compute_transform(sample : Batch):
        device = sample.device
        W, H = sample.meta.image_wh
        tr = position_normalization(W, H)
        tr = tr.to(device)
        return tr

    def _process(tr, key, value, category):
        if category == FieldCategory.image:
            return _normalize_image(v)
        elif category == FieldCategory.semseg:
            return v.to(torch.long)
        elif v.dtype == torch.bool:
            return _normalize_bool(v)
        else:
            return apply_affine2d(tr, key, value, category)

    tr = _compute_transform(sample)
    sample = copy(sample)
    for k, v in sample.items():
        sample[k] = _process(tr, k, v, get_category(sample,k))
    return sample


def unnormalize_batch(sample : Batch):
    def _compute_transform(sample):
        device = sample.device
        W, H = sample.meta.image_wh
        tr = position_unnormalization(W, H)
        tr = tr.to(device)
        return tr

    def _denormalize_image(x: torch.Tensor):
        x = torch.clamp(x.mul(256.), 0., 255.).to(torch.uint8)
        return x

    def _process(tr, key, value, category):
        if category == FieldCategory.image:
            return _denormalize_image(v)
        else:
            return apply_affine2d(tr, key, value, category)

    tr = _compute_transform(sample)
    sample = copy(sample)
    for k, v in sample.items():
        sample[k] = _process(tr, k, v, get_category(sample,k))
    return sample


def offset_points_by_half_pixel(sample : Batch):
    sample = copy(sample)
    tr_for_points = Affine2d.trs(translations=torch.tensor([0.5,0.5]))
    for k, v in sample.items():
        c = get_category(sample,k)
        if c in (FieldCategory.points, FieldCategory.xys):
            sample[k] = apply_affine2d(tr_for_points, k, v, c)
    return sample


def batch_to_torch_nchw(sample : Batch):
    sample = copy(sample)
    for k, v in sample.items():
        sample[k] = (v := from_numpy_or_tensor(v))
        if get_category(sample,k) in imagelike_categories:
            sample[k] = _ensure_image_nchw(v)
    return sample