import numpy as np
from copy import copy
from typing import Callable, Set, Sequence, Union, List, Tuple, Dict, Optional, NamedTuple
from PIL import Image
import enum

import torch
from torch import Tensor

from trackertraincode.datasets.batch import Batch
from trackertraincode.datasets.dshdf5pose import FieldCategory


def _ensure_image_nchw(img : torch.Tensor):
    assert not ((img.shape[-1] in (1,3)) and (img.shape[-3] in (1,3)))
    if img.shape[-3] in (1,3):
        return img
    else:
        return img.swapaxes(-1,-2).swapaxes(-2,-3)


def _ensure_image_nhwc(img : torch.Tensor):
    assert not ((img.shape[-1] in (1,3)) and (img.shape[-3] in (1,3)))
    if img.shape[-1] in (1,3):
        return img
    else:
        return img.swapaxes(-3,-2).swapaxes(-2,-1)


def to_numpy(batch : Batch):
    """Convert batch from tensor to numpy array"""
    batch = copy(batch)
    for k, v in batch.items():
        batch[k] = v.numpy()
    return batch


def to_tensor(batch : Batch):
    """Convert ndarrays in sample to Tensors."""
    batch = copy(batch)
    for k, v in batch.items():
        batch[k] = torch.from_numpy(v)
    return batch


def get_category(batch : Batch, k : str):
    return batch.meta.categories.get(k, FieldCategory.general)


def from_numpy_or_tensor(x : Union[np.ndarray,Tensor]):
    return x if isinstance(x, Tensor) else torch.from_numpy(x)