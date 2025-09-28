import numpy as np
from copy import copy
from typing import Callable, Set, Sequence, Union, List, Tuple, Dict, Optional, NamedTuple
from PIL import Image
import enum

import torch
from torch import Tensor

from trackertraincode.datasets.batch import Batch
from trackertraincode.datasets.dshdf5pose import FieldCategory


def ensure_image_nchw(img: torch.Tensor):
    assert not ((img.shape[-1] in (1, 3)) and (img.shape[-3] in (1, 3)))
    if img.shape[-3] in (1, 3):
        return img
    else:
        return img.swapaxes(-1, -2).swapaxes(-2, -3)


def ensure_image_nhwc(img: torch.Tensor):
    assert not ((img.shape[-1] in (1, 3)) and (img.shape[-3] in (1, 3)))
    if img.shape[-1] in (1, 3):
        return img
    else:
        return img.swapaxes(-3, -2).swapaxes(-2, -1)
