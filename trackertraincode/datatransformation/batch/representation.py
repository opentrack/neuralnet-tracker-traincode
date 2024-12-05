import numpy as np
from copy import copy
from typing import Callable, Set, Sequence, Union, List, Tuple, Dict, Optional, NamedTuple
from PIL import Image
import enum

import torch
from torch import Tensor

from trackertraincode.datasets.batch import Batch
from trackertraincode.datasets.dshdf5pose import FieldCategory


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