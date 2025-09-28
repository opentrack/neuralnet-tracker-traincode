import numpy as np
from copy import copy
from typing import Callable, Set, Sequence, Union, List, Tuple, Dict, Optional, NamedTuple

import torch
from torch import Tensor

from trackertraincode.pipelines import Batch
from trackertraincode.datasets.dshdf5pose import FieldCategory, imagelike_categories
from trackertraincode.neuralnets.affine2d import Affine2d
from trackertraincode.datatransformation.tensors.affinetrafo import (
    apply_affine2d,
    position_normalization,
    position_unnormalization,
)
from trackertraincode.datatransformation.tensors.representation import ensure_image_nchw


def whiten_image(image: torch.Tensor):
    return image.sub(0.5)


def unwhiten_image(image: torch.Tensor):
    return image.add(0.5)
