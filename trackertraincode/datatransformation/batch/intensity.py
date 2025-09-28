from copy import copy
from typing import Any, Callable, Set, Sequence, Union, List, Tuple, Dict, Optional, NamedTuple
from functools import wraps
from torch import Tensor

from trackertraincode.datasets.batch import Batch
from trackertraincode.datasets.dshdf5pose import FieldCategory

from kornia.augmentation import (
    ColorJiggle,
    ColorJitter,
    RandomBoxBlur,
    RandomPlasmaBrightness,
    RandomPlasmaContrast,
    RandomPlasmaShadow,
    RandomGaussianBlur,
    RandomSolarize,
    RandomInvert,
    RandomPosterize,
    RandomGamma,
    RandomEqualize,
    AugmentationSequential,
    RandomGaussianNoise,
    RandomContrast,
    RandomBrightness,
)


class KorniaImageDistortions(object):
    @wraps(AugmentationSequential.__init__)
    def __init__(self, *args, **kwargs):
        self.augs = AugmentationSequential(*args, **kwargs)

    def __call__(self, batch: Batch):
        batch = copy(batch)
        for k, v in batch.items():
            if batch.get_category(k) != FieldCategory.image:
                continue
            batch[k] = self.augs(v)
        return batch


class RandomGaussianNoiseWithClipping(RandomGaussianNoise):
    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        output = super().apply_transform(input, params, flags, transform)
        output = output.clip(0.0, 1.0)
        return output
