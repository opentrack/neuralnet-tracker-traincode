from copy import copy
from typing import Any, Callable, Set, Sequence, Union, List, Tuple, Dict, Optional, NamedTuple
from functools import wraps
from torch import Tensor
import torch

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
    IntensityAugmentationBase2D,
)


class KorniaImageDistortions(object):
    ingest_batch_list = True

    def __init__(self, *args, **kwargs):
        self.augs = AugmentationSequential(*args, **kwargs)

    def __call__(self, batches: list[Batch]):
        first = next(iter(batches))
        image_fields = {k: [] for k in first.keys() if first.get_category(k) == FieldCategory.image}
        sections = [batch.meta.batchsize for batch in batches]
        # Collate images into single tensors.
        for batch in batches:
            for k, v in image_fields.items():
                v.append(batch[k])
        # Then augment the collated images. This is much faster than running the augmentations multiple times.
        collated_augmented_images = {
            k: torch.split(self.augs(torch.cat(v)), sections) for k, v in image_fields.items()
        }
        # Split the image again and assign to copies of the original batches.
        batches = [copy(b) for b in batches]
        for k, v in collated_augmented_images.items():
            for batch, image in zip(batches, v):
                batch[k] = image
        return batches


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


class OnlyClip(IntensityAugmentationBase2D):
    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        return input.clip(0.0, 1.0)
