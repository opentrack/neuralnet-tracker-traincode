from torch.utils.data import Sampler, ConcatDataset
from torch.quasirandom import SobolEngine
import torch

from abc import ABC,abstractmethod
from typing import List, Dict, Iterator, Optional, Tuple, Union, Sequence, Any, Callable, Generic, TypeVar
import math
import numpy as np
import sys
import trackertraincode.utils as utils
import os
import copy

from torch.utils.data import ConcatDataset, BatchSampler, RandomSampler, Sampler

T_co = TypeVar('T_co', covariant=True)


BatchSamper = Sampler[List[int]]
ItemSampler = Sampler[int]
ChoicesSampler = Callable[[],int]
ChoicesSamplerFactory = Callable[[Sequence[float],Optional[int]], ChoicesSampler]


def weights_normalized(w):
     w = np.asarray(w)
     assert w.ndim == 1
     wsum = np.sum(w)
     assert wsum > 0.
     return  w / wsum 


class SobolChoices(object):
    def __init__(self, weights, seed=None):
        probs = torch.from_numpy(weights_normalized(weights))
        self.accum = torch.cumsum(probs, dim=0)
        assert torch.abs(self.accum[-1]-1.) < 1.e-6
        self.qrng = SobolEngine(1, scramble=True, seed=seed)

    def __call__(self):
        i = torch.searchsorted(self.accum, self.qrng.draw())
        i = torch.clamp(i, 0, len(self.accum)-1)
        return i


class PseudoRandomChoices(object):
    def __init__(self, weights, seed=None):
        probs = weights_normalized(weights)
        self.probs = np.asarray(probs)
        self.n = len(probs)
        self.rng = np.random.RandomState(seed=seed)

    def __call__(self):
        return self.rng.choice(self.n, p=self.probs)

# TODO: Doc
class ConcatDatasetSampler(Sampler[T_co]):
    def __init__(self,
        dataset : ConcatDataset,
        wrapped : Sequence[Sampler[T_co]],
        dataset_index_sampler : ChoicesSampler,
        stop_after : int = sys.maxsize):
            self.stop_after = stop_after
            self.samplers = wrapped
            self.dataset_index_sampler = dataset_index_sampler
            self.offsets = np.roll(dataset.cumulative_sizes, 1)
            self.offsets[0] = 0

    def _generate_item(self, sampler_output : Union[List[int],int], dataset_start_index : int) -> T_co:
        if isinstance(sampler_output, int):
             return sampler_output+dataset_start_index
        else:
             return [ j+dataset_start_index for j in sampler_output ]

    def __iter__(self) -> Iterator[T_co]:
        rng = copy.deepcopy(self.dataset_index_sampler)
        iters = [
            utils.cycle(ds) for ds in self.samplers ]
        for _ in range(self.stop_after):
            i = rng()
            yield self._generate_item(next(iters[i]), self.offsets[i])

    def __len__(self) -> int:
        return self.stop_after


def make_concat_dataset_item_sampler(
    dataset : ConcatDataset,
    weights : Sequence[float],
    wrapped : Optional[Sequence[Sampler[int]]] = None,
    stop_after : int = sys.maxsize):

    if wrapped is None:
        wrapped = [ RandomSampler(ds) for ds in dataset.datasets ]
    return ConcatDatasetSampler(
        dataset,
        wrapped,
        PseudoRandomChoices(weights),
        stop_after
     )