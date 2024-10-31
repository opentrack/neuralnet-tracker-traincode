import numpy as np
from copy import copy
from typing import Callable, Set, Sequence, Union, List, Tuple, Dict, Optional, NamedTuple

from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

import trackertraincode.utils as utils
from trackertraincode.datasets.batch import Batch, Tag


class TransformedDataset(Dataset):
    def __init__(self, wrapped, transform):
        super(TransformedDataset, self).__init__()
        self.transform = transform
        self.wrapped = wrapped
    
    def __len__(self):
        return len(self.wrapped)

    def __iter__(self):
        for x in self.wrapped:
            yield self.transform(x)

    def __getitem__(self, key):
        return self.transform(self.wrapped[key])


class PostprocessingDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('unroll_list_of_batches', True)
        kwargs.setdefault('collate_fn', collate_list_of_batches)
        self.postprocess = kwargs.pop('postprocess', None)
        if self.postprocess is None:
            self.postprocess = utils.identity
        self._iter = self._iteration_with_lists_unrolled \
            if kwargs.pop('unroll_list_of_batches') else \
                self._iteration_over_list_of_batches
        super(PostprocessingDataLoader, self).__init__(*args, **kwargs)
    
    def _iteration_with_lists_unrolled(self):
        for items in super(PostprocessingDataLoader, self).__iter__():
            if not isinstance(items, list):
                items = [ items ]
            yield from map(self.postprocess, items)

    def _iteration_over_list_of_batches(self):
        for itemlist in super(PostprocessingDataLoader, self).__iter__():
            itemlist = [ self.postprocess(item) for item in itemlist ]
            yield itemlist

    def __iter__(self):
        yield from self._iter()


class ComposeChoiceByTag(object):
    def __init__(self, tag2aug : Dict[Tag, Callable]):
        self.tag2aug = tag2aug
    def __call__(self, batch : Batch):
        try:
            aug = self.tag2aug[batch.meta.tag]
        except KeyError:
            return batch
        return aug(batch)


class ComposeChoice(object):
    def __init__(self, *augs):
        self.augs = augs
    def __call__(self, batch : Batch):
        i = np.random.randint(len(self.augs))
        return self.augs[i](batch)



def collate_list_of_batches(samples):
    def pack(list_or_sample):
        return list_or_sample if isinstance(list_or_sample, list) else [ list_or_sample ]
    items = sum((pack(s) for s in samples), [])
    first = next(iter(items))
    if hasattr(type(first), 'collate'):
        return type(first).collate(items)
    else:
        return default_collate(items)


def undo_collate(batch):
    """
        Generator that takes apart a batch. Opposite of collate_fn.
    """
    if hasattr(type(batch), 'undo_collate'):
        yield from type(batch).undo_collate(batch)
    else:
        assert isinstance(batch, dict), f"Got {type(batch)}"
        keys = [*batch.keys()]
        assert keys
        N = batch[keys[0]].shape[0]  # First dimension should be the batch size
        for i in range(N):
            yield { k:batch[k][i] for k in keys }


class DeleteKeys(object):
    def __init__(self, *keys):
        self.keys = keys

    def __call__(self, sample):
        for k in self.keys:
            del sample[k]
        return sample


class WhitelistKeys(object):
    def __init__(self, *keys):
        self.keys = frozenset(keys)

    def __call__(self, sample):
        for k in list(sample.keys()):
            if not k in self.keys:
                del sample[k]
        return sample