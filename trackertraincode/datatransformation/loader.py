from typing import Callable, Generator, Any, Generic, TypeVar
from torch.utils.data import Dataset, DataLoader

from trackertraincode.datasets.batch import Batch


class TransformedDataset(Dataset[Batch]):
    def __init__(self, wrapped : Dataset[Batch], transform : Callable[[Batch],Batch]):
        super(TransformedDataset, self).__init__()
        self.transform = transform
        self.wrapped = wrapped
    
    def __len__(self):
        return len(self.wrapped)

    def __iter__(self) -> Generator[Batch,Any,None]:
        for x in self.wrapped:
            yield self.transform(x)

    def __getitem__(self, key) -> Batch:
        return self.transform(self.wrapped[key])


class SegmentedCollationDataLoader:
    def __init__(self,
                 dataset : Dataset[Batch],
                 *,
                            batch_size : int,
                            num_workers : int,
                            segmentation_key_getter : Callable[[Batch],Any],
                            pin_memory : bool,
                            sampler = None,
                            worker_init_fn = None,
                            postprocess : Callable[[Batch],Batch] = lambda x: x):
        self._loader = DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            sampler = sampler,
            num_workers = num_workers,
            collate_fn=Batch.Collation(segmentation_key_getter),
            worker_init_fn=worker_init_fn,
            pin_memory=pin_memory
        )
        self._postprocess = postprocess
    
    def __iter__(self) -> Generator[list[Batch],Any,None]:
        for items in self._loader:
            assert isinstance(items, list)
            yield [ self._postprocess(item) for item in items ]
    
    def iter_unrolled(self) -> Generator[Batch,Any,None]:
        for items in self:
            yield from items

    def __len__(self):
        return len(self._loader)

T_co = TypeVar("T_co", covariant=True)


class PostprocessingLoader(Generic[T_co]):
    def __init__(self, *args, **kwargs):
        self._postprocess = kwargs.pop('postprocess', None) or (lambda x: x)
        self._loader = DataLoader[T_co](*args, **kwargs)

    @property
    def dataset(self) -> Dataset:
        return self._loader.dataset

    def __iter__(self):
        for items in self._loader:
            yield self._postprocess(items)

    def __len__(self):
        return len(self._loader)


class SampleBySampleLoader(Generic[T_co]):
    def __init__(self,
                 dataset : Dataset[T_co],
                 *,
                            num_workers : int,
                            pin_memory : bool = False,
                            shuffle = False,
                            sampler = None,
                            worker_init_fn = None,
                            postprocess : Callable[[T_co],T_co] | None = None):
        self._loader = DataLoader(
            dataset = dataset,
            batch_size = max(1,num_workers),
            sampler = sampler,
            num_workers = num_workers,
            collate_fn=lambda items: items,
            worker_init_fn=worker_init_fn,
            pin_memory=pin_memory,
            shuffle = shuffle,
            drop_last = False,
        )
        self._postprocess = postprocess or (lambda x: x)

    @property
    def dataset(self) -> Dataset:
        return self._loader.dataset

    def __iter__(self) -> Generator[T_co,Any,None]:
        for items in self._loader:
            assert isinstance(items, list)
            yield from (self._postprocess(item) for item in items)
    
    def __len__(self):
        return len(self._loader.dataset)