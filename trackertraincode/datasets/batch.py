import torch
from dataclasses import dataclass, field
from typing import Union, Tuple, Dict, Any, Optional, List, Iterator, Callable
import numpy as np
import enum
import copy
from collections import defaultdict
from torch.utils.data._utils.pin_memory import pin_memory

TensorOrArray = Union[torch.Tensor, np.ndarray]
OptionalTensorOrArray = Union[None, torch.Tensor, np.ndarray]
Tag = Any

@dataclass
class Metadata():
    _imagesize : Union[int, Tuple[int,int]]
    batchsize : int
    tag : Optional[Any] = field(default=None)
    seq : Optional[List[int]] = field(default=None)
    categories : Dict[str,Any] = field(default_factory=dict)

    @property
    def image_wh(self):
        return self._imagesize if isinstance(self._imagesize, tuple) else (self._imagesize, self._imagesize)

    @property
    def imagesize(self):
        assert isinstance(self._imagesize, int)
        return self._imagesize

    @property
    def sequence_start_end(self):
        assert self.seq
        return list(zip(self.seq[:-1], self.seq[1:]))

    @property
    def prefixshape(self):
        return (self.seq[-1],) if self.seq else ((self.batchsize,)  if self.batchsize else ())

    @property
    def is_single_frame(self):
        return self.seq is None and self.batchsize==0


@dataclass
class Batch:
    _concat_func = {
        torch.Tensor: (lambda l: torch.cat([x for x in l], dim=0)),
        np.ndarray: (lambda l: np.concatenate([x for x in l], axis=0))
    }

    def __init__(self, meta : Metadata, *data, **kwargs):
        self.meta : Metadata = meta
        self._data : dict[str,TensorOrArray] = dict(*data, **kwargs)

    @staticmethod
    def from_data_with_categories(meta : Metadata, *args, **kwargs):
        """Create Batch taking arguments which would create a dict with (tensor,category) values."""
        with_categories = dict(*args, **kwargs)
        meta = copy.copy(meta)
        meta.categories.update(((k,c) for k,(_,c) in with_categories.items()))
        return Batch(meta, ((k,v) for k,(v,_) in with_categories.items()))

    @property
    def device(self):
        val = next(iter(self.values()))
        return val.device

    def items(self):
        return self._data.items()

    def __getitem__(self, k):
        return self._data[k]
    
    def __setitem__(self, k, v):
        self._data[k] = v

    def __delitem__(self, k):
        del self._data[k]

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def __contains__(self, k):
        return k in self._data

    def pop(self, k):
        return self._data.pop(k)

    def __str__(self):
        seq_str = f',N={self.meta.seq[-1][-1]}' if self.meta.seq is not None else ''
        return f"Batch({self.meta.tag},B={self.meta.batchsize}{seq_str})"

    def get_category(self, k, default = None) -> Any | None:
        assert k in self._data
        return self.meta.categories.get(k,default)

    def with_batchdim(self) -> "Batch":
        """
        Returns a view where the batch size is set to 1 and the additional dimension is
        added to all tensors if it hadn't a batch dimension already.
        """
        if self.meta.batchsize > 0:
            return self
        meta = copy.copy(self.meta)
        meta.batchsize = max(meta.batchsize,1)
        if self.meta.seq is not None:
            return Batch(meta, self.items())
        else:
            return Batch(
                meta,
                ((k,v[None,...]) for k,v in self.items()))

    def iter_frames(self) -> Iterator["Batch"]:
        '''
        Iterates over individual samples in the batch which are returned
        themselves as Batch instances but with batchsize=0. In case of
        sequences it will return each frame separately.
        '''
        if self.meta.is_single_frame:
            yield self
        else:
            n, = self.meta.prefixshape
            meta = copy.copy(self.meta)
            meta.batchsize = 0
            meta.seq = None
            for i in range(n):
                yield Batch(meta,((k,v[i,...]) for k,v in self.items()))

    def iter_sequences(self) -> Iterator["Batch"]:
        assert self.meta.seq is not None
        for a,b in self.meta.sequence_start_end:
            meta = copy.copy(self.meta)
            meta.batchsize = 0
            meta.seq = (0,b-a)
            yield Batch(
                meta,
                ((k,v[a:b,...]) for k,v in self.items())
            )

    def undo_collate(self) -> List["Batch"]:
        '''
        Similar to iter_frames but this function will return entire sequences
        which are in the batch.
        '''
        if self.meta.seq:
            yield from self.iter_sequences()
        else:
            yield from self.iter_frames()

    def pin_memory(self):
        return Batch(self.meta, pin_memory(self._data))

    def copy(self):
        '''Shallow copy.'''
        return Batch(self.meta, **self._data)

    def to(self, *args, **kwargs):
        assert all(isinstance(x,torch.Tensor) for x in self._data.values()), "Only applicable to PyTorch"
        return Batch(self.meta, ((k,v.to(*args,**kwargs)) for k,v in self._data.items()))


    class Collation(object):
        def __init__(self, key_getter : Callable[["Batch"],Any] | None = None):
            self._key_getter = key_getter if key_getter is not None else (lambda b: True)
            self._divide_samples = key_getter is not None


        def __call__(self, samples : List["Batch"]) -> List["Batch"] | "Batch":
            '''
            Concatenates the input into one big batch
            '''
            divisions = defaultdict(list)
            for item in samples:
                assert isinstance(item, Batch), f"Expected list of Batch types. Got {type(item)}"
                divisions[self._key_getter(item)].append(item)
            batches = list(map(self._collate_single_class, divisions.values()))
            if not self._divide_samples:
                batches, = batches
            return batches


        def _collate_single_class(self, samples : List["Batch"]) -> "Batch":
            # TODO: assert they are all equal in kind (sequence or frame or batch)
            first = next(iter(samples))
            return self._collate_stills(samples, first) if first.meta.seq is None \
                else self._collate_videos(samples, first)


        def _collate_videos(self, samples : List["Batch"], first : "Batch") -> "Batch":
            '''
            Concatenates along the batch dimension
            '''
            # TODO: assert the samples are all equal
            return Batch(
                self._combine_metadata(samples, first),
                self._combine_samples(samples, first).items())


        def _collate_stills(self, samples : List["Batch"], first : "Batch") -> "Batch":
            return Batch(
                self._combine_metadata(samples, first),
                self._combine_samples([ s.with_batchdim() for s in samples ], first))


        def _combine_metadata(self, samples : List["Batch"], first : "Batch") -> Metadata:
            meta = copy.copy(first.meta)
            if first.meta.seq is None:
                meta.batchsize = sum(max(s.meta.batchsize,1) for s in samples)
            else:
                lengths = np.asarray([0] + [s.meta.seq[-1] for s in samples])
                # Shift the sequence starts to account for them now starting one after another.
                offsets = np.cumsum(lengths)[:-1]
                seq = np.concatenate([ np.zeros((1,),dtype=np.int32) ] + [ np.asarray(s.meta.seq[1:])+o for s,o in zip(samples,offsets) ]).tolist()
                # Adjust metadata
                meta = copy.copy(first.meta)
                meta.batchsize = len(seq)-1
                meta.seq = seq
            return meta

        def _combine_samples(self, samples : List["Batch"], first : "Batch") -> dict[str,TensorOrArray]:
            assert all(s.meta.prefixshape != () for s in samples)
            combined_data = {}
            for k in first.keys():
                concat = Batch._concat_func[type(first[k])]
                combined_data[k] = concat([s[k] for s in samples])
            return combined_data

    collate = Collation()
