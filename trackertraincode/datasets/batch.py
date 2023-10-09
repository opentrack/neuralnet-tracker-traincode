import torch
from dataclasses import dataclass, field, fields
from typing import Union, Tuple, Dict, Any, Optional, List, Iterator, DefaultDict
import numpy as np
import enum
import copy
from collections import defaultdict

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
class Batch(dict):
    _concat_func = {
        torch.Tensor: (lambda l: torch.cat([x for x in l], dim=0)),
        np.ndarray: (lambda l: np.concatenate([x for x in l], axis=0))
    }

    def __init__(self, meta : Metadata, *data, **kwargs):
        self.meta : Metadata = meta
        super().__init__(*data, **kwargs)

    @property
    def device(self):
        val = next(iter(self.values()))
        return val.device

    def __str__(self):
        seq_str = f',N={self.meta.seq[-1][-1]}' if self.meta.seq is not None else ''
        return f"Batch({self.meta.tag},B={self.meta.batchsize}{seq_str})"

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


    class Collation(object):
        def __init__(self, divide_by_tag : bool = True, divide_by_image_size : bool = False):
            get_key_for_divide_by_size_and_tag = lambda b: (b.meta.image_wh, b.meta.tag)
            get_key_for_divide_by_size = lambda b: b.meta.image_wh
            get_key_for_divide_by_tag = lambda b: b.meta.tag
            get_key_for_no_divide = lambda b: True
            self._get_division_key = {
                (True, True) : get_key_for_divide_by_size_and_tag,
                (False, True) : get_key_for_divide_by_size,
                (True, False): get_key_for_divide_by_tag,
                (False,False) : get_key_for_no_divide
            }[divide_by_tag, divide_by_image_size]
            self._expect_single_item = not (divide_by_tag or divide_by_image_size)

        def __call__(self, samples : List["Batch"]) -> List["Batch"]:
            '''
            Concatenates the input into one big batch
            '''
            divisions = defaultdict(list)
            for item in samples:
                assert isinstance(item, Batch), f"Expected list of Batch types. Got {type(item)}"
                divisions[self._get_division_key(item)].append(item)
            batches = list(map(Batch._collate_single_class, divisions.values()))
            if self._expect_single_item:
                batches, = batches # Unpack single item. If different tags and divide_by_tag is false then it's user error.
            return batches

    collate = Collation(divide_by_tag=False)

    @staticmethod
    def _combine_samples(samples : List["Batch"], first : "Batch") -> Dict:
        assert all(s.meta.prefixshape != () for s in samples)
        combined_data = {}
        for k in first.keys():
            assert isinstance(first[k], (torch.Tensor, np.ndarray)), "Only tensor and ndarray is supported for now"
            concat = Batch._concat_func[type(first[k])]            
            t = concat((s[k] for s in samples))
            combined_data[k] = t
        return combined_data


    @staticmethod
    def _collate_videos(samples : List["Batch"], first : "Batch") -> "Batch":
        '''
        Concatenates along the batch dimension
        '''
        # TODO: assert they are all equal
        combined_data = Batch._combine_samples(samples, first)
        lengths = np.asarray([0] + [s.meta.seq[-1] for s in samples])
        # Shift the sequence starts to account for them now starting one after another.
        offsets = np.cumsum(lengths)[:-1]
        seq = np.concatenate([ np.zeros((1,),dtype=np.int32) ] + [ np.asarray(s.meta.seq[1:])+o for s,o in zip(samples,offsets) ]).tolist()
        # Adjust metadata
        meta = copy.copy(first.meta)
        meta.batchsize = len(seq)-1
        meta.seq = seq
        return Batch(
            meta,
            combined_data.items())

    @staticmethod
    def _collate_stills(samples : List["Batch"], first : "Batch") -> "Batch":
        # Warning: default_collate converts to tensors. So I'm not using it.
        # TODO: assert they are all equal
        samples_as_batches = [ s.with_batchdim() for s in samples ]
        combined_data = Batch._combine_samples(samples_as_batches, first)
        # Adjust metadata
        meta = copy.copy(first.meta)
        meta.batchsize = sum(s.meta.batchsize for s in samples_as_batches)
        return Batch(
            meta,
            combined_data.items())

    @staticmethod
    def _collate_single_class(samples : List["Batch"]) -> "Batch":
        assert len(frozenset(s.meta.image_wh for s in samples))==1, f"Image size mismatch in batch: {frozenset(s.meta.image_wh for s in samples)}"
        first = next(iter(samples))
        return Batch._collate_stills(samples, first) if first.meta.seq is None \
            else Batch._collate_videos(samples, first)
