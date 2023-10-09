import numpy as np
import torch
import h5py
import math
from typing import List, Dict, Optional, Iterator, Sequence, Tuple, NamedTuple, Union, Any, Callable
import functools
from strenum import StrEnum

from trackertraincode.datasets import batch
from trackertraincode.datasets.dshdf5 import TorchHdf5DatasetBase, Whitelist, MaybeWrappedH5Dataset, ImageVariableLengthBufferDs, create_dataset

Tag = Any


class FieldCategory(StrEnum):
    general = ''
    image = 'img'
    quat = 'q'
    xys = 'xys'
    roi = 'roi'
    points = 'pts' # Landmarks
    semseg = 'seg'
imagelike_categories = [FieldCategory.image, FieldCategory.semseg]


_inconsistent_name_mapping = dict([
    ('images', 'image'),
    ('keys', 'image'),
    ('seg_image','semseg'),
    ('rois', 'roi'),
    ('coords', 'coord'),
    ('quats','pose'),
    ('pt3d_68','pt3d_68'),
    ('pt2d_68','pt2d_68'),
    ('shapeparams','shapeparam'),
    ('hasface', 'hasface')])


_field_default_names = {
    FieldCategory.image : 'images',
    FieldCategory.semseg : 'semseg',
    FieldCategory.quat : 'quats',
    FieldCategory.xys : 'coords',
    FieldCategory.roi : 'rois',
}


def create_pose_dataset(
        g : h5py.Group, 
        kind : FieldCategory,
        name : Optional[str] = None,
        count : Optional[int] = None,
        shape_wo_batch_dim : Optional[Tuple[int,...]] = None,
        data = None,
        dtype = None,
        **kwargs):
    def equal_or_updated(x, update):
        assert (x is None) or (update is None) or (x == update)
        return update if x is None else x
    shape_postfix_by_kind = {
        FieldCategory.quat : (4,),
        FieldCategory.xys : (3,),
        FieldCategory.roi : (4,),
    }.get(kind, None)
    if name is None:
        name = _field_default_names.get(kind, None)
    assert name in _inconsistent_name_mapping, f"Got {name} which should have been in {_inconsistent_name_mapping.keys()}"
    if kind in (FieldCategory.image, FieldCategory.semseg):
        assert shape_wo_batch_dim is None
        assert dtype is None
        shape = (count,)
    elif kind in (FieldCategory.quat, FieldCategory.xys, FieldCategory.roi):
        shape = (count,) + shape_postfix_by_kind
    elif kind == FieldCategory.points:
        shape = (count,) + (None, None)
    elif kind == FieldCategory.general:
        shape = (count,)
    else:
        assert False, "Not implemented"
    if kind in (FieldCategory.image, FieldCategory.semseg):
        assert dtype is None
    elif kind in (FieldCategory.quat, FieldCategory.xys, FieldCategory.roi, FieldCategory.points):
        assert np.dtype(dtype) in (np.float16, np.float32, np.float64)
    elif kind == FieldCategory.general:
        assert (dtype is not None) or (data is not None)
    if shape_wo_batch_dim is not None:
        if kind == FieldCategory.general:
            shape = (count,) + shape_wo_batch_dim
        shape = (count,) + tuple(
            equal_or_updated(x,u) for x,u in zip(shape[1:],shape_wo_batch_dim))
    if data is not None:
        data = np.asarray(data)
        shape = shape + tuple([None]*(data.ndim-len(shape))) # Pad shape with None's
        # Then fill in missing dimensions from data. And check if the other ones match.
        shape = tuple(
            equal_or_updated(x,u) for x,u in zip(shape,data.shape))
        assert data.shape == shape, f"Expected shape {shape} but got from data the shape {data.shape}"
    assert all(x is not None for x in shape)
    if kind == FieldCategory.image:
        ds = ImageVariableLengthBufferDs.create(g, name, count, **kwargs)
    elif kind == FieldCategory.semseg:
        ds = ImageVariableLengthBufferDs.create(g, name, count, lossy=False, **kwargs)
    else:
        ds = create_dataset(g, name, shape, dtype, shape, data, **kwargs)
    ds.attrs['category'] = kind.value
    return ds


def _find_image_size_and_give_channel_dim(values, categories):
    h, w = None, None
    # Note that itertools.count is a good bit slower than enumerate
    iter = ((i, value) for i, (category, value) in enumerate(zip(categories, values)) if (category in imagelike_categories))
    for i, value in iter:
        if value.ndim==2:
            values[i] = value = value[...,None]
        new_h, new_w, _ = value.shape
        if h is None:
            h, w = new_h, new_w
        else:
            assert (h,w) == (new_h,new_w), "Differently sized images in one sample are not supported"
    assert (w is not None) and (h is not None), f"Currently requires an image. Couldn't find one in {categories}"
    return w, h


def _change_strange_types(value):
    if value.dtype in (torch.float16, torch.float64):
        value = value.to(torch.float32)
    return value


Field2Categories = Dict[str,FieldCategory]


def _get_categories_of_h5datasets(names_datasets : List[Tuple[str,MaybeWrappedH5Dataset]]) -> Field2Categories:
    return {
        name:ds.attrs.get('category', default=FieldCategory.general) for name,ds in names_datasets }


default_whitelist = [
    '/images',
    '/keys',
    '/rois',
    '/coords',
    '/quats',
    '/pt3d_68',
    '/pt2d_68',
    '/shapeparams',
    '/semseg',
    '/seg_image',
    '/hasface'
]


def _transfrom_to_pose_sample(sample : List[Tuple[str,np.ndarray]], dataclass : Tag, categories_mapping : Field2Categories):
    names, values = list(zip(*sample))
    categories = [ categories_mapping[n] for n in names ]
    values = list(map(_change_strange_types, values))
    names = [ _inconsistent_name_mapping.get(n,n) for n in names ]
    w,h = _find_image_size_and_give_channel_dim(values, categories)
    sample = batch.Batch(
        batch.Metadata((w, h), 0, dataclass, None, categories = dict(zip(names, categories))),
            dict(zip(names, values)))
    return sample


class Hdf5PoseDataset(TorchHdf5DatasetBase):
    def __init__(self, filename, transform=None, monochrome=True, dataclass : Tag = None, whitelist : Whitelist = None):
        whitelist = whitelist or default_whitelist
        self._sequence_starts = None
        self._add_individual_to_sample = lambda x, i : x # Nothing to do if sequence data is not available
        super(Hdf5PoseDataset, self).__init__(filename, monochrome, whitelist)
        self.transform = (lambda x: x) if transform is None else transform
        self.dataclass = dataclass

    def _init_from_file(self, f : h5py.File, whitelist : Whitelist):
        names_datasets = super()._init_from_file(f, whitelist)
        self._categories = _get_categories_of_h5datasets(names_datasets)
        if "sequence_starts" in f:
            self._sequence_starts = np.array(f['sequence_starts'][...]).astype(np.int32)
            self._frame_to_individual = np.concatenate([
                np.full(b-a, i, dtype=np.int32) for i,(a,b) in enumerate(self.sequences)
            ])
            def add_individual(self : Hdf5PoseDataset, sample, index):
                sample['individual'] = torch.tensor(self._frame_to_individual[index], dtype=torch.int32)
                return sample
            self._add_individual_to_sample = lambda sample, index: add_individual(self, sample, index)
        return names_datasets

    @property
    def sequence_starts(self):
        return self._sequence_starts

    @functools.cached_property
    def sequences(self):
        return np.stack([self.sequence_starts[:-1], self.sequence_starts[1:]], axis=-1)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample = _transfrom_to_pose_sample(sample, self.dataclass, self._categories)
        sample = self._add_individual_to_sample(sample, index)
        sample['index'] = torch.tensor(index, dtype=torch.int32)
        return self.transform(sample)

    def load_cluster_uniform_sampling_weights(self):
        with h5py.File(self.filename, 'r') as f:
            assert 'cluster-labels' in f
            labels = np.array(f['cluster-labels'][...]).astype(np.int64)
        cluster_indices, counts = np.unique(labels, return_counts=True)
        label2weight = np.full(np.amax(cluster_indices)+1, np.nan, dtype=np.float64)
        label2weight[cluster_indices] = np.reciprocal(counts.astype(np.float64))
        weights = label2weight[labels]
        assert np.all(np.isfinite(weights))
        return weights


# class Hdf5StillPoseVideoDataset(TorchHdf5DatasetBase):
#     '''
#     Repeats still images a couple of times ...
#     This is used as additional dataset to train recurrent networks to hold the output still when the image is not changing.
#     '''
#     def __init__(self, filename, sequencelength, frame_transform=None, transform=None, monochrome=True, dataclass : Tag = None, whitelist : Whitelist = None):
#         super().__init__(
#             filename, 
#             monochrome = monochrome)
#         self.transform = (lambda x: x) if transform is None else transform
#         self.frame_transform = (lambda x: x) if frame_transform is None else frame_transform
#         self.sequencelength = sequencelength

#     def __len__(self):
#         return self.frame_count

#     def __getitem__(self, index):
#         if index<0 or index >= len(self):
#             raise IndexError
#         self._ensure_h5opened()
#         sample = self._load_sample(index)
#         out, = batch.Batch.collate([ self.frame_transform(sample) for _ in range(self.sequencelength) ])
#         out.meta.batchsize = 0
#         out.meta.seq = (0, self.sequencelength)
#         out = self.transform(out)
#         return out


class Hdf5PoseVideoDataset(TorchHdf5DatasetBase):
    def __init__(self, filename, min_sequence_size, max_sequence_size, frame_transform=None, transform=None, monochrome=True, dataclass : Tag = None, whitelist : Whitelist = None):
        '''
        Creates small batches comprised of different images of the same individual
        '''
        self.min_sequence_size = min_sequence_size
        self.max_sequence_size = max_sequence_size
        whitelist = whitelist or default_whitelist
        super().__init__(
            filename, 
            monochrome = monochrome,
            whitelist=whitelist)
        self.dataclass = dataclass
        self.transform = (lambda x: x) if transform is None else transform
        self.frame_transform = (lambda x: x) if frame_transform is None else frame_transform

    def _init_from_file(self, f : h5py.File, whitelist : Whitelist):
        names_datasets = super()._init_from_file(f, whitelist)
        self._categories = _get_categories_of_h5datasets(names_datasets)
        assert "sequence_starts" in f # Must have sequences to sample two images from
        self.sequence_starts = np.array(f['sequence_starts'])
        sequences = zip(self.sequence_starts[:-1], self.sequence_starts[1:])
        sequences = sum((self._postprocess_sequence(*s, self.min_sequence_size, self.max_sequence_size) for s in sequences), [])
        self.sequences = sequences

    @staticmethod
    def _postprocess_sequence(a, b, min_sequence_size, max_sequence_size):
        if b-a < min_sequence_size:
            return []
        if b-a > max_sequence_size:
            # Split sequences in equally sized parts.
            # Then expand the parts symmetrical to the maximum size.
            # Overlap is thereby allowed. Return list of new subsequences.
            splits = math.ceil((b-a)/max_sequence_size)
            centers = np.floor((np.arange(splits)+0.5)*(b-a)/splits)
            starts = np.maximum(0, centers - max_sequence_size//2)
            starts = np.minimum(b-a-max_sequence_size, starts)
            starts = starts.astype(np.int64) + a
            ends = starts + max_sequence_size
            return [ *zip(starts, ends) ]
        return [ (a,b) ]

    def __len__(self):
        return len(self.sequences)

    def _load_sample(self, sequence_index, index):
        s = _transfrom_to_pose_sample(super().__getitem__(index), self.dataclass, self._categories)
        s['individual'] = torch.tensor(sequence_index, dtype=torch.int32)
        s = self.frame_transform(s)
        return s

    def __getitem__(self, index):
        if index<0 or index >= len(self):
            raise IndexError
        a, b = self.sequences[index]
        out = batch.Batch.collate([ self._load_sample(index,i) for i in range(a,b) ])
        out.meta.batchsize = 0
        out.meta.seq = [0, b-a]
        out = self.transform(out)
        return out

    def load_cluster_uniform_sampling_weights(self):
        with h5py.File(self.filename, 'r') as f:
            assert 'cluster-labels' in f
            labels = np.array(f['cluster-labels'][...]).astype(np.int64)
        cluster_indices, counts = np.unique(labels, return_counts=True)
        assert np.all(cluster_indices == np.arange(len(cluster_indices)))
        sequence_weights = []
        for a,b in self.sequences:
            w = np.average(counts[labels[a:b]])
            sequence_weights.append(w)
        sequence_weights = np.reciprocal(np.asarray(sequence_weights).astype(np.float64))
        return sequence_weights