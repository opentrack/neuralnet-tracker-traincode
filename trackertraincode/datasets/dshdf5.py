from typing import List, Dict, Optional, Iterator, Sequence, Tuple, NamedTuple, Union, Any, Callable
from os.path import join, dirname, splitext, isfile, basename
import numpy as np
import h5py
import torch
from trackertraincode.datasets.preprocessing import imdecode, imencode, ImageFormat, which_image_format
from trackertraincode.utils import glob_hdf_datasets
from torch.utils.data import Dataset
from functools import cached_property
import PIL.Image

variable_length_hdf5_buffer_dtype = h5py.special_dtype(vlen=np.dtype('uint8'))


class DatasetEncoding(object):
    varsize_array_buffer = 'varsize_array_buffer'
    varsize_image_buffer = 'varsize_image_buffer'
    image_filename = 'image_filename'


def _chunk_shape(shape, maxshape):
    _chunksize = 1024
    if shape is None:
        shape = maxshape
    n, rest = shape[0], shape[1:]
    return (min(_chunksize, n),)+rest


def _ensure_image_is_in_expected_color_mode(img : np.ndarray, monochrome : bool):
    assert not monochrome or img.ndim == 2
    assert monochrome or (img.ndim == 3 and img.shape[-1]==3)
    # Ensure we always have the channel dimension.
    # if img.ndim == 2:
    #     img = img[...,None]
    return img


class ImageDs(object):
    def __init__(self):
        self.monochrome = True

    def _decode(self, buffer):
        decoded = imdecode(buffer, color=False if self.monochrome else 'rgb')
        decoded = _ensure_image_is_in_expected_color_mode(decoded, self.monochrome)
        return decoded

    def __getitem__(self, index : int) -> np.ndarray:
        raise RuntimeError("not implemented")

    def __len__(self):
        raise RuntimeError("not implemented")


class ImageVariableLengthBufferDs(ImageDs):
    def __init__(self, ds : h5py.Dataset):
        super().__init__()
        if ds.attrs.get('lossy', True):
            self._format = ImageFormat.JPG
            self._encode = lambda value: imencode(value, format=ImageFormat.JPG, quality=95)
        else:
            self._format = ImageFormat.PNG
            self._encode = lambda value: imencode(value, format=ImageFormat.PNG)
        assert ds.attrs['storage'] == DatasetEncoding.varsize_image_buffer
        self.ds = ds

    def __getitem__(self, index : int) :
        return self._decode(self.ds[index])

    def __setitem__(self, index : int, value) :
        assert (isinstance(value, np.ndarray) and value.dtype==np.uint8) or isinstance(value,PIL.Image.Image)
        if isinstance(value, PIL.Image.Image):
            value = np.asarray(value)
        if len(value.shape) in (2,3):
            value = self._encode(value)
        else:
            if which_image_format(value) != self._format:
                raise ValueError(f"Buffer for lossy/lossless data must be encoded as jpg/png, got {which_image_format(value)}")
            assert len(value.shape) == 1
        self.ds[index] = value

    def __len__(self):
        return len(self.ds)
    
    @cached_property
    def attrs(self):
        return self.ds.attrs

    @staticmethod
    def create(g : h5py.Group, name : str, size : int, maxsize : Optional[int] = None, lossy : bool = True):
        ds = g.create_dataset(name, (size,), variable_length_hdf5_buffer_dtype, maxshape = (maxsize,), chunks=_chunk_shape((size,),(maxsize,)))
        ds.attrs['storage'] = DatasetEncoding.varsize_image_buffer
        ds.attrs['lossy'] = lossy
        return ImageVariableLengthBufferDs(ds)


class ImagePathDs(ImageDs):
    def __init__(self, ds : h5py.Dataset):
        super().__init__()
        assert ds.attrs['storage'] == DatasetEncoding.image_filename
        self._ds = ds
        self._filelist = ImagePathDs._find_filenames(ds)

    @staticmethod
    def _find_filenames(ds : h5py.Dataset):
        supported_extensions = ('.jpg', '.png', '.jpeg')
        names : Sequence[bytearray] = ds[...]
        first = names[0].decode('ascii')
        extensions_to_try = supported_extensions if (splitext(first.lower())[1] not in supported_extensions) else ('',)
        directories_to_try = [ dirname(ds.file.filename), splitext(ds.file.filename)[0] ]
        found = False
        for root_dir in directories_to_try:
            for ext in extensions_to_try:
                if isfile(join(root_dir, first+ext)):
                    found = True
                    break
        if not found:
            raise RuntimeError(f"Cannot find images for image path dataset. Looking for name {first} with roots {directories_to_try} and extensions {extensions_to_try}")
        return [ join(root_dir,s.decode('ascii')+ext) for s in names ]

    def __getitem__(self, index : int):
        with open(self._filelist[index], 'rb') as f:
            buffer = f.read()
        return self._decode(buffer)

    def __len__(self):
        return len(self._filelist)

    @cached_property
    def attrs(self):
        return self._ds.attrs

    @staticmethod
    def create(g : h5py.Group, name, data):
        ds = g.create_dataset(name, data=data)
        ds.attrs['storage'] = DatasetEncoding.image_filename
        return ImagePathDs(ds)


def create_dataset(g : h5py.Group, name : str, shape : Tuple = None, dtype = None, maxshape : Tuple = None, data = None):
    if data is not None:
        data = np.asarray(data)
        assert shape is None or data.shape == shape
    if shape is None:
        assert data is not None
        shape = data.shape
    return g.create_dataset(name, shape, dtype, chunks=_chunk_shape(shape,maxshape), maxshape=maxshape, data=data)


def _quantize(values : np.ndarray):
    assert values.dtype in (np.float32, np.float64)
    minval = np.amin(values, keepdims=True)
    maxval = np.amax(values, keepdims=True)
    buffer = ((values-minval)/(maxval-minval+1.)*256).astype(np.uint8)
    return minval, maxval, buffer


def _dequantize(minval, maxval, buffer, shape):
    buffer = buffer/256.*(maxval-minval+1) + minval
    buffer = buffer.astype(np.float32)
    buffer = buffer.reshape(shape)
    return buffer


class QuantizedVarsizeArrayDs(object):
    def __init__(self, ds : h5py.Dataset):
        assert ds.attrs['storage'] == DatasetEncoding.varsize_array_buffer
        self.ds = ds

    def __getitem__(self, index : int) :
        shape, minval, maxval, buffer = self.ds[index]
        buffer = np.frombuffer(buffer, dtype=np.uint8)
        return _dequantize(minval, maxval, buffer, shape)

    def __setitem__(self, index : int, value : np.ndarray) :
        assert self.ds.attrs['storage'] == DatasetEncoding.varsize_array_buffer
        minval, maxval, buffer = _quantize(value)
        self.ds[index] = (value.shape, float(minval), float(maxval), buffer.ravel())

    @cached_property
    def attrs(self):
        return self.ds.attrs

    def __len__(self):
        return len(self.ds)

    @staticmethod
    def create(g : h5py.Group, name : str, size : int, sample_dimensionality : int, maxsize : Optional[int] = None):
        dt = np.dtype([('shape','i4',(sample_dimensionality,)), ('minval', 'f4'), ('maxval','f4'), ('buffer',variable_length_hdf5_buffer_dtype)])
        ds = g.create_dataset(name, (size,), chunks=_chunk_shape((size,),(maxsize,)), maxshape=(maxsize,), dtype=dt)
        ds.attrs['storage'] = DatasetEncoding.varsize_array_buffer
        return QuantizedVarsizeArrayDs(ds)


MaybeWrappedH5Dataset = Union[h5py.Dataset, QuantizedVarsizeArrayDs, ImageDs]
Whitelist = List[str]


def open_dataset(g : h5py.Group, name : str) -> MaybeWrappedH5Dataset:
    ds = g[name]
    if not 'storage' in ds.attrs:
        return ds
    typeattr = ds.attrs['storage']
    if typeattr == DatasetEncoding.varsize_array_buffer:
        return QuantizedVarsizeArrayDs(ds)
    if typeattr == DatasetEncoding.image_filename:
        return ImagePathDs(ds)
    if typeattr == DatasetEncoding.varsize_image_buffer:
        return ImageVariableLengthBufferDs(ds)
    raise RuntimeError(f"Cannot create dataset wrapper. Unknown value of attribute 'storage': {typeattr}")


def open_all_datasets(root : h5py.Group, whitelist : Whitelist) -> List[Tuple[str, MaybeWrappedH5Dataset]]:
    opened = []
    for ds in glob_hdf_datasets(root, whitelist):
        opened.append((basename(ds.name), open_dataset(root, ds.name)))
    assert len(set(k for k,_ in opened)) == len(opened), "Found datasets must have unique base names."
    return opened


class TorchHdf5DatasetBase(Dataset):
    def __init__(self, filename, monochrome=True, whitelist : Whitelist = None):
        super(Dataset, self).__init__()
        self.monochrome = monochrome
        self.filename = filename
        self.whitelist = whitelist
        self._h5file = None
        self._names_datasets = None
        with h5py.File(self.filename, 'r') as f:
            self._init_from_file(f, whitelist)

    def _init_from_file(self, f : h5py.File, whitelist : Whitelist) -> List[Tuple[str,MaybeWrappedH5Dataset]]:
        names_datasets = open_all_datasets(f, whitelist)
        lengths = [len(v) for _,v in names_datasets ]
        assert lengths and all(l == lengths[0] for l in lengths), f"Inconsistent lengths among data: {[k for k,v in names_datasets]}"
        self._frame_count = lengths[0]
        return names_datasets

    @property
    def frame_count(self):
        return self._frame_count

    def __len__(self):
        return self.frame_count

    def _set_monochrome_flag_to_ds_wrappers(self):
        for _, ds in self._names_datasets.items():
            if isinstance(ds, ImageDs):
                ds.monochrome = self.monochrome

    def _ensure_h5opened(self):
        if self._h5file is None:
            self._h5file = h5py.File(self.filename, 'r')
            # Lazy creation because it needs access to the hdf5 which is lost when the dataset is copied to the remote workers
            self._names_datasets = dict(open_all_datasets(self._h5file, self.whitelist))
            self._set_monochrome_flag_to_ds_wrappers()

    def _get_field(self, ds : MaybeWrappedH5Dataset, index : int):
        item = torch.as_tensor(ds[index])
        return item

    def __getitem__(self, index):
        if index<0 or index >= len(self):
            raise IndexError(f"Index {index} on dataset of length {len(self)}")
        self._ensure_h5opened()
        return  [ (name,self._get_field(dataset, index)) for name, dataset  in self._names_datasets.items() ]

    def close(self):
        assert (self._h5file is None) == (self._names_datasets is None)
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None
            self._names_datasets = None