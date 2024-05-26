import h5py
from pathlib import Path
import numpy as np
import cv2
import os
from collections import namedtuple

from unittest.mock import patch
import pytest

import trackertraincode.datasets.dshdf5
import trackertraincode.datasets.dshdf5 as dshdf5

assert isinstance(dshdf5.DatasetEncoding.image_filename, str), f"Not a string but a {type(dshdf5.DatasetEncoding.image_filename)}"


Data1 = namedtuple('TestData1', ['datanormal', 'dataimagepath', 'datavarsizeimage1', 'datavarsizeimage2'])


def _make_test_data1():
    datanormal = np.arange(8).reshape(2,4)
    dataimagepath = [ "foo", "bar" ]
    img1 = np.zeros((10,20), dtype=np.uint8)
    img1[1,2] = 255
    img2 = np.zeros((5,10), dtype=np.uint8)
    img2[2,1] = 128
    datavarsizeimage1 = [
        img1, img2
    ]
    img1 = np.random.normal(size=(10,20,2)).astype(np.float32)
    img2 = np.random.normal(size=(5,10,2)).astype(np.float64)
    datavarsizeimage2 = [ img1, img2]
    return Data1(
        datanormal,
        dataimagepath,
        datavarsizeimage1,
        datavarsizeimage2,
    )


def make_testdata1_hdf5(td : Data1, tmpdir : Path):
    os.mkdir(tmpdir / 'test')
    for img, name in zip(td.datavarsizeimage1, td.dataimagepath):
        cv2.imwrite(str(tmpdir / 'test' / (name + '.png')), img)

    with h5py.File(tmpdir / 'test.h5', 'w') as f:
        normalds = dshdf5.create_dataset(f, "normal", (2, 4), 'i4', data = td.datanormal)
        imagepathds = dshdf5.ImagePathDs.create(f, "imagepath", data = td.dataimagepath)
        varsizeimageds1 = dshdf5.ImageVariableLengthBufferDs.create(f, "varsizeimage1", 2)
        varsizeimageds2 = dshdf5.ImageVariableLengthBufferDs.create(f, "varsizeimage2", 2)
        varsizeimageds3 = dshdf5.ImageVariableLengthBufferDs.create(f, "varsizeimage3", 2, lossy=False)
        varsizeimageds4 = dshdf5.ImageVariableLengthBufferDs.create(f, "varsizeimage4", 2, lossy=False)
        for i, img in enumerate(td.datavarsizeimage1):
            varsizeimageds1[i] = img
            varsizeimageds2[i] = dshdf5.imencode(img)
            varsizeimageds3[i] = img
            with pytest.raises(ValueError):
                varsizeimageds4[i] = dshdf5.imencode(img)
            varsizeimageds4[i] = dshdf5.imencode(img, dshdf5.ImageFormat.PNG)
        vararrayds = dshdf5.QuantizedVarsizeArrayDs.create(f, "varsizearray", 2, 3)
        for i, img in enumerate(td.datavarsizeimage2):
            vararrayds[i] = img
    
    return tmpdir / 'test.h5'


def test_H5DatasetWrappers(tmpdir : Path):
    td = _make_test_data1()
    h5filename = make_testdata1_hdf5(td, tmpdir)
    
    with h5py.File(h5filename, 'r') as f:
        assert dshdf5.open_dataset(f,"normal") is not None
        assert dshdf5.open_dataset(f,"imagepath") is not None
        assert dshdf5.open_dataset(f,"varsizeimage1") is not None
        assert dshdf5.open_dataset(f,"varsizeimage2") is not None
        assert dshdf5.open_dataset(f,"varsizeimage3") is not None
        assert dshdf5.open_dataset(f,"varsizeimage4") is not None
        assert dshdf5.open_dataset(f,"varsizearray") is not None

        normalds = f["normal"]
        imagepathds = dshdf5.ImagePathDs(f["imagepath"])
        varsizeimageds1 = dshdf5.ImageVariableLengthBufferDs(f['varsizeimage1'])
        varsizeimageds2 = dshdf5.ImageVariableLengthBufferDs(f['varsizeimage2'])
        varsizeimageds3 = dshdf5.ImageVariableLengthBufferDs(f['varsizeimage3'])
        varsizeimageds4 = dshdf5.ImageVariableLengthBufferDs(f['varsizeimage4'])
        vararrayds = dshdf5.QuantizedVarsizeArrayDs(f['varsizearray'])

        assert np.all(normalds[...] == td.datanormal)
        for i, img in enumerate(td.datavarsizeimage1):
            assert np.all(imagepathds[i] == img)
            assert np.allclose(varsizeimageds1[i], img, rtol=1., atol=6.)
            assert np.allclose(varsizeimageds2[i], img, rtol=1., atol=6.)
            assert np.array_equal(varsizeimageds3[i], img)
            assert np.array_equal(varsizeimageds4[i], img)
        for i, img in enumerate(td.datavarsizeimage2):
            assert np.allclose(vararrayds[i], img, rtol=0.1, atol=0.1)


def test_openalldatasets(tmpdir : Path):
    td = _make_test_data1()
    h5filename = make_testdata1_hdf5(td, tmpdir)

    with h5py.File(h5filename, 'r') as f:
        names_datasets = dict(dshdf5.open_all_datasets(f, whitelist = list(map(lambda s: '/'+s, f.keys()))))
        assert "normal" in names_datasets
        assert "imagepath" in names_datasets
        assert "varsizeimage1" in names_datasets
        assert "varsizeimage2" in names_datasets
        assert "varsizearray" in names_datasets


def test_ImageDsColorTransform(tmpdir : Path):
    img_grey = np.zeros((10,20), dtype=np.uint8)
    img_grey[1,2] = 255
    img_color = np.zeros((5,10,3), dtype=np.uint8)
    img_color[2,1] = (128,196,255)
    images = [
        img_grey, img_color
    ]

    with h5py.File(tmpdir / 'test.h5', 'w') as f:
        ds = dshdf5.ImageVariableLengthBufferDs.create(f, "varsizeimage", 2)
        for i, img in enumerate(images):
            ds[i] = img
    
    with h5py.File(tmpdir / 'test.h5', 'r') as f:
        ds = dshdf5.open_dataset(f, 'varsizeimage')
        ds.monochrome = False
        assert np.allclose(ds[0], np.repeat(img_grey[:,:,None],3,-1), rtol=1., atol=6.)
        assert np.allclose(ds[1], img_color, rtol=1., atol=10.)
        ds.monochrome = True
        assert np.allclose(ds[0], img_grey, rtol=1., atol=6.)
        assert np.allclose(ds[1], np.average(img_color,axis=-1), rtol=1., atol=6.)


def make_test2_hdf5(tmpdir : Path):
    N = 10
    F = 2
    data = np.random.randint(0, 42, size=(N,F))
    sequence_starts = [ 0, 2, N ]
    with h5py.File(tmpdir / 'test.h5', 'w') as f:
        dshdf5.create_dataset(f, 'test', data = data)
    return (tmpdir / 'test.h5'), data, sequence_starts


def test_TorchHdf5DatasetBase(tmpdir : Path):
    filename, data, sequence_starts = make_test2_hdf5(tmpdir)
    ds = dshdf5.TorchHdf5DatasetBase(filename, whitelist = ['/test'])
    
    ((name1, data1),) = ds[0]
    ((name2, data2),) = ds[1]

    assert name1 == 'test'
    assert name2 == 'test'
    assert np.array_equal(data1, data[0])
    assert np.array_equal(data2, data[1])


@patch('trackertraincode.datasets.dshdf5.TorchHdf5DatasetBase._set_monochrome_flag_to_ds_wrappers', side_effect=trackertraincode.datasets.dshdf5.TorchHdf5DatasetBase._set_monochrome_flag_to_ds_wrappers, autospec=True)
def test_TorchHdf5DatasetBase_monochrome_propagation(mock__set_monochrome_flag_to_ds_wrappers, tmpdir : Path):
    filename, data, sequence_starts = make_test2_hdf5(tmpdir)
    
    ds = trackertraincode.datasets.dshdf5.TorchHdf5DatasetBase(filename, whitelist=['/test'])
    assert ds[0] is not None
    assert ds.monochrome == True
    mock__set_monochrome_flag_to_ds_wrappers.assert_called()

    ds = trackertraincode.datasets.dshdf5.TorchHdf5DatasetBase(filename, whitelist=['/test'], monochrome=False)
    assert ds[0] is not None
    assert ds.monochrome == False
    mock__set_monochrome_flag_to_ds_wrappers.assert_called()