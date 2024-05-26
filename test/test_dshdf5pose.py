import h5py
from pathlib import Path
import numpy as np

from unittest.mock import create_autospec

import trackertraincode.datasets.dshdf5pose as dshdf5pose
import trackertraincode.datasets.dshdf5 as dshdf5
import trackertraincode.datasets.batch as batch


def make_test2_hdf5(tmpdir : Path):
    N = 10
    H = 8
    W = 7
    C = 4
    F = 4
    rois = np.random.randint(0,42,size=(N,F))
    imgs = np.random.randint(0,42,size=(N,H,W,C)).astype(np.uint8)
    sequence_starts = [ 0, 2, N ]
    with h5py.File(tmpdir / 'test.h5', 'w') as f:
        ds = dshdf5.create_dataset(f, 'rois', data = rois)
        ds.attrs['category'] = str(dshdf5pose.FieldCategory.roi)
        ds = dshdf5.ImageVariableLengthBufferDs.create(f, 'images', size=N)
        ds.attrs['category'] = str(dshdf5pose.FieldCategory.image)
        for i, img in enumerate(imgs):
            ds[i] = img
        f.create_dataset('sequence_starts', data=sequence_starts)
    return (tmpdir / 'test.h5'), (rois,imgs), sequence_starts


def test_Hdf5PoseDataset(tmpdir : Path):
    filename, (data,imgs), sequence_starts = make_test2_hdf5(tmpdir)
    h,w = imgs.shape[1:3]

    def mytrafo(sample):
        return sample

    mytrafo = create_autospec(mytrafo, side_effect = mytrafo)

    ds = dshdf5pose.Hdf5PoseDataset(filename, transform = mytrafo, dataclass = 42)

    assert np.alltrue(ds.sequence_starts == np.asarray(sequence_starts))

    sample = ds[0]
    assert isinstance(sample, batch.Batch)
    assert 'image' in sample
    assert 'roi' in sample
    assert 'individual' in sample
    assert sample.meta.tag == 42
    assert sample.meta.batchsize == 0
    assert sample.meta.image_wh == (w,h)
    assert sample.meta.seq is None
    assert sample.meta.categories == { 
        'roi' : dshdf5pose.FieldCategory.roi,
        'image' : dshdf5pose.FieldCategory.image
    }
    assert mytrafo.called


def test_Hdf5PoseVideoDataset(tmpdir : Path):
    filename, (data,imgs), sequence_starts = make_test2_hdf5(tmpdir)
    h,w = imgs.shape[1:3]

    def my_trafo(sample):
        return sample

    my_trafo = create_autospec(my_trafo, side_effect = my_trafo)

    def my_frame_trafo(sample):
        return sample

    my_frame_trafo = create_autospec(my_frame_trafo, side_effect = my_frame_trafo)

    N0, N1 = 2, 5

    ds = dshdf5pose.Hdf5PoseVideoDataset(filename, transform = my_trafo, frame_transform = my_frame_trafo, min_sequence_size=N0, max_sequence_size=N1, dataclass = 42)

    assert len(data)//N1 <= len(ds) <= len(data)//N0

    sample = ds[0]
    assert isinstance(sample, batch.Batch)
    assert 'image' in sample
    assert 'roi' in sample
    assert 'individual' in sample
    assert sample.meta.tag == 42
    assert sample.meta.batchsize == 0
    assert sample.meta.image_wh == (w,h)
    start, end = sample.meta.seq
    assert 0 == start
    assert N0 <= end <= N1
    assert sample.meta.categories == { 
        'roi' : dshdf5pose.FieldCategory.roi,
        'image' : dshdf5pose.FieldCategory.image
    }
    assert my_trafo.call_count == 1
    assert my_frame_trafo.call_count == (end-start)
