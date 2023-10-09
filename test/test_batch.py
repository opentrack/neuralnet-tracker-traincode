import numpy as np
import torch
import enum

from trackertraincode.datasets.batch import Batch, Metadata

class MyTag(enum.Enum):
    POSEBATCH = 1
    CLASSIFYBATCH = 2


class FieldCategories(enum.Enum):
    IMG = 'img'
    ROI = 'roi'

def make_pose_batch(bs, seq=None, imagesize = 7):
    meta = Metadata(imagesize, bs, MyTag.POSEBATCH, seq, { 'image' : FieldCategories.IMG, 'roi' : FieldCategories.ROI})
    return Batch(
            meta,
            image = torch.rand(meta.prefixshape+(3,imagesize,imagesize)),
            roi = torch.rand(meta.prefixshape+(4,)))


def make_facedet_batch(bs, seq=None, imagesize = 42):
    meta = Metadata(imagesize, bs, MyTag.CLASSIFYBATCH, seq, { 'image' : FieldCategories.IMG })
    return Batch(
        meta,
        image = torch.rand(meta.prefixshape+(3,imagesize,imagesize)),
        isface = torch.rand(meta.prefixshape))


def test_meta_properties_still():
    m = Metadata(129, 0)
    assert m.batchsize == 0
    assert m.image_wh == (129,129)
    assert m.imagesize == 129
    assert m.prefixshape == ()


def test_meta_properties_video():
    m = Metadata(129, 0, seq=[0, 2, 5])
    assert m.prefixshape == (5,)
    assert m.sequence_start_end == [(0,2),(2,5)]


def test_meta_properties_still_batch():
    m = Metadata(129, 32)
    assert m.prefixshape == (32,)


def test_meta_properties_video_batch():
    m = Metadata(129, 2, seq=[0, 2, 5])
    assert m.prefixshape == (5,)
    assert m.sequence_start_end == [(0,2),(2,5)]


def test_collate_stills():
    samples = [
        make_facedet_batch(0) for _ in range(5)
    ] + [
        make_pose_batch(0) for _ in range(3)
    ]
    collate = Batch.Collation(divide_by_tag=True)
    batches = { b.meta.tag:b for b  in collate(samples) }

    assert batches[MyTag.CLASSIFYBATCH].meta.batchsize == 5
    assert batches[MyTag.CLASSIFYBATCH].meta.seq is None
    assert batches[MyTag.CLASSIFYBATCH].meta.imagesize == 42
    assert batches[MyTag.CLASSIFYBATCH]['image'].shape[0] == 5

    assert batches[MyTag.POSEBATCH].meta.batchsize == 3
    assert batches[MyTag.POSEBATCH].meta.seq is None
    assert batches[MyTag.POSEBATCH].meta.imagesize == 7
    assert batches[MyTag.POSEBATCH]['image'].shape[0] == 3


def test_collate_divide_by_image_size():
    samples = [
        make_facedet_batch(0, imagesize=7) for _ in range(2)
    ] + [
        make_facedet_batch(0, imagesize=9) for _ in range(3)
    ] + [
        make_pose_batch(0, imagesize=7) for _ in range(4)
    ] + [
        make_pose_batch(0, imagesize=9) for _ in range(5)
    ]
    collate = Batch.Collation(divide_by_image_size=True)
    first, second, third, last = collate(samples)
    assert first.meta.batchsize == 2
    assert second.meta.batchsize == 3
    assert third.meta.batchsize == 4
    assert last.meta.batchsize == 5


def test_collate_divide_by_both():
    samples = [
        make_facedet_batch(0) for _ in range(5)
    ] + [
        make_facedet_batch(0, imagesize=9) for _ in range(3)
    ]
    collate = Batch.Collation(divide_by_image_size=True)
    first, second = collate(samples)
    assert first.meta.batchsize == 5
    assert second.meta.batchsize == 3


def test_collate_still_with_one_tag():
    samples = [ make_facedet_batch(0) for _ in range(5) ]
    batch = Batch.collate(samples)
    assert batch is not None
    assert torch.all(batch['image'] == torch.stack([s['image'] for s in samples]))


def test_collate_videos():
    samples = [ make_facedet_batch(0,seq=(0,9+i)) for i in range(3) ]
    batch = Batch.collate(samples)
    assert batch.meta.batchsize == 3
    assert batch.meta.seq == [0,9,19,30]
    assert batch['image'].shape[0] == 30
    assert torch.all(batch['image'] == torch.cat([s['image'] for s in samples]))


def test_collate_video_batches():
    samples = [
        make_facedet_batch(2,seq=(0,2,5)),
        make_facedet_batch(2,seq=(0,6,7)),
    ]
    batch = Batch.collate(samples)
    assert batch.meta.batchsize == 4
    assert batch.meta.seq == [0,2,5,11,12]
    assert batch['image'].shape[0] == 12
    assert torch.all(batch['image'] == torch.cat([samples[0]['image'],samples[1]['image']]))


def test_collate_still_batches():
    samples = [
        make_facedet_batch(5),
        make_facedet_batch(7),
    ]
    batch = Batch.collate(samples)
    assert batch.meta.batchsize == 12
    assert batch.meta.seq is None
    assert batch['image'].shape[0] == 12
    assert torch.all(batch['image'] == torch.cat([samples[0]['image'],samples[1]['image']]))


def test_with_batchdim():
    sample = make_facedet_batch(0)
    wbd = sample.with_batchdim()
    assert wbd.meta.batchsize == 1
    assert wbd.meta.seq is None

    sample = make_facedet_batch(7)
    wbd = sample.with_batchdim()
    assert wbd.meta.batchsize == 7
    assert wbd.meta.seq is None

    sample = make_facedet_batch(0, seq=[0,5,7])
    wbd = sample.with_batchdim()
    assert wbd.meta.batchsize == 1
    assert wbd.meta.seq == [0,5,7]

    sample = make_facedet_batch(2, seq=[0,5,7])
    wbd = sample.with_batchdim()
    assert wbd.meta.batchsize == 2
    assert wbd.meta.seq == [0,5,7]


def test_iter_frames_video_batch():
    videos = make_facedet_batch(2, seq=[0,5,7])
    frames = list(videos.iter_frames())
    assert len(frames) == 7
    assert len(frames[0]['image'].shape) == 3
    assert frames[0].meta.batchsize == 0
    assert frames[0].meta.seq is None
    assert frames[0].meta.image_wh == videos.meta.image_wh
    assert frames[0].meta.categories == videos.meta.categories
    for i, frame in enumerate(frames):
        assert torch.all(frame['image'] == videos['image'][i])


def test_iter_frames_still():
    sample = make_facedet_batch(0)
    same_sample, = list(sample.iter_frames())
    assert sample.meta == same_sample.meta


def test_iter_sequences():
    videobatch = make_facedet_batch(2, seq=[0,5,7])
    video1, video2 = list(videobatch.iter_sequences())
    assert video1.meta.batchsize == 0
    assert video1.meta.seq == (0,5)
    assert torch.all(video1['image'] == videobatch['image'][:5])
    assert video2.meta.batchsize == 0
    assert video2.meta.seq == (0,2)
    assert torch.all(video2['image'] == videobatch['image'][5:7])