#!/usr/bin/env python
# coding: utf-8

from os.path import join, dirname
import numpy as np
import os
import h5py
from scipy.spatial.transform import Rotation
import enum
from typing import List, Dict, Set, Sequence
import random
from functools import partial
import cv2
from copy import copy

from torch.utils.data import Subset, ConcatDataset
from torchvision import transforms
import torch

from trackertraincode.datasets.batch import Batch, Tag
from trackertraincode.datasets.dshdf5pose import Hdf5PoseDataset, FieldCategory
from trackertraincode.datasets.randomized import make_concat_dataset_item_sampler
import trackertraincode.datatransformation as dtr
import trackertraincode.utils as utils


class  Tag(enum.Enum):
    POSE_WITH_LANDMARKS = 1
    SELF_SUPERVISED_POSE = 2
    FACE_DETECTION = 3
    ONLY_LANDMARKS = 4
    ONLY_LANDMARKS_25D = 5
    EYES_AND_LANDMARKS_2D = 6
    ONLY_POSE = 7
    POSE_WITH_LANDMARKS_3D_AND_2D = 8
    ONLY_LANDMARKS_2D = 9
    SEMSEG = 10


class Id(enum.Enum):
    _300WLP = 2
    SYNFACE = 5
    WFLW_RELABEL = 6
    AFLW2k3d = 8
    BIWI = 9
    WIDER = 11
    _300VW = 12
    LAPA = 13
    REPO_300WLP = 15  
    WFLW_LP = 16
    LAPA_MEGAFACE_LP = 17


# See https://pytorch.org/docs/stable/notes/randomness.html
# The workers use numpy to generate samples from datasets. So the seeds better be different.
def seed_worker(worker_id):
    worker_seed = torch.utils.data.get_worker_info().seed % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    # Also set numpy number of threads
    import mkl
    mkl.set_num_threads(1)


# For now this shall be good enough.
def whiten_image(image : torch.Tensor):
    return image.sub(0.5)


def unwhiten_image(image : torch.Tensor):
    return image.add(0.5)


def whiten_batch(batch : Batch):
    batch = copy(batch)
    for k,v in batch.items():
        if dtr.get_category(batch, k) == FieldCategory.image:
            batch[k] = whiten_image(v)
    return batch


def make_biwi_datasest(transform=None):
    filename = join(os.environ['DATADIR'],'biwi.h5')
    return Hdf5PoseDataset(filename, transform=transform, dataclass=Tag.ONLY_POSE)


def make_300vw_dataset(transform=None):
    filename = join(os.environ['DATADIR'],'300vw.h5')
    return Hdf5PoseDataset(filename, transform=transform, dataclass=Tag.ONLY_LANDMARKS_2D)


def make_lapa_dataset(transform=None):
    filename = join(os.environ['DATADIR'],'lapa.h5')
    return Hdf5PoseDataset(filename, transform=transform, dataclass=Tag.ONLY_LANDMARKS_2D)


def make_lapa_megaface_lp_dataset(transform=None):
    filename = join(os.environ['DATADIR'],'lapa-megaface-augmented.h5')
    return Hdf5PoseDataset(filename, transform=transform, dataclass=Tag.POSE_WITH_LANDMARKS)


def make_synface_dataset(transform=None):
    filename = join(os.environ['DATADIR'],'microsoft_synface_100000.h5')
    ds = Hdf5PoseDataset(filename, transform=transform, dataclass=Tag.ONLY_LANDMARKS_25D)
    return ds


def make_wflw_relabeled_dataset(transform=None, test_transform=None):
    train = Hdf5PoseDataset(join(os.environ['DATADIR'],'wflw_train.h5'), transform=transform, dataclass=Tag.EYES_AND_LANDMARKS_2D)
    test = Hdf5PoseDataset(join(os.environ['DATADIR'],'wflw_test.h5'), transform=test_transform, dataclass=Tag.EYES_AND_LANDMARKS_2D)
    return train, test


def make_wflw_lp_dataset(transform=None):
    return Hdf5PoseDataset(join(os.environ['DATADIR'],'wflw_augmented_v3.h5'), transform=transform, dataclass=Tag.POSE_WITH_LANDMARKS)


def make_widerface_datasets(transform=None):
    widerfacefilename = join(os.environ['DATADIR'],'widerfacessingle.h5')
    ds_widerface = Hdf5PoseDataset(widerfacefilename,transform=transform, dataclass=Tag.FACE_DETECTION)
    ds_widerface_train = Subset(ds_widerface, np.arange(500, len(ds_widerface)))
    ds_widerface_test = Subset(ds_widerface, np.arange(500))
    return ds_widerface_train, ds_widerface_test


def indices_without_extreme_poses(filename):
    with h5py.File(filename, 'r') as f:
        rot = Rotation.from_quat(f['quats'][...])
        coords = f['coords'][...]
    p,y,r = np.asarray([utils.inv_aflw_rotation_conversion(r) for r in rot]).T
    threshold = np.pi*99./180.
    mask = (np.abs(p)<threshold) & (np.abs(y)<threshold) & (np.abs(r)<threshold) & (coords[:,-1] >= 0.)
    indices, = np.nonzero(mask)
    return indices


def make_aflw2k3d_dataset(remove_extreme_poses = True, transform=None):
    filename = join(os.environ['DATADIR'],'aflw2k.h5')
    aflw = Hdf5PoseDataset(filename, transform=transform, dataclass=Tag.POSE_WITH_LANDMARKS)
    if remove_extreme_poses:
        indices = indices_without_extreme_poses(filename)
        print (f"Filtering {len(aflw)-len(indices)} extreme poses from aflw2k-3d dataset")
        aflw = Subset(aflw, indices)
    return aflw


def make_aflw2k3d_grimaces_dataset(transform=None):
    filename = join(os.environ['DATADIR'],'aflw2k.h5')
    # Selected from the first 400 faces which is our test set.
    indices = np.array([ 39, 236,   0, 129, 164, 356, 359, 256, 136, 375, 226, 392, 119,
        366, 293,  56, 305, 303, 397,  10,  11,  96, 173, 124, 115, 153,
        337,  29, 121, 266, 387, 122,   8,  59, 108, 380, 187, 192, 353,
        257, 162, 363, 331,  14, 163])
    ds_grimaces_aflw = Subset(Hdf5PoseDataset(filename, transform=transform, dataclass=Tag.POSE_WITH_LANDMARKS), indices)
    return ds_grimaces_aflw


def make_aflw2k3d_datasets(transform=None):
    filename = join(os.environ['DATADIR'],'aflw2k.h5')
    aflw = Hdf5PoseDataset(filename, transform=transform, dataclass=Tag.POSE_WITH_LANDMARKS)
    aflw_train = Subset(aflw, np.arange(400,  len(aflw)))
    aflw_test  = Subset(aflw, np.arange(400))
    return aflw_train, aflw_test


def make_300wlp_dataset(transform=None):
    ds = Hdf5PoseDataset(join(os.environ['DATADIR'],'300wlp.h5'), transform=transform, dataclass=Tag.POSE_WITH_LANDMARKS_3D_AND_2D)
    return ds


def make_repro_300wlp_dataset(transform=None):
    ds = Hdf5PoseDataset(join(os.environ['DATADIR'],'reproduction_300wlp-v11.h5'), transform=transform, dataclass=Tag.POSE_WITH_LANDMARKS)
    return ds


def make_ibugmask_dataset(transform=None, test_transform=None):
    train = Hdf5PoseDataset(join(os.environ['DATADIR'],'ibugmask.h5'), transform=transform, dataclass=Tag.SEMSEG)
    test = Hdf5PoseDataset(join(os.environ['DATADIR'],'ibugmask_test.h5'), transform=test_transform, dataclass=Tag.SEMSEG)
    return train, test


def make_myself_dataset(transform=None):
    # Videos of the developer
    return Hdf5PoseDataset(os.path.join(os.environ['DATADIR'],'myself.h5'), transform=transform)


def make_myselfyaw_dataset(transform=None):
    # Videos of the developer
    return Hdf5PoseDataset(os.path.join(os.environ['DATADIR'],'myself-yaw.h5'), transform=transform)


def make_pose_estimation_loaders(inputsize, batchsize, datasets : Sequence[Id], device='cuda', auglevel = 1):
    C = transforms.Compose
    assert auglevel in (0,1,2)

    prepare = [
        dtr.batch_to_torch_nchw,
        dtr.offset_points_by_half_pixel, # For when pixels are considered cell centered
    ]

    headpose_train_trafo_eyes = prepare + [
        dtr.PutRoiFromLandmarks(extend_to_forehead=True),
        dtr.RandomFocusRoi(inputsize, align_corners=False),
        partial(dtr.horizontal_flip_and_rot_90, False, 0.01)
    ]
    headpose_train_trafo = headpose_train_trafo_eyes 
    facedet_train_trafo = prepare + [
        dtr.RandomFocusRoi(inputsize, align_corners=False),
        partial(dtr.horizontal_flip_and_rot_90, False, 0.01)
    ]

    headpose_test_trafo = prepare + [
        dtr.PutRoiFromLandmarks(extend_to_forehead=True),
        dtr.FocusRoi(inputsize, 1.2),
    ]
    facedet_test_trafo = prepare + [
        dtr.FocusRoi(inputsize, 1.2),
    ]

    ds_with_sizes = []

    train_sets = []
    test_sets = []
    dataset_weights = []

    for id, ds_ctor, sample_weight in [
        (Id.SYNFACE, make_synface_dataset, 10000), # Synface has 100k frames. But they are synthetic. Not going to weight them that much. So weight is only 10k.
        (Id.BIWI, make_biwi_datasest, 1000),
        # Tons of frames in 300VW but only 200 individuals. Assume 
        # 20 samples with sufficiently  small correlation.
        (Id._300VW, make_300vw_dataset, 5000.),
        (Id.LAPA, make_lapa_dataset, 20000.),
        (Id.WFLW_LP, make_wflw_lp_dataset, 40000.),
        # There are over 100k frames in the latter but the labels are a bit shitty so I don't weight it as high.
        (Id.LAPA_MEGAFACE_LP, make_lapa_megaface_lp_dataset, 10000.) ]:
            if id not in datasets:
                continue
            train = ds_ctor(transform=C(headpose_train_trafo))
            train_sets.append(train)
            dataset_weights.append(sample_weight)
            ds_with_sizes.append((id, len(train)))

    for id, ds_ctor, sample_weight in [
        (Id.WFLW_RELABEL, make_wflw_relabeled_dataset, 10000.) ]:
        if id not in datasets:
            continue
        train, test = ds_ctor(transform=C(headpose_train_trafo), test_transform=C(headpose_test_trafo))
        train_sets.append(train)
        test_sets.append(test)
        dataset_weights.append(sample_weight)
        ds_with_sizes.append((id, len(train)))

    if Id.AFLW2k3d in datasets:
        train, _ = make_aflw2k3d_datasets(transform=C(headpose_train_trafo))
        train_sets.append(train)
        dataset_weights.append(1000.)
        ds_with_sizes.append((Id.AFLW2k3d, len(train)))

    if Id._300WLP in datasets or Id.REPO_300WLP in datasets:
        # The main dataset with the proper labels.
        assert not ((Id._300WLP in datasets) and (Id.REPO_300WLP in datasets))
        if Id._300WLP in datasets:
            train = make_300wlp_dataset(transform=C(headpose_train_trafo))
            weight = 60000.
            id = Id._300WLP
        else:
            train = make_repro_300wlp_dataset(transform=C(headpose_train_trafo))
            weight = 60_000.
            id = Id.REPO_300WLP
        train_sets.append(train)
        dataset_weights.append(weight)
        ds_with_sizes.append((id, len(train)))
        _, test = make_aflw2k3d_datasets(transform=C(headpose_test_trafo))
        test_sets.append(test)

    if Id.WIDER in datasets:
        train, test = make_widerface_datasets()
        train = dtr.TransformedDataset(train, C(facedet_train_trafo))
        test = dtr.TransformedDataset(test, C(facedet_test_trafo))
        train_sets.append(train)
        test_sets.append(test)
        dataset_weights.append(10000.)
        ds_with_sizes.append((Id.WIDER, len(train)))

    dataset_weights = np.asarray(dataset_weights)    
    dataset_weights = dataset_weights / np.sum(dataset_weights)

    ds_test = ConcatDataset(test_sets)
    ds_train = ConcatDataset(train_sets)

    printed_weights = dataset_weights / np.amax(dataset_weights)*100
    print (f"Train datasets:\n\t", ",\n\t".join(f"{id_}: {sz}  weight: {w:0.1f}" for (id_,sz),w in zip(ds_with_sizes, printed_weights)))
    print (f"Train dataset size {len(ds_train)}")
    print (f"Test set size {len(ds_test)}")

    train_sampler = make_concat_dataset_item_sampler(
        dataset = ds_train,
        weights = dataset_weights)

    loader_trafo_test = [ 
        partial(dtr.normalize_batch, align_corners=False),
        whiten_batch,
        partial(dtr.to_device, 'cuda'),
    ]

    image_augs = [
        dtr.RandomEqualize(p=0.2),
        dtr.RandomPosterize((4.,6.), p=0.01),
        dtr.RandomGamma((0.5, 2.0), p = 0.2),
        dtr.RandomContrast((0.7, 1.5), p = 0.2),
        dtr.RandomBrightness((0.7, 1.5), p = 0.2),
    ]
    if auglevel in  (2, 1, 3):
        image_augs += [
            dtr.RandomGaussianBlur(p=0.1, kernel_size=(5,5), sigma=(1.5,1.5)),
            #dtr.RandomGaussianNoiseWithClipping(std=4./255., p=0.1)
        ]

    loader_trafo_train = [
        partial(dtr.normalize_batch, align_corners=False),
        partial(dtr.to_device, 'cuda'),
        dtr.KorniaImageDistortions(*image_augs, random_apply = 4),
        dtr.KorniaImageDistortions(
            dtr.RandomGaussianNoise(std=4./255., p=0.5),
            dtr.RandomGaussianNoise(std=16./255., p=0.1),
        ),
        whiten_batch
    ]

    train_loader = dtr.PostprocessingDataLoader(ds_train,
                            unroll_list_of_batches = False,
                            batch_size = batchsize,
                            sampler = train_sampler,
                            num_workers = utils.num_workers(),
                            postprocess = transforms.Compose(loader_trafo_train),
                            collate_fn = Batch.Collation(divide_by_tag=True),
                            worker_init_fn = seed_worker)
    test_loader = dtr.PostprocessingDataLoader(ds_test,
                            unroll_list_of_batches = True,
                            batch_size = batchsize*2,
                            num_workers = utils.num_workers(),
                            collate_fn = Batch.Collation(divide_by_tag=True),
                            postprocess = transforms.Compose(loader_trafo_test),
                            worker_init_fn = seed_worker)

    return train_loader, test_loader, len(ds_train)


def make_validation_loader(name, order = None): # inputsize = 129, device='cpu', view_expansion : float = 1.2, auto_level = False):
    test_trafo = transforms.Compose([
        # dtr.batch_to_torch_nchw,
        #dtr.to_tensor,
        dtr.offset_points_by_half_pixel, # For when pixels are considered grid cell centers
        dtr.PutRoiFromLandmarks(extend_to_forehead=True),
        # dtr.FocusRoi(inputsize, view_expansion),
    ])
    # if auto_level:
    #     test_trafo.transforms.append(dtr.KorniaImageDistortions(dtr.RandomEqualize(p=1.0)))

    if name == 'aflw2k3d':
        ds = make_aflw2k3d_dataset(transform=test_trafo)
    elif name == 'aflw2k3d_grimaces':
        ds = make_aflw2k3d_grimaces_dataset(transform=test_trafo)
    elif name == 'myself':
        ds = make_myself_dataset(transform=test_trafo)
    elif name == 'myself-yaw':
        ds = make_myselfyaw_dataset(transform=test_trafo)
    elif name == 'biwi':
        ds = make_biwi_datasest(transform=test_trafo)
    else:
        assert False, "Not implemented"

    if order is not None:
        ds = Subset(ds, order)

    return dtr.PostprocessingDataLoader(
        ds, 
        batch_size=2,
        shuffle=False, 
        num_workers = utils.num_workers(),
        postprocess = None, 
        collate_fn = lambda samples: samples,
        unroll_list_of_batches = True)