#!/usr/bin/env python
# coding: utf-8

from typing import List, Dict, Set, Sequence, Any, Type, Optional
from os.path import join, dirname
import numpy as np
import os
import h5py
from scipy.spatial.transform import Rotation
import enum
import random
from functools import partial
import cv2
from copy import copy

from torch.utils.data import Subset, ConcatDataset
from torchvision import transforms
import torch

from trackertraincode.datasets.batch import Batch, Tag
from trackertraincode.datasets.dshdf5pose import Hdf5PoseDataset
from trackertraincode.datasets.randomized import make_concat_dataset_item_sampler
import trackertraincode.datatransformation as dtr
import trackertraincode.utils as utils


class  Tag(enum.Enum):
    POSE_WITH_LANDMARKS = 1
    SELF_SUPERVISED_POSE = 2
    FACE_DETECTION = 3
    ONLY_LANDMARKS = 4
    ONLY_LANDMARKS_25D = 5
    ONLY_POSE = 7
    POSE_WITH_LANDMARKS_3D_AND_2D = 8
    ONLY_LANDMARKS_2D = 9
    SEMSEG = 10
    POSE_WITH_LMKS_NO_SHAPE_PARAMS = 11


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
    REPO_300WLP_WO_EXTRA = 18
    PANOPTIC_CMU = 19
    REPLICANT_FACE = 20


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


def make_biwi_datasest(transform=None):
    filename = join(os.environ['DATADIR'],'biwi-v3.h5')
    return Hdf5PoseDataset(filename, transform=transform, dataclass=Tag.ONLY_POSE)


def make_300vw_dataset(transform=None):
    filename = join(os.environ['DATADIR'],'300vw.h5')
    return Hdf5PoseDataset(filename, transform=transform, dataclass=Tag.ONLY_LANDMARKS_2D)


def make_lapa_dataset(transform=None):
    filename = join(os.environ['DATADIR'],'lapa.h5')
    return Hdf5PoseDataset(filename, transform=transform, dataclass=Tag.ONLY_LANDMARKS_2D)


def make_lapa_megaface_lp_dataset(transform=None):
    filename = join(os.environ['DATADIR'],'lapa-megaface-augmented-v2.h5')
    return Hdf5PoseDataset(filename, transform=transform, dataclass=Tag.POSE_WITH_LANDMARKS)


def make_synface_dataset(transform=None):
    filename = join(os.environ['DATADIR'],'microsoft_synface_100000-v1.1.h5')
    ds = Hdf5PoseDataset(filename, transform=transform, dataclass=Tag.ONLY_LANDMARKS_25D)
    return ds


def make_wflw_relabeled_dataset(transform=None, test_transform=None):
    train = Hdf5PoseDataset(join(os.environ['DATADIR'],'wflw_train.h5'), transform=transform, dataclass=Tag.ONLY_LANDMARKS_2D)
    test = Hdf5PoseDataset(join(os.environ['DATADIR'],'wflw_test.h5'), transform=test_transform, dataclass=Tag.ONLY_LANDMARKS_2D)
    return train, test


def make_wflw_lp_dataset(transform=None):
    return Hdf5PoseDataset(join(os.environ['DATADIR'],'wflw_augmented_v4.h5'), transform=transform, dataclass=Tag.POSE_WITH_LANDMARKS)


def make_widerface_datasets(transform=None):
    widerfacefilename = join(os.environ['DATADIR'],'widerfacessingle.h5')
    ds_widerface = Hdf5PoseDataset(widerfacefilename,transform=transform, dataclass=Tag.FACE_DETECTION)
    ds_widerface_train = Subset(ds_widerface, np.arange(500, len(ds_widerface)))
    ds_widerface_test = Subset(ds_widerface, np.arange(500))
    return ds_widerface_train, ds_widerface_test


def make_panoptic_datasets(transform=None):
    ds = Hdf5PoseDataset(join(os.environ['DATADIR'],'panoptic-v2.h5'), transform=transform, dataclass=Tag.ONLY_POSE, coord_convention_id=1)
    test_indices = np.random.RandomState(seed=1234567).choice(len(ds), 1024, replace=False)
    train_indices = np.setdiff1d(np.arange(len(ds)), test_indices)
    ds_train = Subset(ds, train_indices)
    ds_test  = Subset(ds, test_indices)
    return ds_train, ds_test


def make_replicant_face_datasets(transform=None):
    ds = Hdf5PoseDataset(join(os.environ['DATADIR'],'replicant-face-v4-like-300wlp.h5'), transform=transform, dataclass=Tag.POSE_WITH_LMKS_NO_SHAPE_PARAMS)
    return ds


def make_panoptic_trainset(transform=None):
    return make_panoptic_datasets(transform)[0]


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


def make_aflw2k3d_closedeyes_dataset(remove_extreme_poses = True, transform=None):
    filename = join(os.environ['DATADIR'],'aflw2k3d-closedeyes.h5')
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


def make_repro_300wlp_dataset(transform=None, with_eye_aug=True):
    filename = {
        True : 'reproduction_300wlp-v12.h5',
        False : 'reproduction_300wlp_simple.h5'
    }[with_eye_aug]
    ds = Hdf5PoseDataset(join(os.environ['DATADIR'],filename), transform=transform, dataclass=Tag.POSE_WITH_LANDMARKS)
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


def add_constant_transform(key : str, dtype : Type[torch.dtype], value : Any):
    tensor_value = torch.as_tensor(value, dtype=dtype)
    def _add_constant_transform_func(batch : Batch):
        batch = copy(batch)
        batch[key] = tensor_value
        return batch
    return _add_constant_transform_func


def _make_roi_augmentations(inputsize : int, stage : str, mode : str, rotation_aug_angle : float = 0.):
    assert mode in ['extent_to_forehead', 'original', 'landmarks'], f"got {mode}"
    assert stage in ['train','eval']

    extension_factor = {
        'original' : 1.1,
        'extent_to_forehead' : 1.1,
        'landmarks' : 1.2
    }[mode]
    cropping_aug = {
        'eval': dtr.batch.FocusRoi(inputsize, extension_factor),
        'train':  dtr.batch.RandomFocusRoi(inputsize, rotation_aug_angle=rotation_aug_angle, extension_factor=extension_factor)
    }[stage]

    if mode == 'original':
        # More reasonable approach for second round of runs
        return [ cropping_aug ]
    elif mode == 'landmarks':
        return [
            dtr.batch.PutRoiFromLandmarks(extend_to_forehead=False),
            cropping_aug,
            dtr.batch.PutRoiFromLandmarks(extend_to_forehead=False)
        ]
    else:
        return [
            dtr.batch.PutRoiFromLandmarks(extend_to_forehead=True),
            cropping_aug
            # Forgot to regenerate the bounding box
        ]


def make_pose_estimation_loaders(
        inputsize, 
        batchsize, 
        datasets : Sequence[Id], 
        dataset_weights : Dict[Id,float] = None,
        use_weights_as_sampling_frequency : bool = True,
        enable_image_aug : bool = True,
        rotation_aug_angle : float = 30.,
        roi_override : str = 'original',
        device : Optional[str] = 'cuda',
    ):
    C = transforms.Compose

    prepare = [
        dtr.batch.offset_points_by_half_pixel, # For when pixels are considered cell centered
    ]

    headpose_train_trafo = prepare + _make_roi_augmentations(inputsize, 'train', roi_override, rotation_aug_angle) + [
        partial(dtr.batch.horizontal_flip_and_rot_90, 0.01),
        partial(dtr.batch.normalize_batch),
    ]

    headpose_test_trafo = prepare + _make_roi_augmentations(inputsize, 'eval', roi_override) + [
        partial(dtr.batch.normalize_batch)
    ]


    if dataset_weights is None:
        dataset_weights = {}

    ds_with_sizes = []
    train_sets = []
    test_sets = []
    train_sets_weight_list = []

    for id, ds_ctor, default_sample_weight in [
        (Id.SYNFACE, make_synface_dataset, 10_000.), # Synface has 100k frames. But they are synthetic. Not going to weight them that much. So weight is only 10k.
        (Id.BIWI, make_biwi_datasest, 1000),
        # Tons of frames in 300VW but only 200 individuals. Assume 
        # 20 samples with sufficiently  small correlation.
        (Id._300VW, make_300vw_dataset, 5000.),
        (Id.LAPA, make_lapa_dataset, 20000.),
        (Id.WFLW_LP, make_wflw_lp_dataset, 40000.),
        # There are over 70k frames in the latter but the labels are a bit shitty so I don't weight it as high.
        (Id.LAPA_MEGAFACE_LP, make_lapa_megaface_lp_dataset, 10000.),
        (Id.PANOPTIC_CMU, make_panoptic_trainset, 20_000.),
        (Id.REPLICANT_FACE, make_replicant_face_datasets, 10_000.) ]:
            if id not in datasets:
                continue
            train = ds_ctor(transform=C(headpose_train_trafo))
            train_sets.append(train)
            train_sets_weight_list.append(dataset_weights.get(id, default_sample_weight))
            ds_with_sizes.append((id, len(train)))

    for id, ds_ctor, default_sample_weight in [
        (Id.WFLW_RELABEL, make_wflw_relabeled_dataset, 10000.) ]:
        if id not in datasets:
            continue
        train, test = ds_ctor(transform=C(headpose_train_trafo), test_transform=C(headpose_test_trafo))
        train_sets.append(train)
        test_sets.append(test)
        train_sets_weight_list.append(dataset_weights.get(id, default_sample_weight))
        ds_with_sizes.append((id, len(train)))

    if (id := Id.AFLW2k3d) in datasets:
        train, _ = make_aflw2k3d_datasets(transform=C(headpose_train_trafo))
        train_sets.append(train)
        train_sets_weight_list.append(dataset_weights.get(id, 1000.))
        ds_with_sizes.append((Id.AFLW2k3d, len(train)))

    _300wlp_variants = [ x for x in datasets if x in [Id._300WLP, Id.REPO_300WLP, Id.REPO_300WLP_WO_EXTRA] ]
    if _300wlp_variants:
        id, = _300wlp_variants # Error if more than one variant was requested
        if id == Id._300WLP:
            train = make_300wlp_dataset(transform=C(headpose_train_trafo))
        elif id == Id.REPO_300WLP_WO_EXTRA :
            train = make_repro_300wlp_dataset(transform=C(headpose_train_trafo),with_eye_aug=False)
        elif id == Id.REPO_300WLP: 
            train = make_repro_300wlp_dataset(transform=C(headpose_train_trafo))
        else:
            assert False, "Bad dataset request"
        train_sets.append(train)
        train_sets_weight_list.append(dataset_weights.get(id, 60_000.))
        ds_with_sizes.append((id, len(train)))

    _, test = make_aflw2k3d_datasets(transform=C(headpose_test_trafo))
    test_sets.append(test)

    if (id := Id.WIDER) in datasets:
        train, test = make_widerface_datasets()
        train = dtr.TransformedDataset(train, C(headpose_train_trafo))
        test = dtr.TransformedDataset(test, C(headpose_test_trafo))
        train_sets.append(train)
        test_sets.append(test)
        train_sets_weight_list.append(dataset_weights.get(id, 10_000.))
        ds_with_sizes.append((Id.WIDER, len(train)))

    train_sets_weight_list = np.asarray(train_sets_weight_list)
    train_sets_frequencies = train_sets_weight_list

    ds_test = ConcatDataset(test_sets)
    ds_train = ConcatDataset(train_sets)

    if not use_weights_as_sampling_frequency:
        train_sets_weight_list = train_sets_weight_list / np.amax(train_sets_weight_list)
        #use_weighting_instead_of_sampling_frequencies:
        for ds, w in zip(ds_train.datasets, train_sets_weight_list):
            assert isinstance(ds, (Hdf5PoseDataset, dtr.TransformedDataset))
            assert isinstance(ds.transform, C)
            ds.transform.transforms.append(add_constant_transform('dataset_weight', torch.float32, w))
        train_sets_frequencies = np.ones_like(train_sets_frequencies)/len(train_sets_frequencies)
    else:
        train_sets_weight_list = train_sets_weight_list / np.sum(train_sets_weight_list)
        train_sets_frequencies = train_sets_weight_list
    print (f"Train datasets:\n\t", ",\n\t".join(f"{id_}: {sz}  weight: {w:0.1f}" for (id_,sz),w in zip(ds_with_sizes, train_sets_weight_list*100)))
    print (f"Train dataset size {len(ds_train)}. Weights {'are frequencies!' if use_weights_as_sampling_frequency else 'scale the losses!'}")
    print (f"Test set size {len(ds_test)}")
    del train_sets_weight_list

    train_sampler = make_concat_dataset_item_sampler(
        dataset = ds_train,
        weights = train_sets_frequencies)

    loader_trafo_test = [ dtr.batch.whiten_batch ]
    if device is not None:
        loader_trafo_test = [ lambda b: b.to(device) ] + loader_trafo_test

    if enable_image_aug:
        image_augs = [
            dtr.batch.KorniaImageDistortions(
                dtr.batch.RandomEqualize(p=0.2),
                dtr.batch.RandomPosterize((4.,6.), p=0.01),
                dtr.batch.RandomGamma((0.5, 2.0), p = 0.2),
                dtr.batch.RandomContrast((0.7, 1.5), p = 0.2),
                dtr.batch.RandomBrightness((0.7, 1.5), p = 0.2),
                dtr.batch.RandomGaussianBlur(p=0.1, kernel_size=(5,5), sigma=(1.5,1.5), silence_instantiation_warning=True),
                random_apply = 4),
            dtr.batch.KorniaImageDistortions(
                dtr.batch.RandomGaussianNoise(std=4./255., p=0.5),
                dtr.batch.RandomGaussianNoise(std=16./255., p=0.1),
            )
        ]
    else:
        image_augs = []
    loader_trafo_train = [ lambda b: b.to(device) ] if device is not None else []
    loader_trafo_train += image_augs + [ dtr.batch.whiten_batch ]
    

    train_loader = dtr.SegmentedCollationDataLoader(ds_train,
                            batch_size = batchsize,
                            sampler = train_sampler,
                            num_workers = utils.num_workers(),
                            postprocess = transforms.Compose(loader_trafo_train),
                            segmentation_key_getter=lambda b: b.meta.tag,
                            worker_init_fn = seed_worker,
                            pin_memory = True)
    test_loader = dtr.PostprocessingLoader[Batch](ds_test,
                            batch_size = batchsize*2,
                            num_workers = utils.num_workers(),
                            postprocess = transforms.Compose(loader_trafo_test),
                            collate_fn = Batch.collate,
                            pin_memory = True,
                            worker_init_fn = seed_worker)

    return train_loader, test_loader, len(ds_train)


def make_validation_dataset(name : str, 
                            order : Sequence[int] |None = None, 
                            use_head_roi = True, 
                            additional_transforms : list[Any] | None = None) -> Hdf5PoseDataset | Subset:
    if additional_transforms is None:
        additional_transforms = []
    test_trafo = transforms.Compose([
        dtr.batch.offset_points_by_half_pixel, # For when pixels are considered grid cell centers
        dtr.batch.PutRoiFromLandmarks(extend_to_forehead=use_head_roi)
    ] + additional_transforms)
    if name == 'aflw2k3d':
        ds = make_aflw2k3d_dataset(transform=test_trafo)
    elif name == 'aflw2k3d_grimaces':
        ds = make_aflw2k3d_grimaces_dataset(transform=test_trafo)
    elif name == 'aflw2k3d_closedeyes':
        ds = make_aflw2k3d_closedeyes_dataset(transform=test_trafo)
    elif name == 'myself':
        ds = make_myself_dataset(transform=test_trafo)
    elif name == 'myself_yaw':
        ds = make_myselfyaw_dataset(transform=test_trafo)
    elif name == 'biwi':
        ds = make_biwi_datasest(transform=test_trafo)
    elif name == 'repro_300_wlp':
        ds = make_repro_300wlp_dataset(transform=test_trafo)
    elif name == 'wflw_lp':
        ds = make_wflw_lp_dataset(transform=test_trafo)
    elif name == 'lapa_megaface_lp':
        ds = make_lapa_megaface_lp_dataset(transform=test_trafo)
    elif name == 'panoptic':
        ds = make_panoptic_datasets(transform=test_trafo)[1]
    elif name == 'replicantface-train':
        ds = make_replicant_face_datasets(transform=test_trafo)
        rng = np.random.default_rng(seed=42)
        ds = Subset(ds, rng.integers(0, len(ds)-1, size=1000))
    else:
        assert False, f"Unknown dataset {name}"

    if order is not None:
        ds = Subset(ds, order)
    return ds


def make_validation_loader(name, 
                           order : Sequence[int] |None = None, 
                           use_head_roi = True, 
                           return_single_samples : bool = False,
                           additional_sample_transform : Any | None = None,
                           additional_batch_transform : Any | None = None):
    if isinstance (additional_sample_transform, transforms.Compose):
        additional_sample_transform = list(additional_sample_transform.transforms)
    ds = make_validation_dataset(name, order, use_head_roi, additional_transforms=additional_sample_transform)
    num_workers = utils.num_workers()
    if return_single_samples:
        return dtr.SampleBySampleLoader[Batch](
            ds, num_workers = num_workers, postprocess=additional_batch_transform)
    elif additional_batch_transform:
        return dtr.PostprocessingLoader[Batch](
            ds, num_workers = num_workers, postprocess=additional_batch_transform, collate_fn = Batch.collate)
    else:
        return dtr.DataLoader(
            ds, 
            num_workers = utils.num_workers(),
            batch_size = 128,
            collate_fn = Batch.collate)