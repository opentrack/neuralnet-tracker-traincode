#!/usr/bin/env python
# coding: utf-8

from os.path import join, dirname
import numpy as np
import os
import h5py
from scipy.spatial.transform import Rotation

from .dshdf5pose import Hdf5PoseDataset, generate_uniform_cluster_probabilities
import datatransformation as dtr

import utils

from torch.utils.data import Subset, ConcatDataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms


def make_vggface_datasets(transform=None):
    vggface2filename = join(os.environ['DATADIR'],'vggface2-tracker.h5')
    sample_probabilities = generate_uniform_cluster_probabilities(vggface2filename)

    rng = np.random.RandomState(seed=12345)
    indices_test = rng.choice(len(sample_probabilities), p=sample_probabilities, size=1000, replace=False)
    indices_train  = np.setdiff1d(np.arange(len(sample_probabilities)), indices_test)
    sample_probabilities = np.ascontiguousarray(sample_probabilities[indices_train])

    ds_vggface2 = Hdf5PoseDataset(vggface2filename, transform=transform)
    ds_vggface2_train = Subset(ds_vggface2, indices_train)
    ds_vggface2_test = Subset(ds_vggface2, indices_test)
    assert len(sample_probabilities) == len(ds_vggface2_train)
    return ds_vggface2_train, ds_vggface2_test, sample_probabilities


def make_affectnet_dataset(transform=None):
    filename = join(os.environ['DATADIR'],'affectnet_train.h5')
    sample_probabilities = generate_uniform_cluster_probabilities(filename)
    ds = Hdf5PoseDataset(filename, transform=transform)
    assert len(sample_probabilities) == len(ds)
    return ds, sample_probabilities


def make_widerface_datasets(transform=None):
    widerfacefilename = join(os.environ['DATADIR'],'widerfacessingle.h5')
    ds_widerface = Hdf5PoseDataset(widerfacefilename,transform=transform)
    ds_widerface_train = Subset(ds_widerface, np.arange(500, len(ds_widerface)))
    ds_widerface_test = Subset(ds_widerface, np.arange(500))
    sample_probabilities_widerface = np.ones((len(ds_widerface_train),), dtype=np.float64) / len(ds_widerface_train)
    return ds_widerface_train, ds_widerface_test, sample_probabilities_widerface


def indices_without_extreme_poses(filename):
    with h5py.File(filename, 'r') as f:
        rot = Rotation.from_quat(f['quats'][...])
        coords = f['coords'][...]
    p,y,r = np.asarray([utils.inv_rotation_conversion_from_hell(r) for r in rot]).T
    threshold = np.pi*99./180.
    mask = (np.abs(p)<threshold) & (np.abs(y)<threshold) & (np.abs(r)<threshold) & (coords[:,-1] >= 0.)
    indices, = np.nonzero(mask)
    return indices


def make_aflw2k3d_dataset(remove_extreme_poses = True, transform=None):
    filename = join(os.environ['DATADIR'],'aflw2k.h5')
    aflw = Hdf5PoseDataset(filename, transform=transform)
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
    ds_grimaces_aflw = Subset(Hdf5PoseDataset(filename, transform=transform), indices)
    return ds_grimaces_aflw


def make_aflw2k3d_datasets():
    filename = join(os.environ['DATADIR'],'aflw2k.h5')
    aflw = Hdf5PoseDataset(filename)
    aflw_train = Subset(aflw, np.arange(400,  len(aflw)))
    aflw_test  = Subset(aflw, np.arange(400))
    return aflw_train, aflw_test


def make_300wlp_dataset():
    return Hdf5PoseDataset(join(os.environ['DATADIR'],'300wlp.h5'))


def make_vggface2_based_pose_estimation_loaders(inputsize, batchsize, use_low_sample_count, pytorch_ignite_format=False):
    train_trafo = transforms.Compose([
        dtr.ApplyRoiRandomized(),
        dtr.Rescale((inputsize,inputsize)),
        dtr.AdaptiveBrightnessContrastDistortion(),
        dtr.Flip(),
        dtr.Normalize(monochrome=True),
        dtr.ToTensor()
    ])
    test_trafo = transforms.Compose([
        dtr.ApplyRoi(),
        dtr.Rescale(inputsize),
        dtr.CenterCrop(inputsize),
        dtr.Normalize(monochrome=True),
        dtr.ToTensor()
    ])
    headpose_ds_trafo = transforms.Compose([
        dtr.InPlaneRotation(),
        dtr.InjectZeroKeypoints3d(),
        dtr.InjectPoseEnable(),
        dtr.InjectHasFaceEnable(False),
        dtr.InjectHasFaceTrue()
    ])
    facedet_ds_trafo = transforms.Compose([
        dtr.InjectZeroPose(),
        dtr.InjectZeroKeypoints3d(),
        dtr.InjectHasFaceEnable(True),
        dtr.InjectZeroFullHeadBox()])

    ds_vggface2_train, ds_vggface2_test, vgg_train_sample_probabilities = make_vggface_datasets(transform=headpose_ds_trafo)
    ds_affectnet_train, affectnet_train_sample_probabilities = make_affectnet_dataset(transform=headpose_ds_trafo)
    ds_widerface_train, ds_widerface_test, sample_probabilities_widerface = make_widerface_datasets(transform=facedet_ds_trafo)

    ds_test = dtr.TransformedDataset(ConcatDataset([ds_widerface_test, ds_vggface2_test]), test_trafo)
    ds_train = dtr.TransformedDataset(ConcatDataset([ds_widerface_train, ds_affectnet_train, ds_vggface2_train]), train_trafo)
    # Widerface is 1/10 times as likely to be sampled as vggface
    sample_probabilities = np.concatenate([
        sample_probabilities_widerface*0.1, 
        affectnet_train_sample_probabilities*0.1,
        vgg_train_sample_probabilities*0.8
    ])

    assert len(ds_train) == len(sample_probabilities), f"{len(ds_train)} == {len(sample_probabilities)}?!"

    loader_trafo_test = [ dtr.MoveToGpu() ]
    loader_trafo_train = [ dtr.BlurNoiseDistortion() ] + loader_trafo_test
    if pytorch_ignite_format:
        loader_trafo_test += [ dtr.SplitImageFromTargets() ]
        loader_trafo_train += [ dtr.SplitImageFromTargets() ]

    num_samples = 512 if use_low_sample_count else 32*1024
    train_sampler = WeightedRandomSampler(sample_probabilities, num_samples=num_samples)
    train_loader = dtr.PostprocessingDataLoader(ds_train, 
                            batch_size=batchsize,
                            num_workers=5,
                            sampler = train_sampler,
                            postprocess = transforms.Compose(loader_trafo_train))
    test_loader = dtr.PostprocessingDataLoader(ds_test, 
                            batch_size=batchsize,
                            num_workers=5,
                            postprocess = transforms.Compose(loader_trafo_test))

    return train_loader, test_loader


def make_validation_loaders(return_aflw2k3d : bool, return_vggface2 : bool, inputsize = 129, batchsize=196):
    test_trafo = transforms.Compose([
        dtr.PutRoiFromLandmarks(),
        dtr.InjectPoseEnable(),
        dtr.ApplyRoi(),
        dtr.Rescale(inputsize),
        dtr.Normalize(monochrome=True),
        dtr.ToTensor()
    ])
    # facedet_ds_trafo = transforms.Compose([
    #     dtr.InjectZeroPose(),
    #     dtr.InjectZeroKeypoints3d(),
    #     dtr.InjectHasFaceEnable(True),
    #     dtr.InjectZeroFullHeadBox()])

        # ds_test_biwi = Hdf5PoseDataset(join(datadir,'biwi.h5'), shuffle=True, subset=None, transform=transforms.Compose([
        #     datatransformation.InjectZeroKeypoints3d(),
        #     datatransformation.InjectPoseEnable(),
        #     *testpreprocess,
        #     *normalize_and_tensor
        # ]))

    loaders = {}

    def add_loader(name, ds):
        nonlocal loaders
        loaders[name] = dtr.PostprocessingDataLoader(ds, 
                                        batch_size=batchsize,
                                        shuffle=False, 
                                        num_workers=5,
                                        postprocess = None)

    if return_vggface2:
        _, ds_vggface2_test, _ = make_vggface_datasets(transform=test_trafo)
        add_loader('vggface2', ds_vggface2_test)
    
    if return_aflw2k3d:
        ds_aflw2k3d = make_aflw2k3d_dataset(transform=test_trafo)
        ds_aflw2k3d_grimaces = make_aflw2k3d_grimaces_dataset(transform=test_trafo)
        add_loader('aflw2k3d', ds_aflw2k3d)
        add_loader('aflw2k3d_grimaces', ds_aflw2k3d_grimaces)

    return loaders

    #_, ds_widerface_test, _ = make_widerface_datasets(transform=facedet_ds_trafo)

    # biwi_loader = datatransformation.PostprocessingDataLoader(ds_test_biwi, 
    #                             batch_size=32,
    #                             shuffle=False, 
    #                             num_workers=5,
    #                             postprocess = None)
