#!/usr/bin/env python
# coding: utf-8

from os.path import join, dirname
import numpy as np
import cv2
import os
import argparse
from matplotlib import pyplot

import torch.nn as nn
import torch.nn.functional as tf
import torch.optim as optim
from torchvision import transforms
import torch

import datatransformation
import neuralnets.models as models
import vis
import utils
import train
from datasets.dshdf5pose import Hdf5PoseDataset, generate_uniform_cluster_probabilities
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler
from torch.utils.data import Subset, ConcatDataset


def setup_datasets(args):
    inputsize = args.input_size
    bs = args.batchsize
    monochrome = True

    augment = [
        datatransformation.ApplyRoiRandomized(),
        datatransformation.Rescale((inputsize,inputsize)),
        datatransformation.AdaptiveBrightnessContrastDistortion(),
        datatransformation.Flip(),
    ]
    testpreprocess = [
        datatransformation.ApplyRoi(),
        datatransformation.Rescale(inputsize),
    ]
    normalize_and_tensor = [
        datatransformation.Normalize(monochrome=monochrome),
        datatransformation.ToTensor()
    ]

    datadir = os.environ['DATADIR']

    ds_full_300wlp = Hdf5PoseDataset(join(datadir,'300wlp.h5'), transform=transforms.Compose([
        datatransformation.InPlaneRotation(),
        datatransformation.InjectPt3d68Enable(),
        datatransformation.InjectPoseEnable(),
        datatransformation.PutRoiFromLandmarks(),
        *augment,
        *normalize_and_tensor
    ]))

    ds_test_aflw = Hdf5PoseDataset(join(datadir,'aflw2k_test.h5'), transform=transforms.Compose([
        datatransformation.InjectPt3d68Enable(),
        datatransformation.InjectPoseEnable(),
        datatransformation.PutRoiFromLandmarks(),
        *testpreprocess,
        *normalize_and_tensor
    ]))

    train_loader = datatransformation.PostprocessingDataLoader(ds_full_300wlp, 
                            batch_size=bs,
                            num_workers=5,
                            sampler = RandomSampler(ds_full_300wlp, replacement=True, num_samples=32*1024),
                            postprocess = transforms.Compose([
                                    datatransformation.BlurNoiseDistortion(),
                                    datatransformation.MoveToGpu()
                            ]))
    test_loader = datatransformation.PostprocessingDataLoader(ds_test_aflw, 
                            batch_size=bs,
                            num_workers=5,
                            postprocess = datatransformation.MoveToGpu())

    if 0: # Doesn't work with multiprocessing Matplotlib train curve viewer! :-(
        def iterate_predictions(loader):
            for batch in loader:
                for sample in utils.undo_collate(batch):
                    yield vis.unnormalize_sample_to_numpy(sample)

        def drawfunc(sample):
            return vis.draw_dataset_sample(sample, label=False)

        keepalive = vis.matplotlib_plot_iterable(iterate_predictions(train_loader), drawfunc)
        pyplot.show()
        keepalive = vis.matplotlib_plot_iterable(iterate_predictions(test_loader), drawfunc)
        pyplot.show()
        del keepalive

    return train_loader, test_loader


def create_optimizer(net : models.FinetuningNetworkWrapper, args):
    optimizer = optim.Adam(net.get_finetune_parameters(args.lr), lr=args.lr)
    #optimizer = optim.SGD(to_optimize, lr=args.lr, momentum=0.9)
    if args.find_lr:
        n_epochs = args.epochs
        lr_max = 1.e-1
        # lr_max = lr * b**n -> ln(lr_max/lr) = n*ln(b) -> 
        base = np.power(lr_max/args.lr, 1./n_epochs)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda e: base**e)
    else:
        n_epochs = args.epochs
        n_epochs_up = max(1,args.epochs*3//10)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, 0.01*args.lr, args.lr, n_epochs_up, n_epochs-n_epochs_up, cycle_momentum=type(optimizer) is not optim.Adam)
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args.epochs], gamma=0.1)
    return optimizer, scheduler, n_epochs


def main():
    parser = argparse.ArgumentParser(description="Trains the model")
    parser.add_argument('--batchsize', help="The batch size to train with", type=int, default=128)
    parser.add_argument('--lr', help='learning rate', type=float, default=1.e-4)
    parser.add_argument('--start', help='model checkpoint to start from', type=str, default='')
    parser.add_argument('--find-lr', help="Enable learning rate finder mode", action='store_true', default=False)
    parser.add_argument('--epochs', help="Number of epochs", type=int, default=100)
    args = parser.parse_args()

    cv2.setNumThreads(1)

    #net = models.NetworkWithPointHead(enable_point_head=False, enable_face_detector=True, enable_full_head_box=True)
    net = models.LocalAttentionNetwork(enable_face_detector=True, enable_full_head_box=True)
    args.input_size = net.input_resolution

    train_loader, test_loader = setup_datasets(args)
    
    if args.start:
        state_dict = torch.load(args.start)
        net.load_state_dict(state_dict)

    net = models.FinetuningNetworkWrapper(net)

    net.train() # Sets BN parameters frozen, so they are not optimized!

    net.cuda()

    C = train.Criterion
    criterions = [
        C('rot', train.QuatPoseLoss2(), 0.4, train=True, test=True),
        C('coord', train.CoordPoseLoss(), 0.4, train=True, test=True),
        C('box', train.BoxLoss(), 0.1, train=True, test=True),
        C('quatregularization', train.QuaternionNormalizationRegularization(), 1.e-6, train=True)
    ]

    save_callback = train.SaveBestCallback(net, 'rot', model_dir=join(dirname(__file__),'..','model_files'))
    callbacks = [ save_callback ]

    optimizer, scheduler, n_epochs = create_optimizer(net, args)
    train.run_the_training(n_epochs,
                        optimizer,
                        net,
                        train_loader,
                        test_loader,
                        criterions,
                        callbacks,
                        scheduler = scheduler)

    torch.save(net.state_dict(), join(dirname(__file__),'..','model_files','last_'+type(net).__name__+'.ckpt'))


if __name__ == '__main__':
    main()