#!/usr/bin/env python
# coding: utf-8

from os.path import join, dirname
import numpy as np
import cv2
import argparse
from matplotlib import pyplot

import torch.optim as optim
import torch

import neuralnets.models as models
import vis
import utils
import train

import datasets.datasets


def setup_datasets(args):
    train_loader, test_loader = datasets.datasets.make_vggface2_based_pose_estimation_loaders(
        inputsize = args.input_size, 
        batchsize = args.batchsize,
        use_low_sample_count= args.find_lr)

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


def create_optimizer(net, args):
    if args.freeze_backbone:        
        to_optimize = list(frozenset(net.parameters()) - frozenset(net.convnet.parameters()))
        for p in net.convnet.parameters():
            p.requires_grad = False
        print ("Backbone frozen!")
    else:
        to_optimize = net.parameters()
    optimizer = optim.Adam(to_optimize, lr=args.lr)
    #optimizer = optim.SGD(to_optimize, lr=args.lr, momentum=0.9)
    if args.find_lr:
        print ("LR finding mode!")
        n_epochs = args.epochs
        lr_max = 1.e-1
        # lr_max = lr * b**n -> ln(lr_max/lr) = n*ln(b) -> 
        base = np.power(lr_max/args.lr, 1./n_epochs)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda e: base**e, verbose=True)
    else:
        n_epochs = args.epochs
        n_epochs_up = max(1,args.epochs*3//10)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, 0.01*args.lr, args.lr, n_epochs_up, n_epochs-n_epochs_up, cycle_momentum=type(optimizer) is not optim.Adam)
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args.epochs], gamma=0.1)
    return optimizer, scheduler, n_epochs


def main():
    np.seterr(all='raise')
    cv2.setNumThreads(1)

    parser = argparse.ArgumentParser(description="Trains the model")
    parser.add_argument('--batchsize', help="The batch size to train with", type=int, default=128)
    parser.add_argument('--lr', help='learning rate', type=float, default=1.e-3)
    parser.add_argument('--start', help='model checkpoint to start from', type=str, default='')
    parser.add_argument('--backbone-parameters', help='checkpoint of parameters for the backbone', type=str, default=None)
    parser.add_argument('--freeze-backbone', help="Don't train the parameters of the backbone", action='store_true', default=False)
    parser.add_argument('--find-lr', help="Enable learning rate finder mode", action='store_true', default=False)
    parser.add_argument('--iterations', help='As if you were running this script multiple times using the best rotation snapshot', type=int, default=1)
    parser.add_argument('--epochs', help="Number of epochs", type=int, default=100)
    args = parser.parse_args()

    #net = models.NetworkWithPointHead(enable_point_head=False, enable_face_detector=True, enable_full_head_box=True)
    net = models.LocalAttentionNetwork(enable_face_detector=True, enable_full_head_box=True)
    args.input_size = net.input_resolution

    train_loader, test_loader = setup_datasets(args)

    if args.start:
        state_dict = torch.load(args.start)
        net.load_state_dict(state_dict)

    if args.backbone_parameters:
        net.convnet.load_state_dict(torch.load(args.backbone_parameters))
        print (f"Loaded backone parameters {args.backbone_parameters}!")

    net.cuda()

    C = train.Criterion
    criterions = [
        C('rot', train.QuatPoseLoss2(), 0.4, train=True, test=True),
        C('coord', train.CoordPoseLoss(), 0.4, train=True, test=True),
        C('box', train.BoxLoss(), 0.1, train=True, test=True),
        C('roi_head', train.BoxLoss('roi_head'), 0.01, train=True, test=True),
        C('attention', train.CoordAttentionMapLoss(), 0.4, train=True),
        C('hasface', train.HasFaceLoss(), 0.01, train=True, test=True),
        C('quatregularization', train.QuaternionNormalizationRegularization(), 1.e-6, train=True)
    ]

    save_callback = train.SaveBestCallback(net, 'rot', model_dir=join(dirname(__file__),'..','model_files'))
    callbacks = [ save_callback ]

    for iteration in range(args.iterations):
        optimizer, scheduler, n_epochs = create_optimizer(net, args)
        train.run_the_training(n_epochs,
                            optimizer,
                            net,
                            train_loader,
                            test_loader,
                            criterions,
                            callbacks,
                            scheduler = scheduler)
        # Reload snapshot with lowest rotation loss
        if iteration < args.iterations-1:
            net.load_state_dict(torch.load(save_callback.filename))

    torch.save(net.state_dict(), join(dirname(__file__),'..','model_files','last_'+type(net).__name__+'.ckpt'))


if __name__ == '__main__':
    main()