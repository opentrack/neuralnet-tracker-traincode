#!/usr/bin/env python
# coding: utf-8

# Seems to run a bit faster than with default settings and less bugged
# See https://github.com/pytorch/pytorch/issues/67864
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from typing import List, NamedTuple, Optional
from os.path import join, dirname
import numpy as np
import cv2
import argparse
import functools

import torch.optim as optim
import torch
import trackertraincode.neuralnets.losses as losses
import trackertraincode.neuralnets.models as models
import trackertraincode.neuralnets.negloglikelihood as NLL
import trackertraincode.train as train
import trackertraincode.pipelines
import trackertraincode.datasets.batch
from trackertraincode.pipelines import Tag


class MyArgs(argparse.Namespace):
    backbone : str
    batchsize  : int
    lr : float
    find_lr : bool
    epochs : int
    ds : str
    plotting : bool
    plot_save_filename : Optional[str]
    swa : bool
    outdir : str
    ds_weight_are_sampling_frequencies : bool
    with_pointhead : bool
    with_nll_loss : bool
    rotation_aug_angle : float
    with_image_aug : bool
    with_blurpool : bool
    export_onnx : bool
    input_size : int
    roi_override : str
    with_roi_train : bool


def parse_dataset_definition(arg : str):
    '''Parses CLI dataset specifications

    Of the form <name1>[:<weight1>]+<name2>[:<weight2>]+...
    '''

    dsmap = {
        '300wlp' :  trackertraincode.pipelines.Id._300WLP,
        'synface' : trackertraincode.pipelines.Id.SYNFACE,
        'aflw2k' : trackertraincode.pipelines.Id.AFLW2k3d,
        'biwi' : trackertraincode.pipelines.Id.BIWI,
        'wider' : trackertraincode.pipelines.Id.WIDER,
        'repro_300_wlp' : trackertraincode.pipelines.Id.REPO_300WLP,
        'repro_300_wlp_woextra' : trackertraincode.pipelines.Id.REPO_300WLP_WO_EXTRA,
        'wflw_lp' : trackertraincode.pipelines.Id.WFLW_LP,
        'lapa_megaface_lp' : trackertraincode.pipelines.Id.LAPA_MEGAFACE_LP
    }

    splitted = arg.split('+')

    # Find dataset specification which has weights in them
    # and add them to a dict.
    it = (tuple(s.split(':')) for s in splitted if ':' in s)
    dataset_weights = {
        dsmap[k]:float(v) for k,v in it }

    # Then consider all datasets listed
    dsids = [ dsmap[s.split(':')[0]] for s in splitted ]
    dsids = list(frozenset(dsids))

    return dsids, dataset_weights


def setup_datasets(args : MyArgs):
    dsids, dataset_weights = parse_dataset_definition(args.ds)

    train_loader, test_loader, ds_size = trackertraincode.pipelines.make_pose_estimation_loaders(
        inputsize = args.input_size, 
        batchsize = args.batchsize,
        datasets = dsids,
        dataset_weights = dataset_weights,
        use_weights_as_sampling_frequency=args.ds_weight_are_sampling_frequencies,
        enable_image_aug=args.with_image_aug,
        rotation_aug_angle=args.rotation_aug_angle,
        roi_override=args.roi_override)

    return train_loader, test_loader, ds_size


def parameter_groups_with_decaying_learning_rate(parameter_groups, slow_lr, fast_lr):
    # Note: Parameters are enumerated in the order from the input to the output layers
    factor = (fast_lr/slow_lr)**(1./(len(parameter_groups)-1))
    return [
        { 'params' : p, 'lr' : slow_lr*factor**i } for i,p in enumerate(parameter_groups)
    ]


def create_optimizer(net, args : MyArgs):
    to_optimize = net.parameters()
    #optimizer = optim.AdamW(to_optimize, lr=args.lr, weight_decay=1.e-3)
    optimizer = optim.Adam(to_optimize, lr=args.lr)
    if args.find_lr:
        print ("LR finding mode!")
        n_epochs = args.epochs
        lr_max = 1.e-1
        base = np.power(lr_max/args.lr, 1./n_epochs)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda e: base**e, verbose=True)
    else:
        n_epochs = args.epochs
        scheduler = train.LinearUpThenSteps(optimizer, max(1,n_epochs//(2*10)), 0.1, [n_epochs//2])

    return optimizer, scheduler, n_epochs


def swa_update_callback(state : train.State, net : torch.nn.Module, swa_model : optim.swa_utils.AveragedModel, start_epoch : int):
    if state.epoch > start_epoch:
        swa_model.update_parameters(net)

        
class SaveBestSpec(NamedTuple):
    weights : List[float]
    names : List[str]

        
def setup_losses(args : MyArgs, net):
    C = train.Criterion

    chasface = [ C('hasface', losses.HasFaceLoss(), 0.01) ]
    cregularize = [
        C('quatregularization1', losses.QuaternionNormalizationSoftConstraint(), 1.e-6),
    ]
    poselosses = []
    roilosses = []
    pointlosses = []
    pointlosses25d = []
    shapeparamloss = []

    if args.with_nll_loss:
        nllw = 0.01
        poselosses += [ 
            C('nllrot', NLL.QuatPoseNLLLoss().to('cuda'), 0.3*nllw), 
            C('nllcoord', NLL.CorrelatedCoordPoseNLLLoss().cuda(), 0.3*nllw)
        ]
        if args.with_roi_train:
            roilosses += [
                C('nllbox', NLL.BoxNLLLoss(distribution='gaussian'), 0.01*nllw)
            ]
        if args.with_pointhead:
            pointlosses += [ 
                C('nllpoints3d', NLL.Points3dNLLLoss(chin_weight=0.8, eye_weight=0., distribution='laplace').cuda(), 0.7*nllw) 
            ]
            pointlosses25d = [ 
                C('nllpoints3d', NLL.Points3dNLLLoss(chin_weight=0.8, eye_weight=0., pointdimension=2, distribution='laplace').cuda(), 0.7*nllw) 
            ]
            shapeparamloss += [
                C('nllshape', NLL.ShapeParamsNLLLoss(distribution='gaussian'), 0.05*nllw)
            ]
    if 1:
        poselosses += [
            C('rot', losses.QuatPoseLoss('approx_distance'), 0.5),
            C('xy', losses.PoseXYLoss('l2'), 0.5*0.5),
            C('sz', losses.PoseSizeLoss('l2'), 0.5*0.5)
        ]
        if args.with_roi_train:
            roilosses += [ 
                C('box', losses.BoxLoss('l2'), 0.01) 
            ]
        if args.with_pointhead:
            pointlosses += [ 
                C('points3d', losses.Points3dLoss('l1', chin_weight=0.8, eye_weights=0.).cuda(), 0.7)
            ]
            pointlosses25d += [ C('points3d', losses.Points3dLoss('l1', pointdimension=2, chin_weight=0.8, eye_weights=0.).cuda(), 0.7) ]
            shapeparamloss += [
                C('shp_l2', losses.ShapeParameterLoss(), 0.05),
            ]

    train_criterions = { 
        Tag.ONLY_POSE : train.MultiTaskLoss(poselosses + cregularize),
        Tag.POSE_WITH_LANDMARKS :           train.MultiTaskLoss(poselosses + cregularize + pointlosses + shapeparamloss + roilosses),
        Tag.POSE_WITH_LANDMARKS_3D_AND_2D : train.MultiTaskLoss(poselosses + cregularize + pointlosses + shapeparamloss + roilosses),
        Tag.ONLY_LANDMARKS     : train.MultiTaskLoss(pointlosses    + cregularize),
        Tag.ONLY_LANDMARKS_25D : train.MultiTaskLoss(pointlosses25d + cregularize),
        Tag.FACE_DETECTION : train.MultiTaskLoss(chasface + roilosses),
    }
    test_criterions = {
        Tag.POSE_WITH_LANDMARKS : train.DefaultTestFunc(poselosses + pointlosses + roilosses + shapeparamloss),
        Tag.FACE_DETECTION : train.DefaultTestFunc(chasface + roilosses),
    }

    savebest = SaveBestSpec(
        [l.w for l in poselosses],
        [ l.name for l in poselosses])

    return train_criterions, test_criterions, savebest


def create_net(args : MyArgs):
    return models.NetworkWithPointHead(
        enable_point_head=args.with_pointhead,
        enable_face_detector=False,
        config=args.backbone,
        enable_uncertainty=args.with_nll_loss,
        backbone_args={'use_blurpool' : args.with_blurpool}
    )


def main():
    np.seterr(all='raise')
    cv2.setNumThreads(1)

    parser = argparse.ArgumentParser(description="Trains the model")
    parser.add_argument('--backbone', help='Which backbone the net uses', default='mobilenetv1')
    parser.add_argument('--batchsize', help="The batch size to train with", type=int, default=64)
    parser.add_argument('--lr', help='learning rate', type=float, default=1.e-3)
    parser.add_argument('--find-lr', help="Enable learning rate finder mode", action='store_true', default=False)
    parser.add_argument('--epochs', help="Number of epochs", type=int, default=200)
    parser.add_argument('--ds', help='Which datasets to train on. See code.', type=str, default='300wlp')
    parser.add_argument('--no-plotting', help='Disable plotting of losses', action='store_false', default=True, dest='plotting')
    parser.add_argument('--save-plot', help='Filename to enable saving the train history as plot', default=None, type=str, dest='plot_save_filename')
    parser.add_argument('--with-swa', help='Enable stochastic weight averaging', action='store_true', default=False, dest='swa')
    parser.add_argument('--outdir', help="Output sub-directory", type=str, default=join(dirname(__file__),'..','model_files'))
    parser.add_argument('--ds-weighting', help="Sample dataset with equal probability and use weights for scaling their losses", 
                        action="store_false", default=True, dest="ds_weight_are_sampling_frequencies")
    parser.add_argument('--no-pointhead', help="Disable landmark prediction", action="store_false", default=True,dest="with_pointhead" )
    parser.add_argument('--with-nll-loss', default=False, action='store_true')
    parser.add_argument('--raug', default=30, type=float, dest='rotation_aug_angle')
    parser.add_argument('--no-imgaug', default=True, action='store_false', dest='with_image_aug')
    parser.add_argument('--no-blurpool', default=True, action='store_false', dest='with_blurpool')
    parser.add_argument('--no-onnx', default=True, action='store_false', dest='export_onnx')
    parser.add_argument('--roi-override', default='extent_to_forehead', type=str, choices=['extent_to_forehead', 'original', 'landmarks'], dest='roi_override')
    parser.add_argument('--no-roi-train', default=True, action='store_false', dest='with_roi_train')
    args : MyArgs = parser.parse_args()

    net = create_net(args)
    args.input_size = net.input_resolution

    if args.swa:
        swa_model = optim.swa_utils.AveragedModel(net, device='cpu', use_buffers=True)

    train_loader, test_loader, _ = setup_datasets(args)

    net.cuda()

    train_criterions, test_criterions, savebest = setup_losses(
            args,
            net)

    optimizer, scheduler, n_epochs = create_optimizer(net, args)

    save_callback = train.SaveBestCallback(
        net, 
        loss_names= savebest.names, 
        model_dir=args.outdir,
        save_name_prefix='total',
        weights = savebest.weights) 

    callbacks = [ save_callback ]

    if args.swa:
        callbacks += [ functools.partial(swa_update_callback, net=net, swa_model=swa_model, start_epoch=n_epochs*2//3) ]

    train.run_the_training(
        n_epochs, optimizer, net, train_loader, test_loader,
        functools.partial(train.default_update_fun, loss=train_criterions),
        test_criterions,
        callbacks,
        scheduler = scheduler,
        artificial_epoch_length=1024 if args.find_lr else 10*1024,
        plotting=args.plotting,
        plot_save_filename = join(args.outdir, args.plot_save_filename),
        close_plot_on_exit = args.plot_save_filename is not None)

    if args.swa:
        swa_filename = join(args.outdir,f'swa_{swa_model.module.name}.ckpt')
        torch.save(swa_model.module.state_dict(), swa_filename)
          

    last_save_filename = join(args.outdir,f'last_{net.name}.ckpt')
    torch.save(net.state_dict(), last_save_filename)

    if args.export_onnx:
        from scripts.export_model import convert_posemodel_onnx

        net.to('cpu')
        convert_posemodel_onnx(net, filename=last_save_filename)

        net.load_state_dict(torch.load(save_callback.filename))
        convert_posemodel_onnx(net, filename=save_callback.filename)

        if args.swa:
            convert_posemodel_onnx(swa_model.module.to('cpu'), filename=swa_filename)


if __name__ == '__main__':
    main()
