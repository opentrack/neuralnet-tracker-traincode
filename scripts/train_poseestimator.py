#!/usr/bin/env python
# coding: utf-8

# Seems to run a bit faster than with default settings and less bugged
# See https://github.com/pytorch/pytorch/issues/67864
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from os.path import join, dirname
import numpy as np
import cv2
import argparse
import functools
from typing import List, NamedTuple

import torch.optim as optim
import torch
import trackertraincode.neuralnets.losses as losses
import trackertraincode.neuralnets.models as models
import trackertraincode.neuralnets.negloglikelihood as NLL
import trackertraincode.train as train
from scripts.export_model import convert_posemodel_onnx
import trackertraincode.pipelines
import trackertraincode.datasets.batch
from trackertraincode.pipelines import Tag

def setup_datasets(args):
    dsspec : List[str] = args.ds
    dsmap = {
        '300wlp' :  trackertraincode.pipelines.Id._300WLP,
        'synface' : trackertraincode.pipelines.Id.SYNFACE,
        'aflw2k' : trackertraincode.pipelines.Id.AFLW2k3d,
        'biwi' : trackertraincode.pipelines.Id.BIWI,
        'wider' : trackertraincode.pipelines.Id.WIDER,
        'repro_300_wlp' : trackertraincode.pipelines.Id.REPO_300WLP,
        'wflw_lp' : trackertraincode.pipelines.Id.WFLW_LP,
        'lapa_megaface_lp' : trackertraincode.pipelines.Id.LAPA_MEGAFACE_LP
    }
    dsids = []
    for k, v in dsmap.items():
        if k in dsspec:
            dsids += [ v ]

    dsids = list(frozenset(dsids))

    train_loader, test_loader, ds_size = trackertraincode.pipelines.make_pose_estimation_loaders(
        inputsize = args.input_size, 
        batchsize = args.batchsize,
        datasets = dsids,
        auglevel = args.auglevel)

    return train_loader, test_loader, ds_size


def parameter_groups_with_decaying_learning_rate(parameter_groups, slow_lr, fast_lr):
    # Note: Parameters are enumerated in the order from the input to the output layers
    factor = (fast_lr/slow_lr)**(1./(len(parameter_groups)-1))
    return [
        { 'params' : p, 'lr' : slow_lr*factor**i } for i,p in enumerate(parameter_groups)
    ]


def create_optimizer(net, args):
    if args.freeze_backbone:
        to_optimize = list(frozenset(net.parameters()) - frozenset(net.convnet.parameters()))
        for p in net.convnet.parameters():
            p.requires_grad = False
        lr_parameter = args.lr
        min_lr = 0.01*args.lr
        net.prepare_finetune()
        print ("Backbone frozen!")
    elif args.finetune:
        parameter_groups = net.prepare_finetune()
        to_optimize = parameter_groups_with_decaying_learning_rate(parameter_groups, 0.001*args.lr, args.lr)
        to_optimize = [*reversed(to_optimize)] # So the head appears first, which is what is shown with scheduler.get_lr().
        lr_parameter = [d['lr'] for d in to_optimize]
        min_lr = [0.01*lr for lr in lr_parameter]
        print ("Finetuning!")
    else:
        to_optimize = net.parameters()
        lr_parameter = args.lr
        min_lr = 0.01*args.lr
    if args.sgd:
        optimizer = optim.SGD(to_optimize, lr=args.lr) # weight_decay=1.e-4)
    else:
        #optimizer = optim.AdamW(to_optimize, lr=args.lr, weight_decay=1.e-3)
        optimizer = optim.Adam(to_optimize, lr=args.lr)
    if args.find_lr:
        print ("LR finding mode!")
        n_epochs = args.epochs
        lr_max = 1.e-1
        # lr_max = lr * b**n -> ln(lr_max/lr) = n*ln(b) -> 
        base = np.power(lr_max/args.lr, 1./n_epochs)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda e: base**e, verbose=True)
    else:
        n_epochs = args.epochs
        if args.sgd:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        else:
            #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [n_epochs//2])
            #scheduler = train.TriangularSchedule(optimizer, min_lr, lr_parameter, args.epochs)
            scheduler = train.LinearUpThenSteps(optimizer, 5, 0.1, [n_epochs//2])

    return optimizer, scheduler, n_epochs


def swa_update_callback(state : train.State, net : torch.nn.Module, swa_model : optim.swa_utils.AveragedModel, start_epoch : int):
    if state.epoch > start_epoch:
        swa_model.update_parameters(net)

        
class SaveBestSpec(NamedTuple):
    weights : List[float]
    names : List[str]

        
def setup_losses(args, net):
    C = train.Criterion

    chasface = [ C('hasface', losses.HasFaceLoss(), 0.01) ]
    cregularize = [
        C('quatregularization', losses.QuaternionNormalizationSoftConstraint(), 1.e-6),
    ]
    poselosses = []
    roilosses = [ C('box', losses.BoxLoss(), 0.01) ]
    pointlosses = []
    pointlosses25d = []
    if 1:
        nllw = 0.01
        pointlosses += [ C('nllpoints3d', NLL.Points3dNLLLoss(chin_weight=0.8, eye_weight=0.).cuda(), 0.5*nllw) ]
        poselosses += [ 
            C('nllrot', NLL.QuatPoseNLLLoss(), 0.5*nllw), 
            C('nllcoord', NLL.CoordPoseNLLLoss(), 0.5*nllw) ]
    if 1:
        poselosses += [
            C('rot', losses.QuatPoseLoss2(), 0.5),
            C('xy', losses.PoseXYLoss(), 0.5*0.2),
            C('sz', losses.PoseSizeLoss(), 0.5*0.8)
        ]
        pointlosses += [ 
            C('points3d', losses.Points3dLoss(losses.Points3dLoss.DistanceFunction.MSE, chin_weight=0.8, eye_weights=0.).cuda(), 0.5)
        ]
        pointlosses25d = [ C('points3d', losses.Points3dLoss(losses.Points3dLoss.DistanceFunction.MSE, pointdimension=2, chin_weight=0.8, eye_weights=0.).cuda(), 0.5) ]
        shapeparamloss = [
            C('shp_l2', losses.ShapeParameterLoss(), 0.1),
            C('nll_shp_gmm', losses.ShapePlausibilityLoss().cuda(), 1.e-3),
        ]

    train_criterions = { 
        Tag.POSE_WITH_LANDMARKS : train.MultiTaskLoss(poselosses + cregularize + pointlosses + shapeparamloss + roilosses),
        Tag.ONLY_POSE : train.MultiTaskLoss(poselosses + cregularize),
        Tag.POSE_WITH_LANDMARKS_3D_AND_2D : train.MultiTaskLoss(poselosses + cregularize + pointlosses + shapeparamloss + roilosses),
        Tag.ONLY_LANDMARKS : train.MultiTaskLoss(pointlosses),
        Tag.ONLY_LANDMARKS_25D : train.MultiTaskLoss(pointlosses25d),
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


def create_net(args):
    return models.NetworkWithPointHead(
        enable_point_head=True,
        enable_face_detector=False,
        config=args.backbone,
        enable_uncertainty=True
    )
    

def main():
    np.seterr(all='raise')
    cv2.setNumThreads(1)

    parser = argparse.ArgumentParser(description="Trains the model")
    parser.add_argument('--backbone', help='Which backbone the net uses', default='mobilenetv1')
    parser.add_argument('--batchsize', help="The batch size to train with", type=int, default=64)
    parser.add_argument('--lr', help='learning rate', type=float, default=1.e-3)
    parser.add_argument('--start', help='model checkpoint to start from', type=str, default='')
    parser.add_argument('--backbone-parameters', help='checkpoint of parameters for the backbone', type=str, default=None)
    parser.add_argument('--freeze-backbone', help="Don't train the parameters of the backbone", action='store_true', default=False)
    parser.add_argument('--finetune', help='Finetune - slow training of backbone with frozen normalization', action='store_true', default=False)
    parser.add_argument('--find-lr', help="Enable learning rate finder mode", action='store_true', default=False)
    parser.add_argument('--epochs', help="Number of epochs", type=int, default=200)
    parser.add_argument('--ds', help='Which datasets to train on. See code.', type=str, default='300wlp')
    parser.add_argument('--no-plotting', help='Disable plotting of losses', action='store_false', default=True, dest='plotting')
    parser.add_argument('--save-plot', help='Filename to enable saving the train history as plot', default=None, type=str, dest='plot_save_filename')
    parser.add_argument('--sgd', help='train with sgd and cosine lr decay', action='store_true', default=False)
    parser.add_argument('--auglevel', help='augmentation level (1,2 or 3)', type=int, default=1)
    parser.add_argument('--with-swa', help='Enable stochastic weight averaging', action='store_true', default=False, dest='swa')
    args = parser.parse_args()
    args.ds = args.ds.split('+')

    net = create_net(args)
    args.input_size = net.input_resolution

    if args.swa:
        swa_model = optim.swa_utils.AveragedModel(net, device='cpu', use_buffers=True)

    train_loader, test_loader, _ = setup_datasets(args)

    if args.start:
        state_dict = torch.load(args.start)
        net.load_partial(state_dict)

    if args.backbone_parameters:
        net.convnet.load_state_dict(torch.load(args.backbone_parameters))
        print (f"Loaded backone parameters {args.backbone_parameters}!")

    net.cuda()

    train_criterions, test_criterions, savebest = setup_losses(
            args,
            net)

    optimizer, scheduler, n_epochs = create_optimizer(net, args)

    save_callback = train.SaveBestCallback(
        net, 
        loss_names= savebest.names, 
        model_dir=join(dirname(__file__),'..','model_files'),
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
        plot_save_filename = args.plot_save_filename,
        close_plot_on_exit = args.plot_save_filename is not None)

    if args.swa:
        filename = join(dirname(__file__),'..','model_files',f'swa_{swa_model.module.name}.ckpt')
        torch.save(swa_model.module.state_dict(), filename)
        convert_posemodel_onnx(swa_model.module, filename=filename)

    last_save_filename = join(dirname(__file__),'..','model_files',f'last_{net.name}.ckpt')
    torch.save(net.state_dict(), last_save_filename)

    net.to('cpu')
    convert_posemodel_onnx(net, filename=last_save_filename)

    net.load_state_dict(torch.load(save_callback.filename))
    convert_posemodel_onnx(net, filename=save_callback.filename)


if __name__ == '__main__':
    main()
