import enum
from typing import Literal, Mapping, Any
from os.path import dirname, join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

import trackertraincode.facemodel.keypoints68 as kpts68
import trackertraincode.neuralnets.torchquaternion as torchquaternion
import trackertraincode.neuralnets.torch6drotation as torch6drotation
from trackertraincode.neuralnets.modelcomponents import GaussianMixture
from trackertraincode.neuralnets.rotrepr import RotationRepr, QuatRepr, Mat33Repr

SimpleLossSwitch = Literal['l2','smooth_l1']
LOSS_OBJECT_MAP : Mapping[SimpleLossSwitch, Any] = {
    'l2' : torch.nn.MSELoss(reduction='none'),
    'l1' : torch.nn.L1Loss(reduction='none'),
    'smooth_l1' : torch.nn.SmoothL1Loss(reduction='none', beta=0.01)
}


def smooth_geodesic_distance(pred, target):
    smooth_zone = 1.*torch.pi/180. # One degree
    normed_delta = torchquaternion.geodesicdistance(pred, target)
    return F.smooth_l1_loss(normed_delta, torch.zeros_like(normed_delta), reduction='none', beta=smooth_zone)/torch.pi


SimpleRotLossSwitch = Literal['approx_distance','smooth_geodesic']
LOSS_FUNC_MAP_FOR_ROTATION = {
    'approx_distance' : torchquaternion.distance,
    'smooth_geodesic' : smooth_geodesic_distance
}


class QuatPoseLoss(object):
    def __init__(self, loss : SimpleRotLossSwitch, prefix=''):
        self._prefix = prefix
        self.loss_func = LOSS_FUNC_MAP_FOR_ROTATION[loss]
    def __call__(self, pred, sample):
        target : torch.Tensor = sample['pose']
        quat : QuatRepr = pred[self._prefix+'rot']
        return self.loss_func(quat.value, target)


class Rot6dReprLoss:
    def __call__(self, pred_batch, target_batch):
        pred : Mat33Repr = pred_batch['rot']
        target : torch.Tensor = target_batch['pose']
        target = torchquaternion.tomatrix(target)
        return torch6drotation.rotation_distance_loss(pred.value, target)


class Rot6dNormalizationSoftConstraint:
    def __call__(self, pred_batch, target_batch):
        pred = pred_batch['unnormalized_6drepr']
        return torch6drotation.orthonormality_loss(pred)


class PoseSizeLoss(object):
    def __init__(self, loss : SimpleLossSwitch, prefix=''):
        self._prefix = prefix
        self.loss_obj = LOSS_OBJECT_MAP[loss]

    def __call__(self, pred, sample):
        target = sample['coord'][...,2]
        coord = pred[self._prefix+'coord'][...,2]
        loss = self.loss_obj(coord, target)
        return loss


class PoseXYLoss(object):
    def __init__(self, loss : SimpleLossSwitch, prefix=''):
        self._prefix = prefix
        self.loss_obj = LOSS_OBJECT_MAP[loss]

    def __call__(self, pred, sample):
        target = sample['coord'][...,:2]
        coord = pred[self._prefix+'coord'][...,:2]
        loss = self.loss_obj(coord, target).mean(dim=-1)
        return loss


class ShapeParameterLoss(object):
    def eval_on_params(self, pred, target):
        lossvals = F.mse_loss(pred, target, reduction='none').mean(dim=-1)
        return lossvals

    def __call__(self, pred, sample):
        return self.eval_on_params(pred['shapeparam'], sample['shapeparam'])


class ShapePlausibilityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        with h5py.File(join(dirname(__file__),'../facemodel/shapeparams_gmm.h5'), 'r') as f:
            self.gmm = GaussianMixture.from_hdf5(f)
        self.register_buffer('fudge_factor', torch.as_tensor(0.001/self.gmm.n_components)) 

    def _eval(self, x):
        return -self.gmm(x)*self.fudge_factor

    def forward(self, pred, sample):
        mean_nll = self._eval(pred['shapeparam'].to(torch.float64)).to(torch.float32)
        assert len(mean_nll.shape) == 1
        return mean_nll


class QuaternionNormalizationSoftConstraint(object):
    def __init__(self, prefix=''):
        self._prefix = prefix
    def __call__(self, pred, sample):
        unnormalized = pred[self._prefix+'unnormalized_quat']
        assert len(unnormalized.shape)==2 and unnormalized.shape[-1]==4
        norm_loss = torch.norm(unnormalized, p=2, dim=1)
        norm_loss = torch.square(1.-norm_loss)
        return norm_loss


class Points3dLoss(nn.Module):
    def __init__(self, loss : SimpleLossSwitch, pointdimension : int = 3, chin_weight=1., eye_weights=0., prefix=''):
        super().__init__()
        self._prefix=prefix
        assert pointdimension in (2,3)
        self.loss_obj = LOSS_OBJECT_MAP[loss]
        self.pointdimension = pointdimension

        pointweights = np.ones((68,), dtype=np.float32)
        pointweights[kpts68.chin_left[:-1]] = chin_weight
        pointweights[kpts68.chin_right[1:]] = chin_weight
        pointweights[kpts68.eye_not_corners] = eye_weights

        self.register_buffer('pointweights', torch.from_numpy(pointweights))

    def _eval_on_points(self, pred : torch.Tensor, target : torch.Tensor):
        assert target.shape == pred.shape, f"Mismatch {target.shape} vs {pred.shape}"
        assert target.shape[2] == 3
        assert target.shape[1] == 68
        pointwise = self.loss_obj(pred[...,:self.pointdimension], target[...,:self.pointdimension]).sum(dim=-1)
        return torch.mean(pointwise*self.pointweights[None,:], dim=-1)

    def forward(self, pred, sample):
        return self._eval_on_points(pred[self._prefix+'pt3d_68'], sample['pt3d_68'])


class BoxLoss(object):
    def __init__(self, loss : SimpleLossSwitch,  dataname = 'roi'):
        self.dataname = dataname
        self.loss_obj = LOSS_OBJECT_MAP[loss]
    # Regression loss for bounding box prediction
    def __call__(self, pred, sample):
        # Only train this if the image shows a face
        target = sample[self.dataname]
        pred = pred[self.dataname]
        return self.loss_obj(pred, target).mean(dim=-1)


class HasFaceLoss(object):
    def __call__(self, pred, sample):
        target = sample['hasface']
        logits = pred['hasface_logits']
        return F.binary_cross_entropy_with_logits(logits, target, reduction='none').mean(dim=-1)


##########################################
## Localizer losses
##########################################


class LocalizerProbLoss(object):
    # Pytorch Manual:
    #   "This loss combines a Sigmoid layer and the BCELoss in one single class"
    bce = nn.BCEWithLogitsLoss(reduction='mean')
    def __call__(self, net, pred, sample):
        target = sample['hasface']
        pred = pred[:,0] # Grab the logit value for the "is face" score
        return self.bce(pred, target)


class LocalizerBoxLoss(object):
    # Regression loss for bounding box prediction
    def __call__(self, net, pred, sample):
        # Only train this if the image shows a face
        target = sample['roi']
        enable = sample['hasface']
        # FIXME: I adapted this for the pose estimator output
        # Old code: `pred = pred[:,1:]`
        roipred = pred['roi']
        err = F.smooth_l1_loss(roipred, target, reduction='none', beta=0.1)
        return torch.mean(enable[:,None]*err)