import enum
from os.path import dirname, join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import trackertraincode.facemodel.keypoints68 as kpts68
from trackertraincode.neuralnets.gmm import unpickle_scipy_gmm
import trackertraincode.neuralnets.torchquaternion as torchquaternion


def rwing(x, omega=0.1, epsilon=0.02, r = 0.01, reduction=None):
    x_r = torch.clamp(x - r, 0., omega-r)
    logterm = omega * torch.log(1 + x_r / epsilon)
    linearterm = torch.maximum(x - omega, x.new_zeros(()))
    loss = logterm + linearterm
    if reduction=='mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    return loss


def _element_wise_rotation_loss(pred, target):
    assert (len(target.shape)==2 and target.shape[-1]==4), f"target tensor has invalid shape {target.shape}"
    # TODO Is this better? torchquaternion.geodesicdistance(pred, target)/3.14
    return torchquaternion.distance(pred, target)


class QuatPoseLoss2(object):
    def __init__(self, prefix=''):
        self._prefix = prefix
    def __call__(self, pred, sample):
        target = sample['pose']
        quat = pred[self._prefix+'pose']
        loss = _element_wise_rotation_loss(quat, target)
        return torch.mean(loss)


class PoseXYLoss(object):
    def __init__(self, prefix=''):
        self._prefix = prefix
    def __call__(self, pred, sample):
        target = sample['coord'][...,2]
        coord = pred[self._prefix+'coord'][...,2]
        return F.mse_loss(coord, target, reduction='mean')


class PoseSizeLoss(object):
    def __init__(self, prefix=''):
        self._prefix = prefix
    def __call__(self, pred, sample):
        target = sample['coord'][...,:2]
        coord = pred[self._prefix+'coord'][...,:2]
        return F.mse_loss(coord, target, reduction='mean')


class ShapeParameterLoss(object):
    def eval_on_params(self, pred, target):
        lossvals = F.mse_loss(pred, target, reduction='none')
        return torch.mean(lossvals)

    def __call__(self, pred, sample):
        return self.eval_on_params(pred['shapeparam'], sample['shapeparam'])


class ShapePlausibilityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._gmm = unpickle_scipy_gmm(join(dirname(__file__),'../facemodel/shapeparams_gmm.pkl'))

    def __call__(self, pred, sample):
        mean_nll = -self._gmm.score_samples(pred['shapeparam']).mean()
        return mean_nll


class QuaternionNormalizationSoftConstraint(object):
    def __init__(self, prefix=''):
        self._prefix = prefix
    def __call__(self, pred, sample):
        unnormalized = pred[self._prefix+'unnormalized_quat']
        assert len(unnormalized.shape)==2 and unnormalized.shape[-1]==4
        norm_loss = torch.norm(unnormalized, p=2, dim=1)
        norm_loss = torch.square(1.-norm_loss)
        return torch.mean(norm_loss)


class ShapeRegularization(object):
    def __call__(self, pred, sample):
        params = pred['shapeparam']
        return torch.mean(torch.square(params))


class Points3dLoss(nn.Module):
    class DistanceFunction(enum.Enum):
        RWING = 1
        MSE = 2
        SMOOTH_L1 = 3
        L1 = 4

    def __init__(self, lossfunc=DistanceFunction.RWING, pointdimension : int = 3, chin_weight=1., eye_weights=0., prefix=''):
        super().__init__()
        self._prefix=prefix
        assert pointdimension in (2,3)
        self.lossfunc = {
            self.DistanceFunction.RWING : lambda pred,target: rwing(torch.norm(pred-target, dim=-1), reduction='none'),
            self.DistanceFunction.MSE : lambda pred, target: F.mse_loss(pred, target, reduction='none').sum(dim=-1),
            self.DistanceFunction.SMOOTH_L1 : lambda pred, target: F.smooth_l1_loss(torch.norm(pred-target, dim=-1), pred.new_zeros(()), reduction='none', beta=0.01),
            self.DistanceFunction.L1 : lambda pred, target: torch.norm(pred-target, dim=-1)
        }[lossfunc]
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
        unreduced = self.lossfunc(pred[...,:self.pointdimension], target[...,:self.pointdimension])
        return torch.mean(unreduced*self.pointweights[None,:])

    def forward(self, pred, sample):
        return self._eval_on_points(pred[self._prefix+'pt3d_68'], sample['pt3d_68'])


class BoxLoss(object):
    def __init__(self, dataname = 'roi'):
        self.dataname = dataname
    # Regression loss for bounding box prediction
    def __call__(self, pred, sample):
        # Only train this if the image shows a face
        target = sample[self.dataname]
        pred = pred[self.dataname]
        return F.smooth_l1_loss(pred, target, reduction='mean', beta=0.1)


class HasFaceLoss(object):
    def __call__(self, pred, sample):
        target = sample['hasface']
        logits = pred['hasface_logits']
        return F.binary_cross_entropy_with_logits(logits, target, reduction='mean')


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