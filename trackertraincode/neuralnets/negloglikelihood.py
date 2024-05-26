import torch
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from trackertraincode.neuralnets.math import smoothclip0
import trackertraincode.facemodel.keypoints68 as kpts68
import trackertraincode.neuralnets.torchquaternion as Q
from torch.distributions import Normal, MultivariateNormal
from trackertraincode.neuralnets.math import inv_smoothclip0


class FeaturesAsUncorrelatedVariance(nn.Module):
    def __init__(self, num_features, initial_values = None):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(self.num_features+1)
        self.bn.weight.data.fill_(1.e-3)
        self.bn.bias.data[...] = inv_smoothclip0(torch.ones((num_features+1,)) if initial_values is None else torch.cat([torch.ones(1,),initial_values]))
        self.eps = torch.tensor(1.e-6) # Prevent numerical problems due to too small variance
    
    def forward(self, x : Tensor):
        x = self.bn(x) # Normalize so we don't run into vanishing grads
        x = smoothclip0(x[...,:1]) * smoothclip0(x[...,1:]) + self.eps
        return x


class UncorrelatedVarianceParameter(nn.Module):
    '''
    Provides a trainable, input-independent scale parameter 
    which starts off as 1 and is guaranteed to be always positive.
    '''
    def __init__(self, num_features, initial_values = None):
        super().__init__()
        initial_values = inv_smoothclip0(torch.ones((num_features+1,)) if initial_values is None else torch.cat([torch.ones(1,),initial_values]))
        self.hidden_scale = nn.Parameter(initial_values.requires_grad_(True), requires_grad=True)
        self.eps = torch.tensor(1.e-6)
    
    def forward(self):
        return smoothclip0(self.hidden_scale[:1]) * smoothclip0(self.hidden_scale[1:]) + self.eps



@torch.jit.script
def _fill_triangular_matrix(dim : int, z : Tensor):
    '''
    dim: Matrix dimension
    z:  Tensor with values to fill into the lower triangular part.
        First the diagonal amounting to `dim` values. Then offdiagonals.
    '''
    m = z.new_zeros(z.shape[:-1]+(dim,dim))
    if dim == 3:
        # Special case for our application because ONNX does not support tril_indices
        m[:,0,0] = z[:,0]
        m[:,1,1] = z[:,1]
        m[:,2,2] = z[:,2]
        m[:,1,0] = z[:,3]
        m[:,2,0] = z[:,4]
        m[:,2,1] = z[:,5]
        return m
    else:
        # General case
        idx = torch.tril_indices(dim, dim, -1, device=z.device)
        irow, icol = idx[0], idx[1]
        # Ellipsis for the batch dimensions is not supported by the script compiler.
        # The tracer does unfortunately also produce wrong results, messing up some indexing
        m[:,irow,icol] = z[...,dim:] 
        i = torch.arange(dim, device=z.device)
        m[:,i,i] = z[:,:dim]
        return m

class FeaturesAsTriangularCovFactor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.num_matrix_params = (dim*(dim+1))//2
        self.num_features = self.num_matrix_params+1
        self.bn = nn.BatchNorm1d(self.num_features)
        self.bn.weight.data.fill_(1.e-3)
        self.bn.bias.data.fill_(0.) 
        self.bn.bias.data[0] = inv_smoothclip0(0.5)
        self.bn.bias.data[1:dim+1] = inv_smoothclip0(1.)
        self.eps = torch.tensor(1.e-6)

    def forward(self, x : Tensor):
        x = self.bn(x) # Normalize so we don't run into vanishing grads
        x_scaling = smoothclip0(x[...,:1])
        x_diags = smoothclip0(x[...,1:1+self.dim])
        x_offdiags = x[...,1+self.dim:]
        z = x.new_empty(tuple(x.shape[:-1])+(self.num_matrix_params,))
        z[...,:self.dim] = x_scaling * x_diags + self.eps # Diagonals must be positive
        z[...,self.dim:] = x_scaling * x_offdiags
        return _fill_triangular_matrix(self.dim, z)


class TangentSpaceRotationDistribution(object):
    def __init__(self, quat : Tensor, scale_tril : Tensor):
        self.dist = MultivariateNormal(quat.new_zeros(quat.shape[:-1]+(3,)), scale_tril=scale_tril)
        self.quat = quat
    
    def log_prob(self, otherquat : Tensor):
        rotvec = Q.rotation_delta(self.quat, otherquat)
        return self.dist.log_prob(rotvec)
    
    def rsample(self, size):
        rotvec = self.dist.rsample(size)
        rotq = Q.from_rotvec(rotvec)
        return Q.mult(self.quat.expand_as(rotq), rotq)

    def sample(self, size):
        return self.rsample(size)


class QuatPoseNLLLoss(object):
    def __call__(self, preds, sample):
        target = sample['pose']
        quat = preds['pose']
        cov = preds['pose_scales_tril']
        # Note: Scaling by 0.25 to bring it in line with the other losses
        return -0.25*TangentSpaceRotationDistribution(quat, cov).log_prob(target).mean()


class CoordPoseNLLLoss(object):
    def __call__(self, preds, sample):
        target = sample['coord']
        pred = preds['coord']
        scale = preds['coord_scales']
        return -Normal(pred, scale).log_prob(target).mean()


class BoxNLLLoss(nn.Module):
    def __init__(self, dataname = 'roi'):
        super().__init__()
        self.dataname = dataname

    def __call__(self, pred, sample):
        target : Tensor = sample[self.dataname]
        return -Normal(pred[self.dataname], pred[self.dataname+'_scales']).log_prob(target).mean()


class Points3dNLLLoss(nn.Module):
    def __init__(self, chin_weight, eye_weight):
        super().__init__()
        pointweights = np.ones((68,), dtype=np.float32)
        pointweights[kpts68.chin_left[:-1]] = chin_weight
        pointweights[kpts68.chin_right[1:]] = chin_weight
        pointweights[kpts68.eye_not_corners] = eye_weight
        self.register_buffer('pointweights', torch.from_numpy(pointweights))
        
    def __call__(self, preds, sample):
        pred, scales, target = preds['pt3d_68'], preds['pt3d_68_scales'], sample['pt3d_68']
        loss = -self.pointweights[None,:,None]*Normal(pred, scales).log_prob(target)
        return loss.mean()

# TODO: proper tests
def test_tangent_space_rotation_distribution():
    B = 5
    S = 7
    q = torch.rand((B, 4), requires_grad=True)
    cov_features = torch.rand((B, 6), requires_grad=True)
    r = torch.rand((B, 4))
    cov_converter = FeaturesAsTriangularCovFactor(3)
    dist = TangentSpaceRotationDistribution(q, cov_converter(cov_features))
    dist.rsample((S,))
    val = dist.log_prob(r).sum()
    val.backward()
    assert cov_converter.scales.grad is not None
    assert q.grad is not None
    assert cov_features.grad is not None


def test_feature_to_variance_mapping():
    B = 5
    N = 7
    q = torch.rand((B, N), requires_grad=True)
    m = FeaturesAsUncorrelatedVariance(N)
    v = m(q)
    val = v.sum()
    val.backward()
    assert m.scales.grad is not None
    assert q.grad is not None

if __name__ == '__main__':
    with torch.autograd.set_detect_anomaly(True):
        test_tangent_space_rotation_distribution()
        test_feature_to_variance_mapping()