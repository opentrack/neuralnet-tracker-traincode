from typing import Dict, NamedTuple, Optional, Literal
import torch
import numpy as np
from os.path import join, dirname
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from trackertraincode.neuralnets.math import sqrclip0, inv_sqrclip0, smoothclip0, inv_smoothclip0
import trackertraincode.facemodel.keypoints68 as kpts68
import trackertraincode.neuralnets.torchquaternion as Q
from torch.distributions import Normal, MultivariateNormal, Laplace

make_positive = smoothclip0
inv_make_positive = inv_smoothclip0


class FeaturesAsUncorrelatedVariance(nn.Module):
    def __init__(self, num_in_features, num_out_features):
        super().__init__()
        self.num_out_features = num_out_features
        self.num_in_features = num_in_features
        self.lin = nn.Linear(num_in_features, num_out_features, bias=False)
        self.bn = nn.BatchNorm1d(num_out_features)
        self.bn.weight.data[...] *= 1.e-4
        self.bn.bias.data[...] = 1.
        self.eps = torch.tensor(1.e-2) # Prevent numerical problems due to too small variance
    
    def forward(self, x : Tensor):
        x = self.bn(self.lin(x))
        x = torch.square(x) + self.eps
        return x


class UncorrelatedVarianceParameter(nn.Module):
    '''
    Provides a trainable, input-independent scale parameter 
    which starts off as 1 and is guaranteed to be always positive.
    '''
    def __init__(self, num_out_features):
        super().__init__()
        self.num_out_features = num_out_features
        self.num_in_features = num_out_features
        self.hidden_scale = nn.Parameter(
            torch.ones((self.num_in_features)).requires_grad_(True), 
            requires_grad=True)
        self.eps = torch.tensor(1.e-2)
    
    def forward(self):
        return torch.square(self.hidden_scale) + self.eps


SimpleDistributionSwitch = Literal['gaussian','laplace']
DISTRIBUTION_CLASS_MAP = {
    'gaussian' : Normal,
    'laplace' : Laplace
}

class CoordPoseNLLLoss(nn.Module):
    def __init__(self, xy_weight : float, head_size_weight : float, distribution : SimpleDistributionSwitch = 'gaussian'):
        super().__init__()
        self.register_buffer('weights', torch.as_tensor([xy_weight/2.,xy_weight/2.,head_size_weight], dtype=torch.float32))
        self.distribution_class = DISTRIBUTION_CLASS_MAP[distribution]

    def __call__(self, preds, sample):
        target = sample['coord']
        pred = preds['coord']
        scale : Tensor = preds['coord_scales']
        return -self.distribution_class(pred, scale).log_prob(target).mul(self.weights[None,:]).mean(dim=-1)


class CorrelatedCoordPoseNLLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, preds, sample):
        target = sample['coord']
        pred = preds['coord']
        scale : Tensor = preds['coord_scales']
        return -MultivariateNormal(pred, scale_tril=scale, validate_args=False).log_prob(target)


class BoxNLLLoss(nn.Module):
    def __init__(self, dataname = 'roi', distribution : SimpleDistributionSwitch = 'gaussian'):
        super().__init__()
        self.dataname = dataname
        self.distribution_class = DISTRIBUTION_CLASS_MAP[distribution]

    def __call__(self, pred, sample):
        target : Tensor = sample[self.dataname]
        return -self.distribution_class(pred[self.dataname], pred[self.dataname+'_scales']).log_prob(target).mean(dim=-1)


class Points3dNLLLoss(nn.Module):
    def __init__(self, chin_weight, eye_weight,  pointdimension : int = 3, distribution : SimpleDistributionSwitch = 'gaussian'):
        super().__init__()
        self.distribution_class = DISTRIBUTION_CLASS_MAP[distribution]
        pointweights = np.ones((68,), dtype=np.float32)
        pointweights[kpts68.chin_left[:-1]] = chin_weight
        pointweights[kpts68.chin_right[1:]] = chin_weight
        pointweights[kpts68.eye_not_corners] = eye_weight
        self.register_buffer('pointweights', torch.from_numpy(pointweights))
        self.pointdimension = pointdimension
        
    def __call__(self, preds, sample):
        pred, scales, target = preds['pt3d_68'], preds['pt3d_68_scales'], sample['pt3d_68']
        loss = -self.pointweights[None,:,None]*self.distribution_class(pred[:,:,:self.pointdimension], scales[:,:,:self.pointdimension]).log_prob(target[:,:,:self.pointdimension])
        return loss.mean(dim=(-2,-1))


class ShapeParamsNLLLoss(nn.Module):
    def __init__(self, distribution : SimpleDistributionSwitch = 'gaussian'):
        super().__init__()
        self.distribution_class = DISTRIBUTION_CLASS_MAP[distribution]
    
    def __call__(self, preds, sample):
        pred, scales, target = preds['shapeparam'], preds['shapeparam_scales'], sample['shapeparam']
        loss = -self.distribution_class(pred, scales).log_prob(target)
        return loss.mean(dim=-1)



###########################################################
## Tangent space gaussian rotation distribution
###########################################################
## There is the little problem that this distribution
## is not normalized over SO3 ...

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


def _mult_cols_to_make_diag_positive(m: Tensor):
    if m.size(-1) == 3 and not m.requires_grad: # Fix for ONNX export
        f1 = torch.sign(m[...,0,0])
        f2 = torch.sign(m[...,1,1])
        f3 = torch.sign(m[...,2,2])
        m = m.clone()
        m[...,:,0] *= f1[...,None]
        m[...,:,1] *= f2[...,None]
        m[...,:,2] *= f3[...,None]
        return m
    else:
        return m*torch.sign(torch.diagonal(m, dim1=-2, dim2=-1))[...,None,:]


class FeaturesAsTriangularCovFactor(nn.Module):
    def __init__(self, num_in_features, dim):
        super().__init__()
        self.dim = dim
        self.num_matrix_params = (dim*(dim+1))//2
        self.num_features = self.num_matrix_params
        self.lin = nn.Linear(num_in_features, self.num_features, bias=False)
        self.bn = nn.BatchNorm1d(self.num_features)
        self.bn.weight.data[...] *= 1.e-4
        self.bn.bias.data[...] = 1.
        self.bn.bias.data[:self.dim] = 1.
        self.bn.bias.data[self.dim:] = 0.
        self.min_diag = 1.e-2

    def forward(self, x : Tensor):
        x = self.bn(self.lin(x))
        x_diags = x[...,:self.dim]
        x_offdiags = x[...,self.dim:]
        z = x.new_empty(tuple(x.shape[:-1])+(self.num_matrix_params,))
        z[...,:self.dim] = x_diags
        z[...,self.dim:] = x_offdiags
        m = _fill_triangular_matrix(self.dim, z)
        # Equivalent to cholesky(m @ m.mT)
        m = _mult_cols_to_make_diag_positive(m)
        m += self.min_diag*torch.eye(self.dim, device=x.device).expand(*x.shape[:-1],3,3)
        return m


class TangentSpaceRotationDistribution(object):
    def __init__(self, quat : Tensor, scale_tril : Optional[Tensor] = None, precision : Optional[Tensor] = None):
        self.dist = MultivariateNormal(quat.new_zeros(quat.shape[:-1]+(3,)), scale_tril=scale_tril, precision_matrix=precision, validate_args=False)
        self.quat = quat
    
    def log_prob(self, otherquat : Tensor):
        rotvec = Q.rotation_delta(self.quat, otherquat)
        return self.dist.log_prob(rotvec)


class QuatPoseNLLLoss(nn.Module):
    def __call__(self, preds, sample):
        target = sample['pose']
        quat = preds['pose']
        cov = preds['pose_scales_tril']
        return -TangentSpaceRotationDistribution(quat, cov).log_prob(target)



###########################################################
## Rotation Laplace Distribution
###########################################################

# Reimplemented from https://github.com/yd-yin/RotationLaplace/blob/master/rotation_laplace.py
# Grids file taken unchanged from that repo.

class SO3DistributionParams(NamedTuple):
    mode : Tensor # Where the peak of the distribution is
    cholesky_factor : Tensor # Cholesky decomposition of V S Vt


class RotationLaplaceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        grids = torch.from_numpy(np.load(join(dirname(__file__), 'rotation-laplace-grids3.npy')))
        self.register_buffer('grids', grids)


    def power_function(self, matrix_r : Tensor, cov_factor_tril : Tensor):
        '''sqrt(tr(S - At R)) =

        sqrt(tr(cov - cov R0_t R))
        '''

        # a = (LLt)^-1 = Lt^-1 L^-1
        # a_t = a_t
        # out = Lt^-1 L^-1 R
        # -> solve L Z1 = R
        # -> solve Lt out = Z1

        # m_z = torch.linalg.solve_triangular(cov_factor_tril, matrix_r, upper=False)
        # m_z = torch.linalg.solve_triangular(cov_factor_tril.transpose(-1,-2), m_z, upper=True)
        #m_z = torch.linalg.inv(torch.matmul(cov_factor_tril, cov_factor_tril.transpose(-1,-2)))
        m_z = torch.cholesky_inverse(cov_factor_tril)
        m_cov_diags = torch.diagonal(m_z, dim1=-2, dim2=-1)
        m_z = torch.matmul(m_z, matrix_r)
        trace_quantity = (m_cov_diags - torch.diagonal(m_z, dim1=-2, dim2=-1)).sum(-1)
        if trace_quantity.min() < -1.e-6:
            print (f"Warning: Rotation Laplace failure. Trace negative: {trace_quantity.min()}")
            print (f"cov factor = ", repr(cov_factor_tril.detach().cpu().numpy()))
        power = torch.sqrt(torch.clamp_min(trace_quantity, 1.e-8))
        return power


    def log_normalization(self, grids : Tensor, cov_factor_tril : Tensor):
        # Integral over rotations of exp(-P)/P
        # Numerically:  log (sum 1/N exp(-P_i)/P_i)
        #    = log [ 1/N sum exp(-P_i)/P_i ]

        #  log [ 1/N sum exp(-P_i -c + c)/P_i ] = 
        #  log [ 1/N sum exp(-P_i +c)*exp(-c)/P_i ] = 
        #  log [ 1/N *exp(-c)* sum exp(-P_i +c)/P_i ]

        grids : Tensor = grids[None,:,:,:]  # (1, N, 3, 3)
        cov_factor_tril = cov_factor_tril[:, None, :,:]  # (B, 1, 3, 3)
        N = grids.size(1)

        inv_log_weight = torch.log(torch.tensor(N,dtype=torch.float32))
        # Shape B x N
        powers = self.power_function(grids, cov_factor_tril)
        stability = torch.amin(powers, dim=-1)
        powers = powers.to(torch.float64)
        log_exp_sum = torch.log((torch.exp(-powers+stability[:,None])/powers).sum(dim=1)).to(torch.float32)
        logF = log_exp_sum - inv_log_weight - stability
        return logF


    def compute_nll(self, gt_quat : Tensor, dist_params : SO3DistributionParams):
        # The NLL is log(F(A)) + P + log(P)
        # where P  = sqrt(tr(S - At R))
        matrix_r = Q.tomatrix(gt_quat)

        # At = V S Vt R0t.
        # Therefore
        # A = R0 V S Vt
        # The choleksy factor is L, and LLt = V S Vt
        mr0 = Q.tomatrix(dist_params.mode)
        cov_factor = dist_params.cholesky_factor
        # Beware of correct transpose axis order!
        matrix_r = torch.matmul(mr0.transpose(-1,-2), matrix_r)
        power = self.power_function(matrix_r, cov_factor)
        logF = self.log_normalization(self.grids, cov_factor)
        nll = logF + power + torch.log(power)
        return nll

    def forward(self, preds, sample):
        target = sample['pose']

        dist_params = SO3DistributionParams(
            mode = preds['pose'], 
            cholesky_factor = preds['pose_scales_tril'])

        return self.compute_nll(target, dist_params)


###########################################################
##  Tests
###########################################################

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