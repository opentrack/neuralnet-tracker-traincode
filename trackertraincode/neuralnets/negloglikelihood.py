from typing import Dict, NamedTuple, Optional, Literal
import sys
import torch
import numpy as np
from os.path import join, dirname
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import robust_loss_pytorch.distribution

from trackertraincode.neuralnets.math import sqrclip0, inv_sqrclip0, smoothclip0, inv_smoothclip0
import trackertraincode.facemodel.keypoints68 as kpts68
import trackertraincode.neuralnets.torchquaternion as Q
import trackertraincode.neuralnets.torch6drotation as R6d
from trackertraincode.neuralnets.rotrepr import RotationRepr, QuatRepr, Mat33Repr
from torch.distributions import Normal, MultivariateNormal, Laplace

make_positive = smoothclip0
inv_make_positive = inv_smoothclip0


class Neck(nn.Module):
    def __init__(self, num_in_features, num_out_features):
        super().__init__()
        self.num_out_features = num_out_features
        self.num_in_features = num_in_features
        self.lin = nn.Linear(num_in_features, num_out_features, bias=True)
        self.lin.bias.data[...] = inv_make_positive(torch.ones((num_out_features)))

    def set_biases(self, x: Tensor):
        self.lin.bias.data[...] = x

    def forward(self, x: Tensor):
        x = self.lin(x)
        return x


class FeaturesAsDiagonalScale(nn.Module):
    def __init__(self, num_in_features, num_out_features, init_values=None):
        super().__init__()
        self.neck = Neck(num_in_features, num_out_features)
        if init_values is not None:
            self.neck.set_biases(inv_make_positive(init_values))
        self.eps = 1.0e-6  # Prevent numerical problems due to too small variance

    def forward(self, x: Tensor):
        x = self.neck(x)
        x = make_positive(x) + self.eps
        return x


class DiagonalScaleParameter(nn.Module):
    """
    Provides a trainable, input-independent scale parameter
    which starts off as 1 and is guaranteed to be always positive.
    """

    def __init__(self, num_out_features, init_values=None):
        super().__init__()
        vals = inv_make_positive(torch.ones((num_out_features + 1,)))
        if init_values is not None:
            vals[1:] = inv_make_positive(init_values)
        self.hidden_scale = nn.Parameter(vals.requires_grad_(True), requires_grad=True)
        self.eps = 1.0e-6

    def forward(self):
        return (
            make_positive(self.hidden_scale[:1]) * make_positive(self.hidden_scale[1:]) + self.eps
        )


class AdaptiveDistribution:
    def __init__(self, center, scale):
        self._dist = robust_loss_pytorch.distribution.Distribution()
        self._center, self._scale = torch.broadcast_tensors(center, scale)

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        nll = self._dist.nllfun(
            target - self._center, alpha=torch.full_like(self._center, 0.0), scale=self._scale
        )
        return -nll


SimpleDistributionSwitch = Literal["gaussian", "laplace", "adaptive"]
DISTRIBUTION_CLASS_MAP = {"gaussian": Normal, "laplace": Laplace, "adaptive": AdaptiveDistribution}


def _fill_triangular_matrix(dim: int, z: Tensor):
    """
    dim: Matrix dimension
    z:  Tensor with values to fill into the lower triangular part.
        First the diagonal amounting to `dim` values. Then offdiagonals.
    """

    if dim == 3:
        # Special case for our application because ONNX does not support tril_indices
        m = z[
            ...,
            (
                0,
                0,
                0,
                3,
                1,
                0,
                4,
                5,
                2,
            ),
        ].view(*z.shape[:-1], 3, 3)
        m = m * z.new_tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        return m
    else:
        # General case
        m = z.new_zeros(z.shape[:-1] + (dim, dim))
        idx = torch.tril_indices(dim, dim, -1, device=z.device)
        irow, icol = idx[0], idx[1]
        # Ellipsis for the batch dimensions is not supported by the script compiler.
        # The tracer does unfortunately also produce wrong results, messing up some indexing
        m[:, irow, icol] = z[..., dim:]
        i = torch.arange(dim, device=z.device)
        m[:, i, i] = z[:, :dim]
        return m


class FeaturesAsTriangularScale(nn.Module):
    def __init__(self, num_in_features, dim):
        super().__init__()
        self.dim = dim
        self.num_matrix_params = (dim * (dim + 1)) // 2
        self.neck = Neck(num_in_features, self.num_matrix_params)
        bias_init = inv_make_positive(torch.full((self.num_matrix_params,), fill_value=3.14))
        bias_init[self.dim :] = 0.0  # Offdiagonals
        self.neck.set_biases(bias_init)
        min_diag = torch.full((self.num_matrix_params,), 1.0e-6)
        min_diag[self.dim :] = 0.0  # Offdiagonals
        self.register_buffer("min_diag", min_diag)

    def forward(self, x: Tensor):
        x = self.neck(x)
        z = torch.cat([make_positive(x[..., : self.dim]), x[..., self.dim :]], dim=-1)
        z = z + self.min_diag
        return _fill_triangular_matrix(self.dim, z)


class TangentSpaceRotationDistribution(object):
    def __init__(
        self, quat: Tensor, scale_tril: Optional[Tensor] = None, precision: Optional[Tensor] = None
    ):
        self.dist = MultivariateNormal(
            quat.new_zeros(quat.shape[:-1] + (3,)),
            scale_tril=scale_tril,
            precision_matrix=precision,
            validate_args=not sys.flags.optimize,
        )
        self.quat = quat

    def log_prob(self, otherquat: Tensor):
        rotvec = Q.rotation_delta(self.quat, otherquat)
        return self.dist.log_prob(rotvec)


class QuatPoseNLLLoss(nn.Module):
    def __init__(
        self,
        distribution: SimpleDistributionSwitch = "gaussian",
    ):
        super().__init__()
        self.distribution_class = DISTRIBUTION_CLASS_MAP[distribution]

    def __call__(self, preds, sample):
        target = sample["pose"]
        rot: RotationRepr = preds["rot"]
        cov = preds["pose_scales"]
        rotangle = Q.geodesicdistance(rot.as_quat(), target)
        return -self.distribution_class(torch.zeros_like(rotangle), cov).log_prob(rotangle)


class CoordPoseNLLLoss(nn.Module):
    def __init__(
        self,
        distribution: SimpleDistributionSwitch = "gaussian",
        xy_weight: float = 1.0,
        sz_weight: float = 1.0,
    ):
        super().__init__()
        self.distribution_class = DISTRIBUTION_CLASS_MAP[distribution]
        # Added 0.5 to take the mean of both coordinates.
        self.register_buffer(
            "weights", torch.as_tensor([xy_weight * 0.5, xy_weight * 0.5, sz_weight])
        )
        self.weights: Tensor

    def __call__(self, preds, sample):
        target = sample["coord"]
        pred = preds["coord"]
        scale: Tensor = preds["coord_scales"]
        log_probs: Tensor = self.distribution_class(pred, scale).log_prob(target)
        return log_probs.mul(self.weights).sum(dim=-1).mul(-1.0)


class BoxNLLLoss(nn.Module):
    def __init__(self, dataname="roi", distribution: SimpleDistributionSwitch = "gaussian"):
        super().__init__()
        self.dataname = dataname
        self.distribution_class = DISTRIBUTION_CLASS_MAP[distribution]

    def __call__(self, pred, sample):
        target: Tensor = sample[self.dataname]
        return (
            -self.distribution_class(pred[self.dataname], pred[self.dataname + "_scales"])
            .log_prob(target)
            .mean(dim=-1)
        )


class Points3dNLLLoss(nn.Module):
    def __init__(
        self,
        chin_weight,
        eye_weight,
        pointdimension: int = 3,
        distribution: SimpleDistributionSwitch = "gaussian",
    ):
        super().__init__()
        self.distribution_class = DISTRIBUTION_CLASS_MAP[distribution]
        pointweights = np.ones((68,), dtype=np.float32)
        pointweights[kpts68.chin_left[:-1]] = chin_weight
        pointweights[kpts68.chin_right[1:]] = chin_weight
        pointweights[kpts68.eye_not_corners] = eye_weight
        self.register_buffer("pointweights", torch.from_numpy(pointweights))
        self.pointdimension = pointdimension

    def __call__(self, preds, sample):
        pred, scales, target = preds["pt3d_68"], preds["pt3d_68_scales"], sample["pt3d_68"]
        loss = -self.pointweights[None, :, None] * self.distribution_class(
            pred[:, :, : self.pointdimension], scales[:, :, : self.pointdimension]
        ).log_prob(target[:, :, : self.pointdimension])
        return loss.mean(dim=(-2, -1))


class ShapeParamsNLLLoss(nn.Module):
    def __init__(self, distribution: SimpleDistributionSwitch = "gaussian"):
        super().__init__()
        self.distribution_class = DISTRIBUTION_CLASS_MAP[distribution]

    def __call__(self, preds, sample):
        pred, scales, target = preds["shapeparam"], preds["shapeparam_scales"], sample["shapeparam"]
        loss = -self.distribution_class(pred, scales).log_prob(target)
        return loss.mean(dim=-1)
