from __future__ import annotations

from os.path import join, dirname
from typing import Tuple, Union, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init
from torch import Tensor

from kornia.filters.kernels import get_pascal_kernel_2d
from kornia.filters.blur_pool import _blur_pool_by_kernel2d
from trackertraincode.neuralnets.math import smoothclip0
from trackertraincode.neuralnets.modelcomponents import nn

from trackertraincode.facemodel.bfm import BFMModel
import trackertraincode.neuralnets.torchquaternion as torchquaternion


def set_bn_momentum(model : nn.Module, momentum):
    if isinstance(model, (nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d)):
        assert hasattr(model, 'momentum')
        model.momentum = momentum
    for module in model.children():
        set_bn_momentum(module, momentum)


class Residual(nn.Module):
    def __init__(self, *fns, skip_layer = None):
        super().__init__()
        self.fn = nn.Sequential(*fns)
        self.skip_layer = nn.Identity() if skip_layer is None else skip_layer

    def forward(self, x):
        return self.fn(x) + self.skip_layer(x)


def rigid_transformation_25d(r, t, s, points):
    """
        25d refers to 2.5 dimensional because this uses only a 2d offset.
        r : quaternion (normalized)
        t : 2d translation
        s : scaling
        points: 3d points in local space

        Return: 3d points in parent space. After the rotation, 
                the z-coordinates remain unchanged.
    """
    # Points: (..., 68, 3)
    # r: (..., 4)
    tmp_pos = torchquaternion.rotate(r[...,None,:], points)
    tmp_pos = tmp_pos * s[...,None,:]
    screen_pos = tmp_pos
    screen_pos = screen_pos.clone()
    screen_pos[...,:2] += t[...,None,:]
    return screen_pos


class DeformableHeadKeypoints(nn.Module):
    def __init__(self, num_shape=40, num_expr=10):
        super(DeformableHeadKeypoints, self).__init__()
        self.num_eigvecs = num_shape+num_expr
        self.num_shape = num_shape
        self.num_expr = num_expr
        full = BFMModel()
        keypts = torch.from_numpy(full.scaled_vertices[full.keypoints]).contiguous()
        keyeigvecs = torch.from_numpy(full.scaled_bases[:,full.keypoints,:]).contiguous()
        self.register_buffer('keypts', keypts)
        self.register_buffer('keyeigvecs', keyeigvecs)

    def _deformvector(self, shapeparams):
        # (..., num_eigvecs, 68, 3) x (... , B, num_eigvecs). -> Need for broadcasting and unsqueezing.
        local_keypts = self.keyeigvecs*shapeparams[...,None,None]
        local_keypts = torch.sum(local_keypts, dim=-3)
        return local_keypts

    def forward(self, shapeparams : Tensor):
        local_keypts = self._deformvector(shapeparams)
        local_keypts += self.keypts
        assert local_keypts.shape[-1]==3 and local_keypts.shape[-2]==68
        # Return in format (batch x points x dimensions)
        return local_keypts


class PosedDeformableHead(nn.Module):
    # TODO: Type annotation & interface specification
    def __init__(self, deformable_head):
        super().__init__()
        self.deformable_head = deformable_head

    def forward(self, coord, quats, params):
        local_keypts = self.deformable_head(params)
        points = rigid_transformation_25d(
            quats,
            coord[...,:2],
            coord[...,2:],
            local_keypts)
        assert points.shape[-1] == 3 and points.shape[:-2] == quats.shape[:-1]
        return points


# TODO: replace with kornia functions
class CenterOfMass(nn.Module):
    def __init__(self, half_size=1.):
        '''Computes spatial argmaxes from featuremaps

        Args:
            half_size: Half of the square domain size.
        '''
        super().__init__()
        self.half_size=nn.Parameter(torch.tensor(half_size, requires_grad=True, dtype=torch.float32))
        self.position_code : Optional[Tensor] = None # Filled in forward()

    def forward(self, x):
        B, H, W = x.shape
        p = torch.empty((2,H,W), dtype=torch.float32, device=x.device, requires_grad=False)
        # Positions include the endpoints.
        p[0,...] = torch.linspace(-1., 1., W, device=x.device)[None,:]
        p[1,...] = torch.linspace(-1., 1., H, device=x.device)[:,None]
        self.position_code = p
        mean = self.half_size * torch.sum(x[:,None,:,:] * p[None,...], dim=[2,3])
        return mean


# TODO: replace with kornia functions
class CenterOfMassAndStd(CenterOfMass):
    def __init__(self, eps = 1.e-4, half_size=1.):
        super().__init__(half_size)
        self._eps = eps

    def forward(self, x):
        mean = super().forward(x)
        diff = self.position_code[None,...] - mean[...,None,None]
        diff_squared = diff*diff # ONNX export does not know operator squared
        std  = torch.sqrt(torch.sum(x[:,None,:,:]*diff_squared, dim=[2,3])+self._eps)
        return mean, std


class LocalToGlobalCoordinateOffset(nn.Module):
    '''
    There is a scaling, x-y-offset and pitch-offset
    '''
    def __init__(self):
        super(LocalToGlobalCoordinateOffset, self).__init__()
        # Parameter meaning: pitch angle, forward-up-translation, scaling
        self.p = nn.Parameter(torch.tensor([0., 0., 0.,0.], dtype=torch.float32, requires_grad=True))

    def _compute_correction_quat(self):
        c = self.p[:1]
        return torch.cat([torch.sin(c), c.new_zeros((2,)), torch.cos(c) ])

    def _compute_correction_offset(self):
        return torch.cat([self.p.new_zeros((1,)), self.p[1:3]])

    def _compute_correction_scale(self):
        return smoothclip0(self.p[3])

    def forward(self, quats, coords):
        scale = coords[...,2:]
        head_center_screenspace = coords[...,:2]

        # pw = (xTh + Rh*(xTl + Rl*pl))
        # -> pw = (xTh + Rh*xTl + Rh*Rl*pl) = (xTh + Rh*xTl) + (Rh*Rl)*pl
        # Tw = xTh + Rh * xTl
        # Rw = Rh * Rl
        # 
        # Rh : image based rotation prediction
        # Rl : dataset specific prediction
        # xTh : image based translation prediction
        # xTl : dataset translation offset

        scale = scale * self._compute_correction_scale()

        z = self._compute_correction_quat()
        pred_quat = torchquaternion.mult(quats, z)

        pos_corr = self._compute_correction_offset()
        pos_corr = torchquaternion.rotate(quats, pos_corr)
        pos_corr = pos_corr[...,:2]
        pos_corr = pos_corr * scale
        screen_pos = pos_corr + head_center_screenspace
        pred_pos = torch.cat([
            screen_pos,
            scale
        ], axis=-1)
        return (pred_quat, pred_pos)


class BlurPool2D(nn.Module):
    r"""Compute blur (anti-aliasing) and downsample a given feature map.

    Copy paste from kornia with addition of channel count specification 
    because onnx has trouble figuring out the kernel size when the size
    depends on the input.
    """

    def __init__(self, kernel_size: tuple[int, int] | int, channels : int, stride: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.register_buffer('kernel', get_pascal_kernel_2d(kernel_size, norm=True))
        self.channels = channels

    def forward(self, input: Tensor) -> Tensor:
        return _blur_pool_by_kernel2d(input, self.kernel.repeat((self.channels, 1, 1, 1)), self.stride)


def freeze_norm_stats(m):
    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm, nn.InstanceNorm1d, nn.InstanceNorm2d)):
        #print (f"Freezing {str(m)}")
        m.eval()
        for p in m.parameters():
            p.requires_grad = False


def quaternion_from_features(z : Tensor):
    '''
    Returns:
        (quaternions, unnormalized quaternions)
    '''
    assert torchquaternion.iw == 3
    # The real component can be positive because -q is the same rotation as q.
    # Seems easier to learn like so.
    quats_unnormalized = torch.cat([
        z[...,torchquaternion.iijk],
        smoothclip0(z[...,torchquaternion.iw:])], dim=-1)
    quats = torchquaternion.normalized(quats_unnormalized)
    return quats, quats_unnormalized


# TODO: proper test
def _test_local_to_global_transform_offset():
    from scipy.spatial.transform import Rotation

    tr = LocalToGlobalCoordinateOffset()
    tr.p.data[:] = torch.tensor([ 1., 3., 4., 2. ])

    r = Rotation.from_rotvec([ 0., 0.5, 0.])
    q = r.as_quat().astype(np.float32)
    c = np.asarray([ 1., 2., 3. ], dtype=np.float32)

    pred_quat, pred_coord = tr(torch.from_numpy(q)[None,...], torch.from_numpy(c)[None,...])
    pred_r = Rotation.from_quat(pred_quat.detach().numpy())
    pred_c = pred_coord.detach().numpy()
    expect_scale = 3.*np.exp(2.)
    expect_c = -(r.apply(expect_scale*np.array([ 3., 4., 0. ]))[[2,1]]) + np.array([1.,2.])
    print ((r.inv()*pred_r).magnitude())
    print (pred_c, expect_c, expect_scale)



if __name__ == '__main__':
    _test_local_to_global_transform_offset()