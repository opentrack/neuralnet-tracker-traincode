from os.path import join, dirname
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init

import torchvision.models
from torchvision.models import mnasnet
import neuralnets.torchquaternion as torchquaternion


__all__ = [ 
    'clear_denormals_inplace', 
    'PoseOutputStage', 'SpatialMeanAndStd',
    'load_deformable_head_keypoints', 'depthwise_separable_convolution',
    'DeformableHeadKeypoints'
]


def clear_denormals_inplace(state_dict, threshold=1.e-20):
    # I tuned the threshold so I don't see a performance
    # decrease compared to pretrained weights from torchvision.
    # The real denormals start below 2.*10^-38
    print ("Denormals or zeros:")
    for k, v in state_dict.items():
        if v.dtype == torch.float32:
            mask = torch.abs(v) > threshold
            n = torch.count_nonzero(~mask)
            if n:
                print (f"{k:40s}: {n:10d} ({n/np.product(v.shape)*100}%)")
            v *= mask.to(torch.float32)


def world_to_screen(p):
    # Input ?? x (x,y,z)
    # Note the head/world coordinate system
    #   x - forward
    #   y - up
    #   z - right
    # The camera is looking along the negative x axis
    # Camera x-axis points right, and y points down.
    #        z points into the screen.
    #return -p[...,[2,1,0]]
    p = p[...,[2,1,0]].clone()
    p[...,:2] = -p[...,:2]
    return p


def depthwise_separable_convolution(in_, out, kernel, **kwargs):
    momentum = kwargs.pop("momentum", 0.001)
    return nn.Sequential(
        nn.Conv2d(in_, in_, kernel, groups=in_, bias=False, **kwargs),
        nn.BatchNorm2d(in_, momentum=momentum),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_, out, 1, bias=False),
        nn.BatchNorm2d(out, momentum=momentum)
    )


class PoseOutputStage(nn.Module):
    num_input = 7

    def __init__(self):
        super(PoseOutputStage, self).__init__()

    def headcenter_to_screen_3d(self, p):
        # Input: B x pointcount x (x,y,z)
        assert len(p.shape)==3 and p.shape[2]==3
        assert hasattr(self,'head_quat'), "Run forward pass first"
        assert hasattr(self,'head_position'), "Run forward pass first"
        assert hasattr(self,'head_size'), "Run forward pass first"
        tmp_pos = torchquaternion.rotate(self.head_quat[:,None,:], p)
        tmp_pos = tmp_pos * self.head_size[:,None,:]
        screen_pos = world_to_screen(tmp_pos)
        screen_pos = screen_pos.clone()
        screen_pos[:,:,:2] += self.head_position[:,None,:]
        return screen_pos

    def forward(self, x):
        assert x.shape[1]==7
        self.unnormalized = x[...,3:7]
        self.head_size = x[...,2:3]
        self.head_position = x[...,:2]
        self.head_possize = x[...,:3]
        # In order for the result to represent a rotation we have to normalize it.
        self.head_quat = F.normalize(self.unnormalized, p=2, dim=1)
        return (self.head_possize, self.head_quat)


class DeformableHeadKeypoints(nn.Module):
    def __init__(self):
        super(DeformableHeadKeypoints, self).__init__()
        # TODO: rename
        self.num_eigvecs = 50
        self.keypts, self.keyeigvecs = load_deformable_head_keypoints(40, 10)
    
    def deformvector(self, shapeparams):
        local_keypts = self.keyeigvecs[None,...] * shapeparams[:,:,None,None]
        local_keypts = torch.sum(local_keypts, dim=1)
        return local_keypts

    def forward(self, shapeparams):
        local_keypts = self.deformvector(shapeparams)
        local_keypts += self.keypts[None,...]
        assert len(local_keypts.shape)==3 and local_keypts.shape[2]==3
        # Return in format (batch x points x dimensions)
        return local_keypts


def load_deformable_head_keypoints(num_shape, num_expr, return_nn_parameters=True):
    """
        Returned array shapes:  (point num x dim) and (shape num x point num x dim)
    """
    # Extra offset, matching the one from PoseOutputStage baked in!
    keypts = np.load(join(dirname(__file__),'face_keypoints_base_68_3d.npy'))
    shp = np.load(join(dirname(__file__),'face_keypoints_base_68_3d_shp.npy'))
    exp = np.load(join(dirname(__file__),'face_keypoints_base_68_3d_exp.npy'))
    n = num_shape + num_expr
    assert n>0
    deltas = np.empty((n,)+shp.shape[1:], dtype=np.float32)
    deltas[:num_shape] = shp[:num_shape]
    deltas[num_shape:] = exp[:num_expr]
    if return_nn_parameters:
        keypts = nn.Parameter(torch.tensor(keypts, dtype=torch.float32))
        deltas = nn.Parameter(torch.tensor(deltas, dtype=torch.float32))
        keypts.requires_grad=False
        deltas.requires_grad=False
    return keypts, deltas


class SpatialMeanAndStd(nn.Module):
    def __init__(self, shape, eps = 1.e-4, half_size=1.):
        super(SpatialMeanAndStd, self).__init__()
        p = torch.empty((2,shape[0],shape[1]), dtype=torch.float32)
        p[0,...] = torch.linspace(-half_size, half_size, shape[1])[None,:]
        p[1,...] = torch.linspace(-half_size, half_size, shape[0])[:,None]
        self.position_code = nn.Parameter(p)
        self.position_code.requires_grad=False
        self._shape = shape
        self._eps = eps
    
    def forward(self, x):
        assert x.shape[1] == self._shape[0], f'input shape {x.shape} vs expected {self._shape}'
        assert x.shape[2] == self._shape[1], f'input shape {x.shape} vs expected {self._shape}'
        mean = torch.sum(x[:,None,:,:] * self.position_code[None,...], dim=[2,3])
        diff = self.position_code[None,...] - mean[...,None,None]
        diff_squared = diff*diff # ONNX export does not know operator squared
        std  = torch.sqrt(torch.sum(x[:,None,:,:]*diff_squared, dim=[2,3])+self._eps)
        return mean, std
