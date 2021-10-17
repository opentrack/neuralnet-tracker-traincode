from os.path import join, dirname
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init

#import torchvision.models
#from torchvision.models import mnasnet
import neuralnets.torchquaternion as torchquaternion


__all__ = [ 
    'clear_denormals_inplace', 
    'HeadPoseLayerQuaternion', 'CenterOfMassAndBlobRadiusSqr', 'CenterOfMassAndStd', 'CenterOfMass',
    'load_deformable_head_keypoints', 'depthwise_separable_convolution',
    'DeformableHeadKeypoints','GBN', "rigid_transformation_25d", "BoundingBox",
    "SimilarityTransform"
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


class GBN(torch.nn.Module):
    # Based on the code from 
    # https://medium.com/deeplearningmadeeasy/ghost-batchnorm-explained-e0fa9d651e03
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=32, momentum=0.01):
        super(GBN, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm2d(self.input_dim, momentum=momentum, affine=False)
        self.weight = nn.Parameter(torch.Tensor(input_dim))
        self.bias = nn.Parameter(torch.Tensor(input_dim))
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def forward(self, x):
        if self.training:
            n = max(1, round(x.shape[0] / self.virtual_batch_size))
            y = torch.cat([ self.bn(x_) for x_ in torch.chunk(x,n, 0) ])
        else:
            y = self.bn(x)
        return y.mul(self.weight[None,:,None,None]).add(self.bias[None,:,None,None])


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
    tmp_pos = torchquaternion.rotate(r[:,None,:], points)
    tmp_pos = tmp_pos * s[:,None,:]
    screen_pos = world_to_screen(tmp_pos)
    screen_pos = screen_pos.clone()
    screen_pos[:,:,:2] += t[:,None,:]
    return screen_pos


# TODO: remove this
class HeadPoseLayerQuaternion(nn.Module):
    num_input = 7

    def __init__(self):
        super(HeadPoseLayerQuaternion, self).__init__()

    def headcenter_to_screen_3d(self, p):
        # Input: B x pointcount x (x,y,z)
        assert len(p.shape)==3 and p.shape[2]==3
        assert hasattr(self,'head_quat'), "Run forward pass first"
        assert hasattr(self,'head_position'), "Run forward pass first"
        assert hasattr(self,'head_size'), "Run forward pass first"
        return rigid_transformation_25d(
            self.head_quat,
            self.head_position,
            self.head_size,
            p)

    def forward(self, x):
        assert x.shape[1]==self.num_input
        self.unnormalized = x[...,3:self.num_input]
        self.head_size = x[...,2:3]
        self.head_position = x[...,:2]
        self.head_possize = x[...,:3]
        self.head_quat = torchquaternion.normalized(self.unnormalized)
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


class CenterOfMass(nn.Module):
    def __init__(self, shape, half_size=1.):
        super(CenterOfMass, self).__init__()
        p = torch.empty((2,shape[0],shape[1]), dtype=torch.float32)
        # Positions include the endpoints.
        p[0,...] = torch.linspace(-half_size, half_size, shape[1])[None,:]
        p[1,...] = torch.linspace(-half_size, half_size, shape[0])[:,None]
        self.position_code = nn.Parameter(p)
        self.position_code.requires_grad=False
        self._shape = shape

    def forward(self, x):
        assert x.shape[1] == self._shape[0], f'input shape {x.shape} vs expected {self._shape}'
        assert x.shape[2] == self._shape[1], f'input shape {x.shape} vs expected {self._shape}'
        mean = torch.sum(x[:,None,:,:] * self.position_code[None,...], dim=[2,3])
        return mean

class CenterOfMassAndStd(CenterOfMass):
    def __init__(self, shape, eps = 1.e-4, half_size=1.):
        super(CenterOfMassAndStd, self).__init__(shape, half_size)
        self._eps = eps

    def forward(self, x):
        mean = super(CenterOfMassAndStd, self).forward(x)
        diff = self.position_code[None,...] - mean[...,None,None]
        diff_squared = diff*diff # ONNX export does not know operator squared
        std  = torch.sqrt(torch.sum(x[:,None,:,:]*diff_squared, dim=[2,3])+self._eps)
        return mean, std

class CenterOfMassAndBlobRadiusSqr(CenterOfMass):
    def __init__(self, shape, half_size=1.):
        super(CenterOfMassAndBlobRadiusSqr, self).__init__(shape, half_size)

    def forward(self, x):
        mean = super(CenterOfMassAndBlobRadiusSqr, self).forward(x)
        diff = self.position_code[None,...] - mean[...,None,None]
        dist_squared = torch.sum(diff*diff, dim=1)
        mean_dist_sqr  = torch.sum(x*dist_squared, dim=[1,2])
        return mean, mean_dist_sqr


class BoundingBox(nn.Module):
    def __init__(self):
        super(BoundingBox, self).__init__()
    def forward(self, boxparams):
        assert boxparams.shape[1] == 4
        boxsize = F.softplus(boxparams[:,2:].clone())
        boxcenter = boxparams[:,:2]
        roi_box = torch.cat([ boxcenter - boxsize, boxcenter + boxsize ], dim=1)
        return roi_box


class SimilarityTransform(nn.Module):
    def __init__(self):
        super(SimilarityTransform, self).__init__()
        self.p = nn.Parameter(torch.tensor([[0., 0., 0.,0.]], dtype=torch.float32))

    def _compute_correction_quat(self, bs):
        c = self.p[:,0]
        q = c.new_empty((bs,4))
        q[:,:2] = 0.
        q[:,2] = torch.sin(c)
        q[:,3] = torch.cos(c)
        return q

    def _compute_correction_offset(self, bs):
        c = self.p[:,1:3]
        q = c.new_empty((bs,3))
        q[:,:2] = c
        q[:,2] = 0.
        return q

    def _compute_correction_scale(self, bs):
        c = self.p[:,3]
        s = c.new_empty((bs,1))
        s[:,:] = F.softplus(c)
        return s

    # TODO: use rigid_transformation_25d?
    def forward(self, quats, coords):
        bs = quats.size(0)
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

        scale = scale * self._compute_correction_scale(bs)

        z = self._compute_correction_quat(bs)
        pred_quat = torchquaternion.mult(quats, z)

        pos_corr = self._compute_correction_offset(bs)
        pos_corr = torchquaternion.rotate(quats, pos_corr)
        pos_corr = world_to_screen(pos_corr)[...,:2]
        pos_corr = pos_corr * scale
        screen_pos = pos_corr + head_center_screenspace
        pred_pos = torch.cat([
            screen_pos,
            scale
        ], axis=1)
        return (pred_quat, pred_pos)


def _test_similarity_transform():
    from scipy.spatial.transform import Rotation

    tr = SimilarityTransform()
    tr.p.data[0,:] = torch.tensor([ 1., 3., 4., 2. ])

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
    _test_similarity_transform()