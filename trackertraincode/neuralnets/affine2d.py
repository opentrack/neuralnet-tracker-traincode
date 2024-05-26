import torch
from torch.nn import functional as F
from typing import Optional, Union

from trackertraincode.neuralnets.math import matvecmul

SQRT2 = torch.sqrt(torch.tensor(2.))

MaybeTensor = Optional[torch.Tensor]

class Affine2d(object):
    '''
        Represents transformations in N x 2 x 3 format
    '''
    def __init__(self, m : torch.Tensor):
        assert len(m.shape)>=2 and m.shape[-2:]==(2,3)
        self.m : torch.Tensor = m.to(torch.float32)
        self.m.requires_grad = False

    @staticmethod
    def identity(device = None):
        return Affine2d(torch.eye(2,3,device=device))

    @staticmethod
    def _new_empty(translations : MaybeTensor, angles : MaybeTensor, scales : MaybeTensor):
        if translations is not None:
            assert translations.shape[-1]==2
            return translations.new_empty(translations.shape+(3,))
        if angles is not None:
            return angles.new_empty(angles.shape+(2,3))
        if scales is not None:
            return scales.new_empty(scales.shape+(2,3))

    @staticmethod
    def trs(translations : MaybeTensor = None, angles : MaybeTensor = None, scales : MaybeTensor = None):
        r = Affine2d._new_empty(translations, angles, scales)
        if angles is None:
            r[...,:2,:2] = torch.eye(2)[None,...]
            if scales is not None:
                r *= scales[...,None,None]
        else:
            cs = torch.cos(angles)
            sn = torch.sin(angles)
            if scales is not None:
                cs *= scales
                sn *= scales
            r[...,0,0] = cs
            r[...,0,1] = -sn
            r[...,1,0] = sn
            r[...,1,1] = cs
        if translations is not None:
            r[...,:2,2] = translations
        else:
            r[...,:2,2] = 0
        return Affine2d(r)

    @staticmethod
    def trs_inv(translations : MaybeTensor = None, angles : MaybeTensor = None, scales : MaybeTensor = None):
        r = Affine2d._new_empty(translations, angles, scales)
        if angles is None:
            r[...,:2,:2] = torch.eye(2)[None,...]
            if scales is not None:
                r /= scales[...,None,None]
        else:
            cs = torch.cos(angles)
            sn = torch.sin(angles)
            if scales is not None:
                cs /= scales
                sn /= scales
            r[...,0,0] = cs
            r[...,0,1] = sn
            r[...,1,0] = -sn
            r[...,1,1] = cs
        if translations is not None:
            r[...,:2,2] = matvecmul(r[...,:2,:2], -translations)
        else:
            r[...,:2,2] = 0
        return Affine2d(r)

    @staticmethod
    def horizontal_flip(xcenter : torch.Tensor):
        # x' = -(x-c) + c
        # -> scale = -1
        # -> offset = 2c
        m = xcenter.new_empty(xcenter.shape+(2,3))
        m[...,0,0] = -1.
        m[...,1,1] = 1.
        m[...,1,0] = m[...,0,1] = 0.
        m[...,0,2] = 2*xcenter
        m[...,1,2] = 0.
        return Affine2d(m)

    @staticmethod
    def range_remap(inmin : torch.Tensor, inmax : torch.Tensor, outmin : torch.Tensor, outmax : torch.Tensor):
        inmin, inmax, outmin, outmax = \
            [ torch.as_tensor(x).to(dtype=torch.float32) for x in (inmin, inmax, outmin, outmax) ]
        # Xout = (Xin - inmin)/(inmax-inmin)*(outmax-outmin)+outmin
        # -> scale = (outmax-outmin)/(inmax-inmin)
        # -> offset = outmin - inmin*scale
        m = inmin.new_empty(inmin.shape+(2,3))
        s = (outmax-outmin)/(inmax-inmin)
        m[...,0,0] = s
        m[...,1,1] = s
        m[...,1,0] = m[...,0,1] = 0.
        m[...,:,2] = outmin - inmin*s[...,None]
        return Affine2d(m)

    @staticmethod
    def range_remap_2d(inmin : torch.Tensor, inmax : torch.Tensor, outmin : torch.Tensor, outmax : torch.Tensor):
        inmin, inmax, outmin, outmax = \
            [ torch.as_tensor(x).to(dtype=torch.float32) for x in (inmin, inmax, outmin, outmax) ]
        # Xout = (Xin - inmin)/(inmax-inmin)*(outmax-outmin)+outmin
        # -> scale = (outmax-outmin)/(inmax-inmin)
        # -> offset = outmin - inmin*scale
        m = inmin.new_empty(inmin.shape[:-1]+(2,3))
        s = (outmax-outmin)/(inmax-inmin)
        m[...,0,0] = s[...,0]
        m[...,1,1] = s[...,1]
        m[...,1,0] = m[...,0,1] = 0.
        m[...,:,2] = outmin - inmin*s
        return Affine2d(m)

    def tensor(self):
        return self.m
    
    def tensor33(self):
        r = self.m.new_empty(*self.m.shape[:-2],3,3)
        r[...,:2,:] = self.m
        r[...,2,:2] = 0.
        r[...,2,2] = 1.
        return r

    def to(self, *args, **kwargs):
        return Affine2d(self.m.to(*args, **kwargs))

    @property
    def R(self):
        return self.m[...,:2,:2]

    @property
    def R33(self):
        r = self.m.new_empty(*self.m.shape[:-2],3,3)
        r[...,:2,:2] = self.R
        r[...,2,:2] = 0.
        r[...,:2,2] = 0.
        r[...,2,2] = 1.
        return r

    @property
    def T(self):
        return self.m[...,:2,2]

    def size(self, i):
        return self.m.size(i)

    def __matmul__(self, other):
        # TODO: update Pytorch and use broadcast_shapes
        tensor_with_right_shape, _ = torch.broadcast_tensors(self.m, other.m)
        m = torch.empty_like(tensor_with_right_shape)
        torch.matmul(self.R, other.R, out=m[...,:2,:2])
        m[...,:,2] = matvecmul(self.R, other.T)
        m[...,:,2] += self.T
        return Affine2d(m)
    
    def inv(self):
        m = torch.empty_like(self.m)
        m[...,:,:2] = r = torch.inverse(self.R)
        m[...,:,2] = -matvecmul(r, self.T)
        return Affine2d(m)

    @property
    def scales(self):
        ''' Recover scaling factor '''
        # sqrt ( s^2 + s^2 )
        return torch.norm(self.m[...,:,:2],dim=(-2,-1)) / SQRT2
    
    @property
    def det(self):
        a, b, c, d = self.m[...,0,0], self.m[...,0,1] \
            , self.m[...,1,0], self.m[...,1,1]
        return a*d - b*c

    def __getitem__(self, val):
        return Affine2d(self.m.__getitem__(val))

    def reshape(self, shape):
        return Affine2d(self.m.reshape(shape+(2,3)))
    
    def expand(self, *shape):
        return Affine2d(self.m.expand(*shape,-1,-1))

    def repeat(self, size):
        return Affine2d(self.m.repeat(size+(1,1)))

    def view(self, *shape):
        return Affine2d(self.m.view(*shape,2,3))
    

def roi_normalizing_transform(roi : torch.Tensor):
    assert roi.shape[-1]==4
    prefixshape = roi.shape[:-1]
    roi = roi.view(-1,4)
    N,_ = roi.shape
    out_min = torch.tensor([-1.,-1.],dtype=roi.dtype,device=roi.device).expand(N,2)
    out_max = torch.tensor([1.,1.],dtype=roi.dtype,device=roi.device).expand(N,2)
    return Affine2d.range_remap_2d(roi[:,:2], roi[:,2:], out_min, out_max).reshape(prefixshape)