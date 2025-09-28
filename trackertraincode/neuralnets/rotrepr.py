import torch
from torch import Tensor
import dataclasses
from typing import Any, Type

import trackertraincode.neuralnets.torch6drotation as torch6drotation
import trackertraincode.neuralnets.torchquaternion as torchquaternion
from trackertraincode.neuralnets.math import smoothclip0


@dataclasses.dataclass
class QuatRepr:
    value : Tensor
    def rotate_points(self, pts : Tensor) -> Tensor:
        return torchquaternion.rotate(self.value[...,None,:], pts)
    def mult(self, other : 'QuatRepr') -> 'QuatRepr':
        return QuatRepr(torchquaternion.mult(self.value, other.value))
    @classmethod
    def make_rotate_x(cls : Any, angle : Tensor) -> 'QuatRepr':
        angle = 0.5*angle
        return QuatRepr(torch.cat([torch.sin(angle)[...,None], angle.new_zeros((*angle.shape,2)), torch.cos(angle)[...,None] ], dim=-1))
    @classmethod
    def from_features(cls : Type['QuatRepr'], z : Tensor) -> tuple['QuatRepr',Tensor]:
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
        return QuatRepr(quats), quats_unnormalized
    def as_quat(self) -> Tensor:
        return self.value
    @property
    def shape(self):
        return self.value.shape[:-1]
    # def relative_to(self, other : 'QuatRepr') -> Tensor:
    #     return torchquaternion.rotation_delta(other.value,self.value)
    def __getitem__(self, *args):
        return QuatRepr(self.value.__getitem__(*args))


@dataclasses.dataclass
class Mat33Repr:
    value : Tensor
    def rotate_points(self, pts : Tensor) -> Tensor:
        return torch.matmul(self.value, pts.swapaxes(-2,-1)).swapaxes(-2,-1)
    def mult(self, other : 'Mat33Repr') -> 'Mat33Repr':
        return Mat33Repr(torch.matmul(self.value, other.value))
    @classmethod
    def make_rotate_x(cls : Any, angle : Tensor) -> 'Mat33Repr':
        m = angle.new_zeros((*angle.shape,9))
        sn, cs = torch.sin(angle), torch.cos(angle)
        # 1  0  0
        # 0 cs -sn
        # 0 sn  cs
        m[...,0] = 1.
        m[...,4] = cs
        m[...,8] = cs
        m[...,5] = -sn
        m[...,7] = sn
        m = m.view(*angle.shape,3,3)
        return Mat33Repr(m)
    @classmethod
    def from_6drepr_features(cls : Type['Mat33Repr'], z : Tensor) -> 'Mat33Repr':
        return Mat33Repr(torch6drotation.tomatrix(z))
    def as_quat(self) -> Tensor:
        return torchquaternion.from_matrix(self.value)
    # def relative_to(self, other : 'Mat33Repr') -> Tensor:
    #     qself = torchquaternion.from_matrix(self.value)
    #     qother = torchquaternion.from_matrix(other.value)
    #     return torchquaternion.rotation_delta(qother,qself)
    @property
    def shape(self):
        return self.value.shape[:-2]
    def __getitem__(self, *args):
        return Mat33Repr(self.value.__getitem__(*args))

RotationRepr = QuatRepr | Mat33Repr
