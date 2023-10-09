'''
Quaternion functions compatible with scipy.

The real component comes last which makes it compatible to the scipy convention.
pytorch3d and kornia are different.
'''
from typing import Union, Final
import torch
from torch import Tensor

# Note to future self: matrix_to_quaternion from pytorch3d triggers an internal
# error when attempted to convert to onnx.

iw : Final[int] = 3
ii : Final[int] = 0
ij : Final[int] = 1
ik : Final[int] = 2
iijk  : Final[slice] = slice(0,3)


def mult(u, v):
    """
    Multiplication of two quaternions.

    Format: The last dimension contains the quaternion components which 
            are ordered as (i,j,k,w), i.e. real component last. 
            The other dimension have to match.
    """
    out = torch.empty_like(u)
    out[...,iw] = u[...,iw]*v[...,iw] - u[...,ii]*v[...,ii] - u[...,ij]*v[...,ij] - u[...,ik]*v[...,ik]
    out[...,ii] = u[...,iw]*v[...,ii] + u[...,ii]*v[...,iw] + u[...,ij]*v[...,ik] - u[...,ik]*v[...,ij]
    out[...,ij] = u[...,iw]*v[...,ij] - u[...,ii]*v[...,ik] + u[...,ij]*v[...,iw] + u[...,ik]*v[...,ii]
    out[...,ik] = u[...,iw]*v[...,ik] + u[...,ii]*v[...,ij] - u[...,ij]*v[...,ii] + u[...,ik]*v[...,iw]
    return out


def rotate(q, p):
    """
    Rotation of vectors in p by quaternions in q

    Format: The last dimension contains the quaternion components which 
            are ordered as (i,j,k,w), i.e. real component last.
            The other dimensions follow the default broadcasting rules.
    """
    shape = torch.broadcast_shapes(p.shape[:-1], q.shape[:-1])
    qi = q[...,ii]
    qj = q[...,ij]
    qk = q[...,ik]
    qw = q[...,iw]
    pi = p[...,ii]
    pj = p[...,ij]
    pk = p[...,ik]

    tmp = q.new_empty(shape+(4,))
    out = p.new_empty(shape+(3,))
    
    # Compute tmp = q*p, identifying p with a purly imaginary quaternion.
    tmp[...,iw] =                 - qi*pi - qj*pj - qk*pk
    tmp[...,ii] = qw*pi                   + qj*pk - qk*pj
    tmp[...,ij] = qw*pj - qi*pk                   + qk*pi
    tmp[...,ik] = qw*pk + qi*pj - qj*pi
    # Compute tmp*q^-1.
    out[...,ii] = -tmp[...,iw]*qi + tmp[...,ii]*qw - tmp[...,ij]*qk + tmp[...,ik]*qj
    out[...,ij] = -tmp[...,iw]*qj + tmp[...,ii]*qk + tmp[...,ij]*qw - tmp[...,ik]*qi
    out[...,ik] = -tmp[...,iw]*qk - tmp[...,ii]*qj + tmp[...,ij]*qi + tmp[...,ik]*qw
    return out


def tomatrix(q):
    """
        Input: Quaternions. 
               Quaternions dimensions must be in the last tensor dimensions. 
               Quaternions must be normalized.
               Real component must be last.
    """
    qi = q[...,ii]
    qj = q[...,ij]
    qk = q[...,ik]
    qw = q[...,iw]
    out = q.new_empty(q.shape[:-1]+(3,3))
    out[...,0,0] = 1. - 2.*(qj*qj + qk*qk)
    out[...,1,0] =      2.*(qi*qj + qk*qw)
    out[...,2,0] =      2.*(qi*qk - qj*qw)
    out[...,0,1] =      2.*(qi*qj - qk*qw)
    out[...,1,1] = 1. - 2.*(qi*qi + qk*qk)
    out[...,2,1] =      2.*(qj*qk + qi*qw)
    out[...,0,2] =      2.*(qi*qk + qj*qw)
    out[...,1,2] =      2.*(qj*qk - qi*qw)
    out[...,2,2] = 1. - 2.*(qi*qi + qj*qj)
    return out


def from_rotvec(r, eps=1.e-12):
    shape = r.shape[:-1]
    q = r.new_empty(shape+(4,))
    angle = torch.norm(r, dim=-1, keepdim=True)
    r = r / (angle + eps)
    sin_term = torch.sin(angle*0.5)
    q[...,iw] = torch.cos(angle[...,0]*0.5)
    q[...,iijk] = r*sin_term
    return q


def to_rotvec(q : Tensor, eps=1.e-12) -> Tensor:
    # Making the real component positive constraints the output
    # to rotations angles between 0 and pi. Negative reals correspond
    # to angles beyond that, which is equivalent to using a negative angle
    # which is equivalent to flipping the axis and using the absolute of
    # the angle.
    q = positivereal(q)
    w = q[...,iw]
    axis = q[...,iijk]
    norm = torch.norm(axis, dim=-1, keepdim=True)
    angle = torch.atan2(norm[...,0], w).mul(2.)
    axis = axis * angle[...,None]/(norm+eps)
    return axis


def rotation_delta(from_ : Tensor, to_ : Tensor):
    frominv = from_.clone()
    frominv[...,iijk] = -frominv[...,iijk]
    rotvec = to_rotvec(mult(frominv, to_))
    return rotvec


def slerp(p : Tensor, q : Tensor, t : Union[float,Tensor], eps=1.e-12):
    # computes p (p* q)^t
    rotvec = rotation_delta(p, q).mul(t)
    rotq = from_rotvec(rotvec)
    return mult(p, rotq)


def positivereal(q):
    s = torch.sign(q[...,iw])
    return q*s[...,None]


def normalized(q):
    return torch.nn.functional.normalize(q, p=2., dim=-1, eps=1.e-6)


def distance(a, b):
    # See https://math.stackexchange.com/questions/90081/quaternion-distance
    # Different variants. Don't make much difference ... I think
    return 1.-torch.square(torch.sum(a * b, dim=-1))
    #return 1. - torch.abs(torch.sum(a * b, dim=-1))
    #return torch.min(torch.norm(a-b,p=2,dim=-1), torch.norm(a+b,p=2,dim=-1))

def geodesicdistance(a,b):
    return 2.*torch.acos(torch.sum(a * b, dim=-1).abs().min(torch.as_tensor(1.,dtype=a.dtype)))