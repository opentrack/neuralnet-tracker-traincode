'''
Quaternion functions compatible with scipy.

The real component comes last which makes it compatible to the scipy convention.
pytorch3d and kornia are different.
'''

from typing import Union, Final
import numpy as np
import torch
from torch import Tensor

# Note to future self: matrix_to_quaternion from pytorch3d triggers an internal
# error when attempted to convert to onnx.

iw: Final[int] = 3
ii: Final[int] = 0
ij: Final[int] = 1
ik: Final[int] = 2
iijk: Final[slice] = slice(0, 3)


def _mat_repr(u: Tensor, indices=(iw, ii, ij, ik, ii, iw, ik, ij, ij, ik, iw, ii, ik, ij, ii, iw)):
    umat = u[..., indices]
    umat = umat * umat.new_tensor(
        [1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0]
    )
    umat = umat.view(*u.shape, 4)
    return umat


def _vec_repr(v: Tensor):
    return v[..., [iw, ii, ij, ik]].view(*v.shape, 1)


def _quat_repr(vec: Tensor):
    return vec.view(*vec.shape[:-1])[..., [1, 2, 3, 0]]


def mult(u: Tensor, v: Tensor):
    """
    Multiplication of two quaternions.

    Format: The last dimension contains the quaternion components which
            are ordered as (i,j,k,w), i.e. real component last.
            The other dimension have to match.
    """
    return _quat_repr(torch.matmul(_mat_repr(u), _vec_repr(v)))


def rotate(q, p):
    """
    Rotation of vectors in p by quaternions in q

    Format: The last dimension contains the quaternion components which
            are ordered as (i,j,k,w), i.e. real component last.
            The other dimensions follow the default broadcasting rules.
    """

    # Compute tmp = q*p, identifying p with a purly imaginary quaternion.
    qmat = _mat_repr(q)
    pvec = p[..., None]
    tmp = torch.matmul(qmat[..., :, 1:], pvec)
    # Compute tmp*q^-1.
    tmpmat = _mat_repr(tmp.view(tmp.shape[:-1]), (0, 1, 2, 3, 1, 0, 3, 2, 2, 3, 0, 1, 3, 2, 1, 0))
    out = torch.matmul(tmpmat[..., 1:, :], _vec_repr(conjugate(q)))
    return out.view(out.shape[:-1])


def tomatrix(q):
    """
    Input: Quaternions.
           Quaternions dimensions must be in the last tensor dimensions.
           Quaternions must be normalized.
           Real component must be last.
    """
    qi = q[..., ii]
    qj = q[..., ij]
    qk = q[..., ik]
    qw = q[..., iw]
    out = q.new_empty(q.shape[:-1] + (3, 3))
    out[..., 0, 0] = 1.0 - 2.0 * (qj * qj + qk * qk)
    out[..., 1, 0] = 2.0 * (qi * qj + qk * qw)
    out[..., 2, 0] = 2.0 * (qi * qk - qj * qw)
    out[..., 0, 1] = 2.0 * (qi * qj - qk * qw)
    out[..., 1, 1] = 1.0 - 2.0 * (qi * qi + qk * qk)
    out[..., 2, 1] = 2.0 * (qj * qk + qi * qw)
    out[..., 0, 2] = 2.0 * (qi * qk + qj * qw)
    out[..., 1, 2] = 2.0 * (qj * qk - qi * qw)
    out[..., 2, 2] = 1.0 - 2.0 * (qi * qi + qj * qj)
    return out


def from_matrix(m: Tensor):
    # See https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    # Also inspired by
    # https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    assert m.shape[-2:] == (3,3)
    shape = m.shape[:-2]
    m = m.reshape((-1,3,3))

    # 4 possibilties to compute the quaternion. Unstable computation with divisions
    # by zero or close to zero can occur. Further down, the best conditioned solution
    # is picked.
    sqrt_args = torch.matmul(
        torch.as_tensor(
            [
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0],  # Using that qj*qj + qk*qk + qi*qi = qw*qw - 1
            ],
            device=m.device,
            dtype=m.dtype,
        ),
        m[:,[0,1,2],[0,1,2],None]
        # Note: the code below with "diagonal" creates some weird onnx export with a conditional operator.
        # torch.diagonal(m, dim1=-2, dim2=-1)[...,None],
    )
    sqrt_args = torch.clamp(sqrt_args.sum(dim=-1) + 1.0, 1.0e-6, None)
    qx_from_x = torch.sqrt(sqrt_args).mul(0.5)

    idx1 = [1, 2, 1, 0, 1, 1, 2, 1, 0, 2, 0, 1]
    idx2 = [0, 0, 2, 2, 0, 2, 1, 0, 2, 1, 2, 0]
    signs = torch.as_tensor(
        [-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0], dtype=m.dtype, device=m.device
    )
    quat_vals = 0.25 * (m[:, idx1, idx2] + signs * m[:, idx2, idx1]) / qx_from_x.repeat_interleave(3, dim=-1)

    (
        qw_from_k,
        qi_from_k,
        qj_from_k,
        qw_from_j,
        qi_from_j,
        qk_from_j,
        qw_from_i,
        qj_from_i,
        qk_from_i,
        qi_from_w,
        qj_from_w,
        qk_from_w,
    ) = quat_vals.unbind(-1)
    qk_from_k, qj_from_j, qi_from_i, qw_from_w = qx_from_x.unbind(-1)

    quat_candidates = torch.stack(
        [
            torch.stack([qi_from_k, qj_from_k, qk_from_k, qw_from_k], dim=-1),
            torch.stack([qi_from_j, qj_from_j, qk_from_j, qw_from_j], dim=-1),
            torch.stack([qi_from_i, qj_from_i, qk_from_i, qw_from_i], dim=-1),
            torch.stack([qi_from_w, qj_from_w, qk_from_w, qw_from_w], dim=-1),
        ],
        dim=1,
    )

    with torch.no_grad():
        quat_pick = torch.argmax(sqrt_args, dim=-1)
        mask = torch.nn.functional.one_hot(quat_pick, 4) == 1
    quat = quat_candidates[mask]
    quat = positivereal(quat)
    quat = quat.view(*shape, 4)
    return quat


def conjugate(q: Tensor):
    assert iw == 3
    return q * q.new_tensor([-1.0, -1.0, -1.0, 1.0])


def from_rotvec(r, eps=1.0e-12):
    shape = r.shape[:-1]
    q = r.new_empty(shape + (4,))
    angle = torch.norm(r, dim=-1, keepdim=True)
    r = r / (angle + eps)
    sin_term = torch.sin(angle * 0.5)
    q[..., iw] = torch.cos(angle[..., 0] * 0.5)
    q[..., iijk] = r * sin_term
    return q


def to_rotvec(q: Tensor, eps=1.0e-12) -> Tensor:
    # Making the real component positive constraints the output
    # to rotations angles between 0 and pi. Negative reals correspond
    # to angles beyond that, which is equivalent to using a negative angle
    # which is equivalent to flipping the axis and using the absolute of
    # the angle.
    q = positivereal(q)
    w = q[..., iw]
    axis = q[..., iijk]
    norm = torch.norm(axis, dim=-1, keepdim=True)
    angle = torch.atan2(norm[..., 0], w).mul(2.0)
    axis = axis * angle[..., None] / (norm + eps)
    return axis


def rotation_delta(from_: Tensor, to_: Tensor):
    frominv = from_.clone()
    frominv[..., iijk] = -frominv[..., iijk]
    rotvec = to_rotvec(mult(frominv, to_))
    return rotvec


def slerp(p: Tensor, q: Tensor, t: Union[float, Tensor], eps=1.0e-12):
    # computes p (p* q)^t
    rotvec = rotation_delta(p, q).mul(t)
    rotq = from_rotvec(rotvec)
    return mult(p, rotq)


def positivereal(q):
    s = torch.sign(q[..., iw])
    return q * s[..., None]


def normalized(q):
    return torch.nn.functional.normalize(q, p=2.0, dim=-1, eps=1.0e-6)


def distance(a, b):
    # See https://math.stackexchange.com/questions/90081/quaternion-distance
    # Different variants. Don't make much difference ... I think
    return 1.0 - torch.square(torch.sum(a * b, dim=-1))
    # return 1. - torch.abs(torch.sum(a * b, dim=-1))
    # return torch.min(torch.norm(a-b,p=2,dim=-1), torch.norm(a+b,p=2,dim=-1))


def geodesicdistance(a, b):
    # Numerically unstable due to infinite gradients at -1 and 1???
    # return torch.arccos((2.*torch.sum(a * b, dim=-1).square() - 1).clip(-1.,1.))
    return rotation_delta(a, b).norm(dim=-1)


def quat_average(quats):
    quats = np.asarray(quats)
    # Ensemble size, number of samples, dimensionality
    E, N, D = quats.shape
    assert D == 4
    # Sum over ensemble to get an idea of the largest axis on average.
    # Then find the actual longest axis, i.e. i,j,k or w.
    pivot_axes = np.argmax(np.sum(np.abs(quats), axis=0), axis=-1)
    assert pivot_axes.shape == (N,)
    mask = np.take_along_axis(quats, pivot_axes[None, :, None], axis=-1) < 0.0
    mask = mask[..., 0]  # Skip quaternion dimension
    quats[mask, :] *= -1
    quats = np.average(quats, axis=0)
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    if not np.all(norms > 0.5):
        print("Oh oh either quat_average is bugged or rotations predictions differ wildly")
    quats /= norms
    return quats