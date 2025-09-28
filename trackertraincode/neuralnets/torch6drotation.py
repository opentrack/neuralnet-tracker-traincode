import torch
from torch import Tensor
import numpy as np

"""
Inspired by Yi Zhou et al. "On the Continuity of Rotation Representations in Neural Networks" (2020)


"""


def _reshape_to_vectors(z: Tensor):
    return z.view(*z.shape[:-1], 2, 3)


def _vdot(a: Tensor, b: Tensor):
    return (a * b).sum(-1)


def orthonormality_loss(m: Tensor):
    assert m.shape[-1] == 6
    m = _reshape_to_vectors(m)
    mm = torch.matmul(m, m.mT)
    return (mm - torch.eye(2, device=m.device)).square().flatten(-2, -1).mean(-1)


def tomatrix(sixdrot: Tensor):
    """
    Input: Tensor where the last dimension are the 6 rotation parameters
    """
    assert sixdrot.shape[-1] == 6
    prefix = sixdrot.shape[:-1]
    sixdrot = sixdrot.reshape(-1, 6)
    x, y = _reshape_to_vectors(sixdrot).unbind(-2)
    z = torch.cross(x, y, dim=-1)
    y = torch.cross(z, x, dim=-1)
    out = torch.stack([x, y, z], dim=-2)
    out = torch.nn.functional.normalize(out, dim=-1, eps=1e-6)
    # Replace very non-orthonormal outputs ...
    eye = torch.eye(3, device=out.device, dtype=sixdrot.dtype)[None, :, :]
    badness = torch.linalg.norm(
        (torch.matmul(out, out.transpose(-2, -1)) - eye).flatten(start_dim=-2),
        ord=torch.inf,
        dim=-1,
    )
    # out[badness > 1.e-3,:,:] = eye
    out = torch.where(badness[:, None, None] > 1.0e-3, eye, out)
    out = out.view(*prefix, 3, 3)
    return out


def frommatrix(m: Tensor):
    """
    Input: Matrix dimensions shall be in the last dimension of the tensor.
           Must be 3x3.
    """
    assert m.shape[-2:] == (3, 3)
    return m[..., :-1, :].flatten(-2, -1)


def orthonormality_loss(m: Tensor):
    assert m.shape[-1] == 6
    m = _reshape_to_vectors(m)
    mm = torch.matmul(m, m.mT)
    return (mm - torch.eye(2, device=m.device)).square().flatten(-2, -1).mean(-1)


def rotation_distance_loss(a: Tensor, b: Tensor):
    """The  cos of the geodesic distance, shifted and scaled."""
    assert a.shape[-2:] == (3, 3)
    assert b.shape[-2:] == (3, 3)
    return 0.75 - 0.25 * torch.diagonal(torch.matmul(a, b.mT), dim1=-2, dim2=-1).sum(dim=-1)
