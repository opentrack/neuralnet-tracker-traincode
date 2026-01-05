from typing import List
import numpy as np
import torch
import torch.nn.functional as F
import functools


def matvecmul(m, v):
    """
    Like matmul, except v must not have the column dimension that would be present for matrices.
    Equivalent to
        torch.matmul(m, v[...,None])[...,0]
    """
    return torch.matmul(m, v[..., None])[..., 0]


def affinevecmul(m, v):
    o = matvecmul(m[..., :, :-1], v)
    o += m[..., :, -1]
    return o


def random_uniform(shape, minval, maxval, *args, **kwargs):
    return torch.rand(shape, *args, **kwargs) * (maxval - minval) + minval


def random_choice(shape: tuple, values: torch.Tensor, weights: torch.Tensor, replacement):
    num_samples = np.prod(shape) if len(shape) else 1
    indices = torch.multinomial(weights, num_samples, replacement=replacement)
    picks = values[indices]
    return torch.reshape(picks, shape)


def smoothclip0(x, inplace: bool = False) -> torch.Tensor:
    y = F.elu(x, inplace=inplace)
    y.add_(1.0)
    return y


def inv_smoothclip0(x: torch.Tensor | float) -> torch.Tensor:
    x = torch.as_tensor(x)
    z = torch.atleast_1d(x)
    mask = z > 1.0
    z = z.clone()
    z[mask] -= 1.0
    z[~mask] = torch.log(z[~mask])
    return z.view(x.shape)


@torch.jit.script
def sqrclip0(x: torch.Tensor, beta: float):
    z = F.relu(x + beta * 0.5)
    return torch.where(z < beta, (0.5 / beta) * torch.square(z), z - 0.5 * beta)


@torch.jit.script
def inv_sqrclip0(y: torch.Tensor, beta: float):
    return torch.where(y > 0.5 * beta, y + 0.5 * beta, torch.sqrt(beta * 2.0 * y)) - beta * 0.5


def chain_gmm(*matrices: List[torch.Tensor]) -> torch.Tensor:
    return functools.reduce(torch.matmul, matrices)


def rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    # R = I + (sin a)K + (1-cos a)KK
    # K = K' / a
    # => R = I + (sin a)/a K' + (1-cos a)/aa K'K'
    # => R = I + sinc (a) K' + 2/4 sinc(a/2)^2 K'K'

    shape = rotvec.shape
    device, dtype = rotvec.device, rotvec.dtype

    # Compute angle magnitude: a = ||rotvec||
    angles = torch.norm(rotvec, p=2, dim=-1)

    # Extract components for skew-symmetric matrix
    rx, ry, rz = rotvec.unbind(-1)
    zeros = rx.new_zeros(rx.shape)

    # Create skew-symmetric matrix K' (cross product matrix)
    cross_product_matrix = torch.stack(
        [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=-1
    ).view(shape + (3,))

    cross_product_matrix_sqrd = cross_product_matrix @ cross_product_matrix

    identity = torch.eye(3, dtype=dtype, device=device)

    # Compute sinc terms
    # Second term: sinc(a) = sin(a)/a = torch.sinc(a/π)
    sinc_a = torch.sinc(angles / torch.pi)[..., None, None]

    # Third term: (1/2) * sinc²(a/2) = (1/2) * [sin(a/2)/(a/2)]²
    sinc_a_half = torch.sinc(angles / (2.0 * torch.pi))[..., None, None]

    # Combine: R = I + sinc(a) K' + (1/2) sinc²(a/2) K'K'
    return (
        identity.expand_as(cross_product_matrix)
        + sinc_a * cross_product_matrix
        + (0.5 * sinc_a_half**2) * cross_product_matrix_sqrd
    )
