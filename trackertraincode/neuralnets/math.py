from typing import List
import numpy as np
import torch
import torch.nn.functional as F
import functools

def matvecmul(m, v):
    '''
    Like matmul, except v must not have the column dimension that would be present for matrices.
    Equivalent to 
        torch.matmul(m, v[...,None])[...,0]
    '''
    return torch.matmul(m, v[...,None])[...,0]


def affinevecmul(m, v):
    o = matvecmul(m[...,:,:-1], v)
    o += m[...,:,-1]
    return o


def random_uniform(shape, minval, maxval, *args, **kwargs):
    return torch.rand(shape, *args, **kwargs)*(maxval-minval) + minval


def random_choice(shape : tuple, values : torch.Tensor, weights : torch.Tensor, replacement):
    num_samples = np.prod(shape) if len(shape) else 1
    indices = torch.multinomial(weights, num_samples, replacement=replacement)
    picks = values[indices]
    return torch.reshape(picks, shape)


def smoothclip0(x, inplace : bool = False) -> torch.Tensor:
    y = F.elu(x, inplace=inplace)
    y.add_(1.)
    return y


@torch.jit.script
def inv_smoothclip0(x : torch.Tensor) -> torch.Tensor:
    x = torch.as_tensor(x)
    z = torch.atleast_1d(x)
    mask = z>1.
    z = z.clone()
    z[mask] -= 1.
    z[~mask] = torch.log(z[~mask])
    return z.view(x.shape)


@torch.jit.script
def sqrclip0(x : torch.Tensor, beta : float):
    z = F.relu(x + beta*0.5)
    return torch.where(
        z < beta, 
        (0.5/beta)*torch.square(z),
        z - 0.5*beta)


@torch.jit.script
def inv_sqrclip0(y : torch.Tensor, beta  : float):
    return torch.where(
        y > 0.5*beta,
        y + 0.5*beta,
        torch.sqrt(beta*2.*y)
    ) - beta*0.5


def chain_gmm(*matrices : List[torch.Tensor]) -> torch.Tensor:
    return functools.reduce(torch.matmul, matrices)