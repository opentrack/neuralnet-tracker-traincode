import torch
import numpy as np


def mult(u, v):
    """
    Multiplication of two quaternions.

    Format: The last dimension contains the quaternion components which 
            are ordered as (i,j,k,w), i.e. real component last. 
            The other dimension have to match.
    """
    # TODO: Allow broadcasting?
    iw = 3
    ii = 0
    ij = 1
    ik = 2
    out = torch.empty_like(u)
    out[:,iw] = u[:,iw]*v[:,iw] - u[:,ii]*v[:,ii] - u[:,ij]*v[:,ij] - u[:,ik]*v[:,ik]
    out[:,ii] = u[:,iw]*v[:,ii] + u[:,ii]*v[:,iw] + u[:,ij]*v[:,ik] - u[:,ik]*v[:,ij]
    out[:,ij] = u[:,iw]*v[:,ij] - u[:,ii]*v[:,ik] + u[:,ij]*v[:,iw] + u[:,ik]*v[:,ii]
    out[:,ik] = u[:,iw]*v[:,ik] + u[:,ii]*v[:,ij] - u[:,ij]*v[:,ii] + u[:,ik]*v[:,iw]
    return out


def rotate(q, p):
    """
    Rotation of vectors in p by quaternions in q

    Format: The last dimension contains the quaternion components which 
            are ordered as (i,j,k,w), i.e. real component last.
            The other dimensions follow the default broadcasting rules.
    """
    iw = 3
    ii = 0
    ij = 1
    ik = 2
    qi = q[...,ii]
    qj = q[...,ij]
    qk = q[...,ik]
    qw = q[...,iw]
    pi = p[...,ii]
    pj = p[...,ij]
    pk = p[...,ik]
    
    # FIXME: This part does not export to the onnx model, i.e. the shape
    #        of the tensors will be hardcoded according to the input during
    #        the export. Not a problem though.
    shape = tuple(np.maximum(pi.shape, qi.shape))
    tmp = q.new_empty(shape + (4,))
    out = q.new_empty(shape + (3,))
    
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