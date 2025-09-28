import numpy as np
import pickle
from os.path import join, dirname

import torch
import torch.nn as nn


def _load(fn):
    with open(fn, "rb") as f:
        return pickle.load(f)


def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order="C")
    return arr


_current_folder = dirname(__file__)


class BFMModel(object):
    def __init__(self, shape_dim=40, exp_dim=10):
        bfm = _load(join(_current_folder, "bfm_noneck_v3.pkl"))
        self.u = bfm.get("u").astype(np.float32)  # fix bug
        self.w_shp = bfm.get("w_shp").astype(np.float32)[..., :shape_dim]
        self.w_exp = bfm.get("w_exp").astype(np.float32)[..., :exp_dim]
        self.tri = _load(
            join(_current_folder, "tri.pkl")
        )  # this tri/face is re-built for bfm_noneck_v3
        self.vertexcount = self.u.shape[0] // 3

        self.tri = _to_ctype(self.tri.T).astype(np.int32)

        self.keypoints = bfm.get("keypoints").astype(np.int64)[::3] // 3
        # Fix up landmarks of the eye because the old positions don't work any more
        # with the deformations I do for the closed eyes.
        left_eye_new = [1959, 3887, 5048, 6216, 3513, 4674]
        right_eye_new = [9956, 11223, 12384, 14327, 11495, 12656]
        self.keypoints[[36, 37, 38, 39, 41, 40]] = left_eye_new
        self.keypoints[[42, 43, 44, 45, 47, 46]] = right_eye_new

        w = np.concatenate((self.w_shp, self.w_exp), axis=1)
        self.w_norm = np.linalg.norm(w, axis=0)

    @property
    def scaled_shp_base(self):
        w_shp = 20.0 * self.w_shp.reshape((self.vertexcount, 3, -1))
        w_shp = w_shp.transpose([2, 0, 1])
        w_shp *= np.array([[[1.0, -1.0, -1.0]]])
        return w_shp

    @property
    def scaled_exp_base(self):
        w_exp = 5.0e-5 * self.w_exp.reshape((self.vertexcount, 3, -1))
        w_exp = w_exp.transpose([2, 0, 1])
        w_exp *= np.array([[[1.0, -1.0, -1.0]]])
        return w_exp

    @property
    def scaled_bases(self):
        """num eigvecs, num vertices, 3"""
        return np.concatenate([self.scaled_shp_base, self.scaled_exp_base], axis=0)

    @property
    def scaled_vertices(self):
        """shape (num vertices,3)"""
        actualcenter = np.array([0.0, -0.26, -0.9], dtype="f4")
        vertices = self.u.reshape((-1, 3)) * 1.0e-5 * np.array([[1.0, -1.0, -1.0]], dtype="f4")
        vertices -= actualcenter[None, :]
        return np.ascontiguousarray(vertices)

    @property
    def scaled_tri(self):
        tri = self.tri
        tri = tri[..., [2, 1, 0]]
        return np.ascontiguousarray(tri)


class ScaledBfmModule(nn.Module):
    def __init__(self, original: BFMModel):
        super().__init__()
        self.register_buffer("vertices", torch.from_numpy(original.scaled_vertices))
        self.register_buffer("deform_base", torch.from_numpy(original.scaled_bases))
        self.register_buffer("tri", torch.from_numpy(original.scaled_tri))
        self.register_buffer("keypoints", torch.from_numpy(original.keypoints).to(torch.long))

    @property
    def num_eigvecs(self):
        return self.deform_base.size(0)

    def forward(self, shapeparams):
        verts = self.deform_base * shapeparams[..., None, None]
        verts = torch.sum(verts, dim=-3)
        verts += self.vertices
        return verts
