from trackertraincode.facemodel.bfm import BFMModel, ScaledBfmModule
from trackertraincode.neuralnets.modelcomponents import PosedDeformableHead
from trackertraincode.neuralnets.rotrepr import QuatRepr
from trackertraincode.pipelines import Batch

import torch
import numpy as np


class PutRoiFromLandmarks(object):
    def __init__(self, extend_to_forehead=False):
        self.extend_to_forehead = extend_to_forehead
        self.headmodel = None
        if extend_to_forehead:
            rng = np.random.default_rng(seed=29387)
            self.headmodel = PosedDeformableHead(ScaledBfmModule(BFMModel()).downsampled(1024, rng))

    def _create_roi(self, sample: Batch):
        shapeparams = (
            sample["shapeparam"]
            if "shapeparams" in sample
            else torch.zeros(50, device=sample.device)
        )
        if self.extend_to_forehead:
            assert self.headmodel is not None
            vertices = self.headmodel(sample["coord"], QuatRepr(sample["pose"]), shapeparams)
            min_ = torch.amin(vertices[..., :2], dim=-2)
            max_ = torch.amax(vertices[..., :2], dim=-2)
        else:
            landmarks3d: torch.Tensor = sample["pt3d_68"]
            min_ = torch.amin(landmarks3d[..., :2], dim=-2)
            max_ = torch.amax(landmarks3d[..., :2], dim=-2)
        roi = torch.cat([min_, max_], dim=0).to(torch.float32)
        return roi

    def __call__(self, sample: Batch):
        if "hasface" in sample:
            # FIXME: They have no pose data. But this will fail if pose datasets ever get the hasface flag
            return sample
        sample["roi"] = self._create_roi(sample)
        return sample
