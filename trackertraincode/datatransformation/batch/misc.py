from trackertraincode.facemodel.bfm import BFMModel, ScaledBfmModule
from trackertraincode.neuralnets.modelcomponents import PosedDeformableHead
from trackertraincode.neuralnets.rotrepr import QuatRepr
from trackertraincode.pipelines import Batch

import torch


class PutRoiFromLandmarks(object):
    def __init__(self, extend_to_forehead = False):
        self.extend_to_forehead = extend_to_forehead
        self.headmodel = PosedDeformableHead(ScaledBfmModule(BFMModel()))

    def _create_roi(self, landmarks3d : torch.Tensor, sample):
        shapeparams = sample['shapeparam'] if 'shapeparams' in sample else landmarks3d.new_zeros((50,))
        if self.extend_to_forehead:
            vertices = self.headmodel(
                sample['coord'],
                QuatRepr(sample['pose']),
                shapeparams
                )
            min_ = torch.amin(vertices[...,:2], dim=-2)
            max_ = torch.amax(vertices[...,:2], dim=-2)
        else:
            min_ = torch.amin(landmarks3d[...,:2], dim=-2)
            max_ = torch.amax(landmarks3d[...,:2], dim=-2)
        roi = torch.cat([min_, max_], dim=0).to(torch.float32)
        return roi

    def __call__(self, sample : Batch):
        if 'pt3d_68' in sample:
            sample['roi'] = self._create_roi(sample['pt3d_68'], sample)
        return sample