from trackertraincode.facemodel.bfm import BFMModel, ScaledBfmModule
from trackertraincode.neuralnets.modelcomponents import PosedDeformableHead
from trackertraincode.pipelines import Batch

import torch


class PutRoiFromLandmarks(object):
    def __init__(self, extend_to_forehead = False):
        self.extend_to_forehead = extend_to_forehead
        self.headmodel = PosedDeformableHead(ScaledBfmModule(BFMModel()))

    def _create_roi(self, landmarks3d, sample):
        if self.extend_to_forehead:
            vertices = self.headmodel(
                sample['coord'],
                sample['pose'],
                sample['shapeparam'])
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