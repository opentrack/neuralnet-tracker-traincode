import itertools
import math
from typing import Union, Optional, List, Tuple, Dict

import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models
from trackertraincode.neuralnets.math import inv_smoothclip0, smoothclip0

from trackertraincode.neuralnets.modelcomponents import (
    freeze_norm_stats, 
    rigid_transformation_25d, 
    DeformableHeadKeypoints, 
    CenterOfMassAndStd,
    LocalToGlobalCoordinateOffset
)

import trackertraincode.neuralnets.negloglikelihood as NLL
import trackertraincode.neuralnets.torchquaternion as torchquaternion
from trackertraincode.backbones.mobilenet_v1 import MobileNet
from trackertraincode.backbones.efficientnet import EfficientNetBackbone
from trackertraincode.backbones.resnet import resnet18


class LocalizerNet(nn.Module):
    def __init__(self):
        super(LocalizerNet, self).__init__()
        self.input_resolution = (224,288)  # H x W

        IR = torchvision.models.mnasnet._InvertedResidual

        self.initial_stage = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        def depthwise_separable_convolution(in_, out, kernel, **kwargs):
            momentum = kwargs.pop("momentum", 0.001)
            return nn.Sequential(
                nn.Conv2d(in_, in_, kernel, groups=in_, bias=False, **kwargs),
                nn.BatchNorm2d(in_, momentum=momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_, out, 1, bias=False),
                nn.BatchNorm2d(out, momentum=momentum)
            )

        self.convnet = nn.Sequential(
            self.initial_stage,
            
            depthwise_separable_convolution(8, 8, 3, padding=1),

            IR(in_ch=8, out_ch=12, kernel_size=3, stride=2, expansion_factor=2),
            IR(in_ch=12, out_ch=12, kernel_size=3, stride=1, expansion_factor=2),

            IR(in_ch=12, out_ch=20, kernel_size=3, stride=2, expansion_factor=4),
            IR(in_ch=20, out_ch=20, kernel_size=3, stride=1, expansion_factor=4),
            IR(in_ch=20, out_ch=20, kernel_size=3, stride=1, expansion_factor=4),
           
            IR(in_ch=20, out_ch=32, kernel_size=5, stride=2, expansion_factor=2),
            IR(in_ch=32, out_ch=32, kernel_size=5, stride=1, expansion_factor=2),
            IR(in_ch=32, out_ch=32, kernel_size=3, stride=1, expansion_factor=2),
            IR(in_ch=32, out_ch=32, kernel_size=3, stride=1, expansion_factor=2),

            IR(in_ch=32, out_ch=56, kernel_size=3, stride=2, expansion_factor=2),
            IR(in_ch=56, out_ch=56, kernel_size=3, stride=1, expansion_factor=2),
            IR(in_ch=56, out_ch=56, kernel_size=3, stride=1, expansion_factor=2),
            
            nn.Conv2d(56, 2, 1, bias=True)
        )
        
        self.boxstddev = CenterOfMassAndStd(half_size=1.5)

    def forward(self, x):
        assert x.shape[2] == self.input_resolution[0] and \
               x.shape[3] == self.input_resolution[1]

        x = self.convnet(x)
        a = torch.mean(x[:,0,...], dim=[1,2])
        x = x[:,1,...]
        x = torch.softmax(x.view(x.shape[0], -1), dim=1).view(*x.shape)
        self.attentionmap = x
        mean, std = self.boxstddev(self.attentionmap)
        pred = x.new_empty((x.shape[0],5))
        pred[:,0] = a[:]
        pred[:,1:3] = mean - std
        pred[:,3:5] = mean + std
        return pred
        # FIXME: return dict
    
    def inference(self, x):
        assert not self.training
        pred = self.forward(x)
        pred = { 
            'hasface' : torch.sigmoid(pred[:,0]),
            'roi' : pred[:,1:] 
        }
        return pred


class Landmarks3dOutput(nn.Module):
    def __init__(self, num_features, enable_uncertainty=False):
        super().__init__()
        self.enable_uncertainty = enable_uncertainty
        self.deformablekeypoints = DeformableHeadKeypoints(40, 10)
        self.shapenet = nn.Linear(num_features, self.deformablekeypoints.num_eigvecs)
        self.scales = NLL.UncorrelatedVarianceParameter(68, torch.full((68,), fill_value=0.3))

    def forward(self, z, quats, coords) -> Dict[str, Tensor]:
        shapeparam = self.shapenet(z)
        pt3d_68 = rigid_transformation_25d(
            quats,
            coords[...,:2],
            coords[...,2:],
            self.deformablekeypoints(shapeparam))
        
        out = { 'pt3d_68' : pt3d_68, 'shapeparam' : shapeparam }
        
        if self.enable_uncertainty:
            out.update({ 'pt3d_68_scales' : self.scales()[None,:,None].expand_as(pt3d_68) })

        return out


class DirectQuaternionWithNormalization(nn.Module):
    def __init__(self, num_features, enable_uncertainty = False):
        super().__init__()
        self.enable_uncertainty = enable_uncertainty
        self.linear = nn.Linear(num_features, 4, bias=True)
        self.linear.bias.data[torchquaternion.iw] = inv_smoothclip0(0.1)
        if enable_uncertainty:
            covfactor = NLL.FeaturesAsTriangularCovFactor(3)
            self.uncertainty_net = nn.Sequential(
                nn.Linear(num_features, covfactor.num_features, bias=False),
                covfactor)
    
    def forward(self, x) -> Dict[str, Tensor]:
        z = self.linear(x)
        quats_unnormalized = torch.empty_like(z)
        quats_unnormalized[...,torchquaternion.iw] = smoothclip0(z[...,torchquaternion.iw])
        quats_unnormalized[...,torchquaternion.iijk] = z[...,torchquaternion.iijk]
        quats = torchquaternion.normalized(quats_unnormalized)
        out = {
            'unnormalized_quat' : quats_unnormalized,
            'pose' : quats, 
        }
        if self.enable_uncertainty:
            scales = self.uncertainty_net(x)
            out.update({
                'pose_scales_tril' : scales
            })
        return out


class BoundingBox(nn.Module):
    def __init__(self, num_features, enable_uncertainty = False):
        super(BoundingBox, self).__init__()
        self.enable_uncertainty = enable_uncertainty
        self.linear = nn.Linear(num_features, 4)
        self.linear.bias.data[...] = torch.tensor([0.0, 0.0, 0.5, 0.5])
        if enable_uncertainty:
            self.scales = NLL.UncorrelatedVarianceParameter(4, torch.full((4,),fill_value=0.1))

    def forward(self, x : Tensor) -> Dict[str, Tensor]:
        z = self.linear(x)
        # Force the size parameter to be positive
        boxsize = z[...,2:].clone()
        boxsize = smoothclip0(boxsize)
        boxcenter = z[...,:2]
        box = torch.cat([ boxcenter - boxsize, boxcenter + boxsize ], dim=-1)
        out = { 'roi' : box }
        if self.enable_uncertainty:
            scales = self.scales()[None,:].expand_as(z)
            out.update({
                'roi_scales' : scales
            })
        return out


class PositionSizeOutput(nn.Module):
    def __init__(self, num_features, enable_uncertainty = False):
        super().__init__()
        self.enable_uncertainty = enable_uncertainty
        self.linear_xy = nn.Linear(num_features, 2)
        self.linear_size = nn.Linear(num_features, 1)
        self.linear_size.bias.data.fill_(0.5)
        if enable_uncertainty:
            self.scales = nn.Sequential(
                nn.Linear(num_features, 4, bias=False),
                NLL.FeaturesAsUncorrelatedVariance(3, torch.tensor([0.5,0.5,0.5])))
    
    def forward(self, x : Tensor):
        coord = torch.cat([
            self.linear_xy(x),
            smoothclip0(self.linear_size(x))
        ], dim=-1)
        out = { 'coord' : coord }
        if self.enable_uncertainty:
            out.update({'coord_scales' : self.scales(x) })
        return out


def create_pose_estimator_backbone(config : str):
    if config == 'mobilenetv1':
        return MobileNet(input_channel=1, num_classes=None)
    elif config == 'resnet18':
        return resnet18()
    elif config.startswith('efficientnet_'):
        kind = config[len('efficientnet_'):]
        assert kind in 'b0,b1,b2,b3,b4'.split(',')
        return EfficientNetBackbone(kind=kind, input_channels=1, stochastic_depth_prob=0.1)
    else:
        assert f"Unsupported backbone {config}"


class NetworkWithPointHead(nn.Module):
    def __init__(
            self, enable_point_head=True, 
            enable_face_detector=False, 
            use_local_pose_offset=True, 
            config='mobilenetv1', 
            enable_uncertainty=False):
        super(NetworkWithPointHead, self).__init__()
        self.enable_point_head = enable_point_head
        self.enable_face_detector = enable_face_detector
        self.use_local_pose_offset = use_local_pose_offset
        self.finetune = False
        self.config = config
        self.enable_uncertainty = enable_uncertainty

        self._input_resolution = (129, 97)

        self.convnet = create_pose_estimator_backbone(config)
        num_features = self.convnet.num_features
        self.dropout = nn.Dropout(0.1)

        self.boxnet = BoundingBox(num_features, enable_uncertainty)
        self.posnet = PositionSizeOutput(num_features, enable_uncertainty)
        self.quatnet = DirectQuaternionWithNormalization(num_features, enable_uncertainty)
        self.local_pose_offset = LocalToGlobalCoordinateOffset()
        self.local_pose_offset_kpts = LocalToGlobalCoordinateOffset()
        if enable_point_head:
            self.landmarks = Landmarks3dOutput(num_features, enable_uncertainty)
        if enable_face_detector:
            self.face_detector = nn.Linear(num_features, 1, bias=True)

    @property
    def input_resolutions(self) -> Tuple[int]:
        return self._input_resolution if isinstance(self._input_resolution,tuple) else (self._input_resolution,)

    @property
    def input_resolution(self) -> int:
        return self._input_resolution[0] if isinstance(self._input_resolution,tuple) else self._input_resolution

    @property
    def name(self) -> str:
        return type(self).__name__+'_'+self.config


    def load_partial(self, state_dict):
        mine = self.state_dict()
        assert (not frozenset(state_dict.keys()).difference(frozenset(mine.keys()))), f"Failed to load model dict. Keys {frozenset(state_dict.keys()).difference(frozenset(mine.keys()))} not found in present model"
        mine.update(state_dict)
        self.load_state_dict(mine)


    def forward(self, x):
        assert x.shape[2] in self.input_resolutions and \
               x.shape[3] == x.shape[2]

        x, _ = self.convnet(x)
        x = self.dropout(x)

        out : Dict[str,Tensor] = self.boxnet(x)

        out.update(self.posnet(x))
        out.update(self.quatnet(x))

        if self.use_local_pose_offset:
            out.update({
                'hidden_pose' : out['pose'],
                'hidden_coord' : out['coord']
            })
            quats, coords = self.local_pose_offset(out.pop('pose'), out.pop('coord'))
            out.update({
                'pose' : quats,
                'coord' : coords
            })
        
        quats, coords = out['pose'], out['coord']

        if self.enable_point_head:
            if self.use_local_pose_offset:
                quats, coords = self.local_pose_offset_kpts(out['hidden_pose'], out['hidden_coord'])
            out.update(self.landmarks(x, quats, coords))

        if self.enable_face_detector:
            hasface_logits = self.face_detector(x).view(x.size(0))
            out.update({'hasface_logits' : hasface_logits, 'hasface' : torch.sigmoid(hasface_logits) })

        out.pop('hidden_pose', None)
        out.pop('hidden_coord', None)

        return out


    def prepare_finetune(self):
        '''
        Returns parameters with lr rate scaling
        '''
        self.finetune = True
        paramgroups = [ list(c.parameters()) for c in sum((list(c.children()) for c in self.convnet.children()),[]) ]
        paramgroups += \
            [ list(frozenset(self.parameters()) - frozenset(sum(paramgroups, []))) ]
        return paramgroups


    def train(self, mode=True):
        super().train(mode)
        if mode and self.finetune:
            self.convnet.apply(freeze_norm_stats)


