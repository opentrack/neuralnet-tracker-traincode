import itertools
import math
from typing import Union, Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models
from trackertraincode.neuralnets.math import inv_smoothclip0, smoothclip0
import trackertraincode.neuralnets.io

from trackertraincode.neuralnets.modelcomponents import (
    freeze_norm_stats, 
    rigid_transformation_25d, 
    DeformableHeadKeypoints, 
    CenterOfMassAndStd,
    LocalToGlobalCoordinateOffset,
    RotationRepr
)

import trackertraincode.neuralnets.negloglikelihood as NLL
from trackertraincode.neuralnets.rotrepr import Mat33Repr, QuatRepr
import trackertraincode.neuralnets.torchquaternion as torchquaternion
from trackertraincode.backbones.mobilenet_v1 import MobileNet
from trackertraincode.backbones.efficientnet import EfficientNetBackbone
from trackertraincode.backbones.resnet import resnet18
from trackertraincode.backbones.hybrid_vit import HybridVitBackbone


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
        if self.enable_uncertainty:
            # pointscales = NLL.FeaturesAsUncorrelatedVariance(num_features, 68, torch.full((68,)))
            # shapescales = NLL.FeaturesAsUncorrelatedVariance(num_features, 50, torch.full((50,)))
            self.point_distrib_scales = NLL.DiagonalScaleParameter(68)
            self.shape_distrib_scales = NLL.DiagonalScaleParameter(50)

    def forward(self, z, quats, coords) -> Dict[str, Tensor]:
        shapeparam = self.shapenet(z)
        pt3d_68 = rigid_transformation_25d(
            quats,
            coords[...,:2],
            coords[...,2:],
            self.deformablekeypoints(shapeparam))
        
        out = { 'pt3d_68' : pt3d_68, 'shapeparam' : shapeparam }
        
        if self.enable_uncertainty:
            out.update({ 'pt3d_68_scales' : self.point_distrib_scales()[None,:,None].expand_as(pt3d_68) })
            out.update({ 'shapeparam_scales' : self.shape_distrib_scales()[None,:].expand_as(shapeparam) })

        return out


class DirectQuaternionWithNormalization(nn.Module):
    def __init__(self, num_features, enable_uncertainty = False):
        super().__init__()
        self.enable_uncertainty = enable_uncertainty
        self.linear = nn.Linear(num_features, 4, bias=True)
        self.linear.bias.data[torchquaternion.iw] = inv_smoothclip0(torch.as_tensor(0.1))
        if enable_uncertainty:
            self.uncertainty_net = NLL.FeaturesAsTriangularScale(
                num_features, 3)
    
    def forward(self, x) -> Dict[str, Tensor]:
        z = self.linear(x)
        quats, quats_unnormalized = QuatRepr.from_features(z)
        out = {
            'unnormalized_quat' : quats_unnormalized,
            'rot' : quats, 
        }
        if self.enable_uncertainty:
            scales = self.uncertainty_net(x)
            out.update({
                'pose_scales_tril' : scales,
            })
        return out


class RotRepr6dWithNormalization(nn.Module):
    def __init__(self, num_features, enable_uncertainty = False):
        super().__init__()
        self.enable_uncertainty = enable_uncertainty
        self.linear = nn.Linear(num_features, 6, bias=True)
        # Bias toward identity
        self.linear.bias.data[...] = 0.001*torch.as_tensor([1.,0.,0.,0.,1.,0.])
        if enable_uncertainty:
            self.uncertainty_net = NLL.FeaturesAsTriangularScale(
                num_features, 3)
    
    def forward(self, x) -> Dict[str, Tensor]:
        z = self.linear(x)
        mrot = Mat33Repr.from_6drepr_features(z)
        out = {
            'unnormalized_6drepr' : z,
            'rot' : mrot
        }
        if self.enable_uncertainty:
            scales = self.uncertainty_net(x)
            out.update({
                'pose_scales_tril' : scales,
            })
        return out


class BoundingBox(nn.Module):
    def __init__(self, num_features, enable_uncertainty = False):
        super(BoundingBox, self).__init__()
        self.enable_uncertainty = enable_uncertainty
        self.linear = nn.Linear(num_features, 4)
        self.linear.bias.data[...] = torch.tensor([0.0, 0.0, 0.5, 0.5])
        if enable_uncertainty:
            self.scales = NLL.DiagonalScaleParameter(4)

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
            self.scales = NLL.FeaturesAsTriangularScale(num_features, 3)

    def forward(self, x : Tensor):
        coord = torch.cat([
            self.linear_xy(x),
            smoothclip0(self.linear_size(x))
        ], dim=-1)
        out = { 'coord' : coord }
        if self.enable_uncertainty:
            out.update({'coord_scales' : self.scales(x) })
        return out


def create_pose_estimator_backbone(num_heads, config : str, args : Dict[str,Any]):
    if config == 'mobilenetv1':
        return MobileNet(input_channel=1, num_classes=None, **args)
    elif config == 'resnet18':
        return resnet18(**args)
    elif config == 'hybrid_vit':
        if args:
            print (f"WARNING: backbone arguments to {config} ignored: {args}")
        return HybridVitBackbone(num_heads=num_heads)
    elif config.startswith('efficientnet_'):
        kind = config[len('efficientnet_'):]
        assert kind in 'b0,b1,b2,b3,b4'.split(',')
        return EfficientNetBackbone(kind=kind, input_channels=1, stochastic_depth_prob=0.1, **args)
    else:
        assert f"Unsupported backbone {config}"


class TransformerNeck(nn.Module):
    def __init__(self, num_heads, args : Dict[str,Any]):
        super().__init__()
        self.num_heads = num_heads
        assert not args, "Has no parameters"

    def forward(self, features : Tensor)  -> tuple[Tensor]:
        B, N, C = features.shape
        assert N == self.num_heads
        return features.unbind(dim=1)


class CnnNeck(nn.Module):
    def __init__(self, num_heads, args : Dict[str,Any]):
        super().__init__()
        self.num_heads = num_heads
        self.dropout_prob = args.get('dropout_prob', 0.5)
        self.dropout = nn.Dropout(self.dropout_prob) if self.dropout_prob>0. else nn.Identity()
    
    def forward(self, features : Tensor) -> tuple[Tensor]:
        B, C = features.shape
        return features[:,None,:].expand(-1,self.num_heads,-1).unbind(dim=1)



class NetworkWithPointHead(nn.Module):
    NUM_DATASET_CONSTANTS = 8
    def __init__(
            self, 
            enable_point_head=True, 
            enable_face_detector=False, 
            config='mobilenetv1', 
            enable_uncertainty=False,
            dropout_prob = None, # For loading old models. Ignored because they all used 0.5.
            use_local_pose_offset = True,
            backbone_args = None,
            enable_6drot = False):
        super(NetworkWithPointHead, self).__init__()
        
        assert dropout_prob is None or dropout_prob in (0.,0.5)

        self.enable_point_head = enable_point_head
        self.enable_face_detector = enable_face_detector
        self.finetune = False
        self.config = config
        self.enable_uncertainty = enable_uncertainty
        self.use_local_pose_offset = use_local_pose_offset
        self.enable_6drot = enable_6drot
        self._backbone_args = {} if (backbone_args is None) else backbone_args
        self._input_resolution = (129,)
        num_heads = 3 + (1 if enable_point_head else 0) + (1 if enable_face_detector else 0)

        self.convnet = create_pose_estimator_backbone(num_heads, config, self._backbone_args)
        num_features = self.convnet.num_features

        if config == 'hybrid_vit':
            self.neck = TransformerNeck(num_heads, self._backbone_args)
        else:
            self.neck = CnnNeck(num_heads, self._backbone_args)

        self.boxnet = BoundingBox(num_features, enable_uncertainty)
        self.posnet = PositionSizeOutput(num_features, enable_uncertainty)
        if self.enable_6drot:
            self.quatnet = RotRepr6dWithNormalization(num_features, enable_uncertainty)
        else:
            self.quatnet = DirectQuaternionWithNormalization(num_features, enable_uncertainty)
        self.local_pose_offset = LocalToGlobalCoordinateOffset(self.NUM_DATASET_CONSTANTS)
        self.local_pose_offset_kpts = LocalToGlobalCoordinateOffset(self.NUM_DATASET_CONSTANTS)
        if enable_point_head:
            self.landmarks = Landmarks3dOutput(num_features, enable_uncertainty)
        if enable_face_detector:
            self.face_detector = nn.Linear(num_features, 1, bias=True)

    def get_config(self):
        return {
            'enable_point_head' : self.enable_point_head,
            'enable_face_detector' : self.enable_face_detector,
            'config' : self.config,
            'enable_uncertainty' : self.enable_uncertainty,
            'use_local_pose_offset' : self.use_local_pose_offset,
            'backbone_args' : self._backbone_args,
            'enable_6drot' : self.enable_6drot
        }

    @property
    def input_resolutions(self) -> Tuple[int]:
        return self._input_resolution if isinstance(self._input_resolution,tuple) else (self._input_resolution,)

    @property
    def input_resolution(self) -> int:
        return self._input_resolution[0] if isinstance(self._input_resolution,tuple) else self._input_resolution

    @property
    def name(self) -> str:
        return type(self).__name__+'_'+self.config

    def forward(self, x : Tensor, coord_convention_id : Tensor | None = None):
        assert x.shape[2] in self.input_resolutions and \
               x.shape[3] == x.shape[2]
        
        x, _ = self.convnet(x)
        zs : list[Tensor] = list(self.neck(x))
        del x

        out : Dict[str,Tensor] = self.boxnet(zs.pop())

        out.update(self.posnet(zs.pop()))
        out.update(self.quatnet(zs.pop()))

        if self.use_local_pose_offset:
            out.update({
                'hidden_rot' : out['rot'],
                'hidden_coord' : out['coord']
            })
            rots, coords = self.local_pose_offset(out.pop('rot'), out.pop('coord'), set_id=coord_convention_id)
            out.update({
                'rot' : rots,
                'coord' : coords
            })
        
        rots, coords = out['rot'], out['coord']

        if self.enable_point_head:
            if self.use_local_pose_offset:
                rots, coords = self.local_pose_offset_kpts(out['hidden_rot'], out['hidden_coord'], set_id=coord_convention_id)
            out.update(self.landmarks(zs.pop(), rots, coords))

        if self.enable_face_detector:
            hasface_logits = self.face_detector(zs.pop()).view(x.size(0))
            out.update({'hasface_logits' : hasface_logits, 'hasface' : torch.sigmoid(hasface_logits) })

        out.pop('hidden_rot', None)
        out.pop('hidden_coord', None)
        if not self.training:
            out['pose'] = out['rot'].as_quat()
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


save_model = trackertraincode.neuralnets.io.save_model


def load_model(filename : str):
    def load_legacy(filename : str):
        sd = torch.load(filename)
        net = NetworkWithPointHead(
            enable_point_head=True,
            enable_face_detector=False,
            config='resnet18',
            enable_uncertainty=True,
            backbone_args = {'use_blurpool' : False}
        )
        net.load_state_dict(sd, strict=True)
    try:
        return trackertraincode.neuralnets.io.load_model(filename, [NetworkWithPointHead])
    except trackertraincode.neuralnets.io.InvalidFileFormatError as e:
        print (f"Failed to load model because: {str(e)}. Will attempt to load legacy config")
        return load_legacy(filename)