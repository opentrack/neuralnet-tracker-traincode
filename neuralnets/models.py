import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
import torchvision.models

from neuralnets.mobilenet_v1 import MobileNet
from neuralnets.modelcomponents import *


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
        
        self.boxstddev = SpatialMeanAndStd((7,9), half_size=1.5)

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
    
    def inference(self, x):
        assert not self.training
        pred = self.forward(x)
        pred = { 
            'hasface' : torch.sigmoid(pred[:,0]),
            'roi' : pred[:,1:] 
        }
        return pred


class MobilnetV1WithPointHead(nn.Module):
    def __init__(self, enable_point_head=True, point_head_dimensions=3):
        super(MobilnetV1WithPointHead, self).__init__()
        self.input_resolution = 129
        self.enable_point_head = enable_point_head
        self.num_eigvecs = 50
        # Load the keypoint data even if not used. That is to make
        # load_state_dict work with the strict=True option.
        self.keypts, self.keyeigvecs = load_deformable_head_keypoints(40, 10)
        self.point_head_dimensions = point_head_dimensions
        assert point_head_dimensions in (2,3)
        num_classes = 7+self.num_eigvecs+4
        self.convnet = MobileNet(num_classes=num_classes, input_channel=1, momentum=0.01, skipconnection=True, dropout=0.5)
        self.out = PoseOutputStage()

    def forward(self, x):
        assert x.shape[2] == self.input_resolution and \
               x.shape[3] == self.input_resolution

        x = self.convnet(x)
        coords = x[:,:3]
        quats = x[:,3:3+4]
        boxparams = x[:,3+4:3+4+4]
        kptweights = x[:,3+4+4:]

        roi_box = x.new_empty((x.size(0),4))
        boxsize = torch.exp(boxparams[:,2:])
        boxcenter = boxparams[:,:2]
        roi_box[:,:2] = boxcenter - boxsize
        roi_box[:,2:] = boxcenter + boxsize
        self.roi_box = roi_box
        self.roi_pred = roi_box

        x = self.out(torch.cat([coords, quats], dim=1))
        
        if self.enable_point_head:
            self.deformweights = kptweights
            local_keypts = self.keyeigvecs[None,...] * kptweights[:,:,None,None]
            local_keypts = torch.sum(local_keypts, dim=1)
            local_keypts += self.keypts[None,...]
            if self.point_head_dimensions == 2:
                self.pt3d_68 = self.out.headcenter_to_screen(local_keypts)
            else:
                self.pt3d_68 = self.out.headcenter_to_screen_3d(local_keypts)
            self.pt3d_68 = self.pt3d_68.transpose(1,2)
            assert self.pt3d_68.shape[1] == self.point_head_dimensions and self.pt3d_68.shape[2] == 68
            # Return in format (batch x dimensions x points)
        return x

    def inference(self, x):
        assert not self.training
        coords, quats = self.forward(x)
        pred = { 
            'pose' : quats,
            'coord' : coords,
            'roi' : self.roi_pred
        }
        if self.enable_point_head:
            pred.update({
                'pt3d_68' : self.pt3d_68
            })
        return pred