import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
import torchvision.models

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


class PretrainedNetwork(nn.Module):
    def __init__(self, num_classes):
        super(PretrainedNetwork, self).__init__()
        net = torchvision.models.mnasnet1_0(pretrained=True)
        self.backbone = nn.Sequential(
            *[*net.children()][:-1][0])
        def set_momentum(m):
            if 'batchnorm' in m.__class__.__name__.lower():
                m.momentum = 0.01
        self.backbone.apply(set_momentum)
        self.drop = nn.Dropout(0.5)
        self.dense = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = x.repeat(1,3,1,1) # Expand grayscale image to 3 channels
        z = self.backbone(x)
        z = torch.mean(z, dim=[2,3])
        z = self.drop(z)
        y = self.dense(z)
        return y
    
    def train(self, mode: bool = True):
        super(PretrainedNetwork, self).train(mode)
        if mode:
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if 'batchnorm' in m.__class__.__name__.lower():
                    #print (f"Layer {str(m)} set to eval mode!")
                    m.eval()
            # Heard the recommendation to keep the statistics of BN layers
            # during fine tuning. Do this only for the lower layers which do
            # basic feature recognition. The rationale is that higher level
            # layers must be changed radically since we do a problem very
            # different from image net classification which the network was
            # pretrained on.
            self.backbone[:9].apply(set_bn_eval)

class NetworkWithPointHead(nn.Module):
    def __init__(self, enable_point_head=True, point_head_dimensions=3):
        super(NetworkWithPointHead, self).__init__()
        self.input_resolution = 129
        self.enable_point_head = enable_point_head
        self.deformablekeypoints = DeformableHeadKeypoints()
        # Load the keypoint data even if not used. That is to make
        # load_state_dict work with the strict=True option.
        self.point_head_dimensions = point_head_dimensions
        assert point_head_dimensions in (2,3)
        num_classes = 7+self.deformablekeypoints.num_eigvecs+4
        self.convnet = PretrainedNetwork(num_classes)
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
            local_keypts = self.deformablekeypoints.forward(kptweights)
            self.pt3d_68 = self.out.headcenter_to_screen_3d(local_keypts)
            if self.point_head_dimensions == 2:
                self.pt3d_68 = self.pt3d_68[:,:,:2]
            self.pt3d_68 = self.pt3d_68.transpose(1,2)
            self.kptweights = kptweights
            assert self.pt3d_68.shape[1] == self.point_head_dimensions and self.pt3d_68.shape[2] == 68
            # Stores keypoint in format (batch x dimensions x points)
        
        # TODO: Return a dict
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