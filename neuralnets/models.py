import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

import neuralnets.torchquaternion
import torchvision.models
from neuralnets.modelcomponents import *
from neuralnets.mobilenet_v1 import MobileNet
from neuralnets.mobilefacenet import MobileFaceNet

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
        
        self.boxstddev = CenterOfMassAndStd((7,9), half_size=1.5)

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


def replace_batchnorm(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_batchnorm(module)
        if isinstance(module, nn.BatchNorm2d):
            setattr(model, n, GBN(module.num_features))


def parameter_list_with_decaying_learning_rate(m : nn.Module, bottom_lr, top_lr):
    # Note: Parameters are enumerated from top to bottom layers
    params = [ *m.parameters() ]
    factor = (top_lr/bottom_lr)**(1./(len(params)-1))
    return [
        { 'params' : [ p ], 'lr' : bottom_lr*factor**i } for i,p in enumerate(params)
    ]


class PretrainedNetwork(nn.Module):
    def __init__(self):
        super(PretrainedNetwork, self).__init__()
        net = torchvision.models.mnasnet1_0(pretrained=True)
        self.backbone = nn.Sequential(
            *[*net.children()][:-1][0])
        replace_batchnorm(self.backbone)

    def forward(self, x):
        x = x.repeat(1,3,1,1) # Expand grayscale image to 3 channels
        z = self.backbone(x)
        z = torch.mean(z, dim=[2,3])
        return z


class RegNetHead(nn.Module):
    # Stupid fix for RegNet backbone outputting unnormalized
    # values because BN is applied before addition of skip
    # connection.
    def __init__(self, n):
        # self.num_features = n
        # bottleneck_size = n
        super(RegNetHead,self).__init__()
        # self.res = nn.Sequential(
        #     nn.Linear(n, bottleneck_size, bias=False),
        #     nn.BatchNorm1d(bottleneck_size),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(bottleneck_size,n)
        # )
        # self.bn = nn.BatchNorm1d(n)
        # self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        assert x.size(2) == x.size(3) == 5
        x = torch.mean(x, dim=(2,3))
        return x
        #return self.act(self.bn(x + self.res(x)))

class RegNetBackbone(nn.Module):
    def __init__(self):
        import pycls.models
        super(RegNetBackbone, self).__init__()
        self.widen = nn.Conv2d(1, 3, 3, padding=1, bias=False)
        net = pycls.models.regnety("600MF", pretrained=True)
        self.backbone = nn.Sequential(*[*net.children()][:-1])
        # 608 For the 600MF version
        # 440 For the 400MF version
        self.head = RegNetHead(608) 
        #replace_batchnorm(self.backbone)

    def get_slower_finetune_parameters(self):
        slowest = [ self.widen] + [*self.backbone.children()][:3]
        slowest = [*filter(lambda p: p.requires_grad, itertools.chain.from_iterable(m.parameters() for m in slowest))]
        slow    = [*self.backbone.children()][3:5]
        slow    = [*filter(lambda p: p.requires_grad, itertools.chain.from_iterable(m.parameters() for m in slow))]
        return slowest, slow

    def forward(self, x):
        # Expand grayscale image to 3 channels
        x = self.widen(x) + x.repeat(1,3,1,1)
        return self.head(self.backbone(x))


class NetworkWithPointHead(nn.Module):
    def __init__(self, enable_point_head=True, point_head_dimensions=3, enable_face_detector=False, enable_full_head_box=False):
        super(NetworkWithPointHead, self).__init__()
        self.enable_point_head = enable_point_head
        self.enable_face_detector = enable_face_detector
        self.enable_full_head_box = enable_full_head_box

        self.convnet = MobileNet(0)
        self.input_resolution = 129
        num_features = 1024
        self.drop = nn.Dropout(p = 0.5)

        # self.convnet = RegNetBackbone()
        # num_features = 608
        # self.input_resolution = 129
        # self.drop = nn.Dropout(p = 0.5)

        # num_features = 512
        # self.input_resolution = 112
        # self.convnet = MobileFaceNet((self.input_resolution,self.input_resolution), 1, embedding_size = num_features)
        # self.drop = nn.Identity()

        self.boxnet = nn.Sequential(
            nn.Linear(num_features, 4, bias=True),
            BoundingBox()
        )
        if enable_full_head_box:
            self.full_head_boxnet = nn.Sequential(
                nn.Linear(num_features, 4, bias=True),
                BoundingBox()
            )
        self.posnet = nn.Linear(num_features, 3, bias=True)
        self.quatnet = nn.Linear(num_features, 4, bias=True)

        if enable_point_head:
            self.deformablekeypoints = DeformableHeadKeypoints()
            # Load the keypoint data even if not used. That is to make
            # load_state_dict work with the strict=True option.
            self.point_head_dimensions = point_head_dimensions
            assert point_head_dimensions in (2,3)
            self.shapenet = nn.Linear(num_features, self.deformablekeypoints.num_eigvecs, bias=True)
        if enable_face_detector:
            self.face_detector = nn.Linear(num_features, 1, bias=True)

    def forward(self, x):
        assert x.shape[2] == self.input_resolution and \
               x.shape[3] == self.input_resolution

        x = self.convnet(x)
        x = self.drop(x)

        roi_box = self.boxnet(x)
        
        coords = self.posnet(x).clone()
        coords[:,2] = F.softplus(coords[:,2].clone())

        quats_unnormalized = self.quatnet(x).clone()
        quats_unnormalized[:,3] = F.softplus(quats_unnormalized[:,3].clone())

        quats = neuralnets.torchquaternion.normalized(quats_unnormalized)

        out = { 'pose' : quats, 'coord' : coords, 'roi' : roi_box, 'unnormalized_quat' : quats_unnormalized }
        if self.enable_point_head:
            kptweights = self.shapenet(x)
            local_keypts = self.deformablekeypoints.forward(kptweights)
            pt3d_68 = rigid_transformation_25d(
                quats,
                coords[:,:2],
                coords[:,2:],
                local_keypts)
            if self.point_head_dimensions == 2:
                pt3d_68 = pt3d_68[:,:,:2]
            pt3d_68 = pt3d_68.transpose(1,2)
            assert pt3d_68.shape[1] == self.point_head_dimensions and pt3d_68.shape[2] == 68
            # Stores keypoint in format (batch x dimensions x points)
            out.update({'pt3d_68' : pt3d_68, 'shapeparam' : kptweights })
        if self.enable_face_detector:
            hasface_logits = self.face_detector(x).view(x.size(0))
            out.update({'hasface_logits' : hasface_logits, 'hasface' : torch.sigmoid(hasface_logits) })
        if self.enable_full_head_box:
            roi_box_full_head = self.full_head_boxnet(x)
            out.update({'roi_head' : roi_box_full_head })

        return out


class FinetuningNetworkWrapper(nn.Module):
    """
    Has an extra local transform to offset the pose from the prediction of the inner model.
    Also handles freezing batch-norm statistics, decreasing learning rate towards input layers.
    """
    def __init__(self, inner_model):
        super(FinetuningNetworkWrapper, self).__init__()
        self.inner_model = inner_model
        self.transform = SimilarityTransform()
        self.input_resolution = self.inner_model.input_resolution
    
    def forward(self, x):
        d = self.inner_model(x)
        q, c = self.transform(d['pose'], d['coord'])
        d['pose'] = q
        d['coord'] = c
        return d
    
    def train(self, mode=True):
        super(FinetuningNetworkWrapper, self).train(mode)
        if mode:
            self.apply(self._freeze_norm_stats)
    
    @staticmethod
    def _freeze_norm_stats(m):
        if isinstance(m, (GBN, nn.BatchNorm2d, nn.BatchNorm1d)):
            #print (f"Freezing {str(m)}")
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
    
    def get_finetune_parameters(self, base_lr):
        return parameter_list_with_decaying_learning_rate(self.inner_model, 0.01*base_lr, base_lr) +\
            [ { 'params' : [ *self.transform.parameters() ] } ]


class LocalAttentionNetwork(nn.Module):
    def __init__(self, enable_face_detector=False, enable_full_head_box=False):
        super(LocalAttentionNetwork, self).__init__()
        self.enable_face_detector = enable_face_detector
        self.enable_full_head_box = enable_full_head_box

        self.convnet = MobileNet(0, widen_factor=0.75, return_only_featuremap=True)
        self.input_resolution = 129
        num_features = 768
        self.featuremap_size = 5
        self.drop = nn.Dropout(p = 0.2)

        self.attention_conv = nn.Conv2d(num_features, 1, 1)
        self.attention_com = CenterOfMass((self.featuremap_size,self.featuremap_size), half_size=1.)

        # self.convnet = RegNetBackbone()
        # num_features = 608
        # self.input_resolution = 129
        # self.drop = nn.Dropout(p = 0.5)

        # num_features = 512
        # self.input_resolution = 112
        # self.convnet = MobileFaceNet((self.input_resolution,self.input_resolution), 1, embedding_size = num_features)
        # self.drop = nn.Identity()

        self.boxnet = nn.Sequential(
            nn.Linear(num_features, 4, bias=True),
            BoundingBox()
        )
        if enable_full_head_box:
            self.full_head_boxnet = nn.Sequential(
                nn.Linear(num_features, 4, bias=True),
                BoundingBox()
            )
        self.posnet = nn.Linear(num_features, 3, bias=True)
        self.quatnet = nn.Linear(num_features, 4, bias=True)

        if enable_face_detector:
            self.face_detector = nn.Linear(num_features, 1, bias=True)

    def normalized_attention_logits(self, x):
        return torch.softmax(x.view(x.shape[0], -1), dim=1).view(*x.shape)

    def forward(self, x):
        assert x.shape[2] == self.input_resolution and \
               x.shape[3] == self.input_resolution

        x = self.convnet(x)
        assert x.shape[2:4] == (self.featuremap_size,self.featuremap_size), f"Bad featuremap size {x.shape} vs expected {self.featuremap_size}"

        a_logits = self.attention_conv(x)
        a = self.normalized_attention_logits(a_logits)
        
        x = torch.mean(a*x, dim=(2,3))

        x = self.drop(x)

        roi_box = self.boxnet(x).clone()
        coords = self.posnet(x).clone()
        coords[:,2] = F.softplus(coords[:,2].clone())
        mean = self.attention_com(a[:,0,:,:])
        coords[:,:2] = coords[:,:2].clone() + mean
        roi_box[:,0] += mean[:,0]
        roi_box[:,1] += mean[:,1]
        roi_box[:,2] += mean[:,0]
        roi_box[:,3] += mean[:,1]

        quats_unnormalized = self.quatnet(x).clone()
        quats_unnormalized[:,3] = F.softplus(quats_unnormalized[:,3].clone())
        quats = neuralnets.torchquaternion.normalized(quats_unnormalized)

        out = { 
            'pose' : quats, 'coord' : coords, 'roi' : roi_box, 
            'unnormalized_quat' : quats_unnormalized,
            'attention_logits' : a_logits
        }
        if self.enable_face_detector:
            hasface_logits = self.face_detector(x).view(x.size(0))
            out.update({'hasface_logits' : hasface_logits, 'hasface' : torch.sigmoid(hasface_logits) })
        if self.enable_full_head_box:
            roi_box_full_head = self.full_head_boxnet(x)
            out.update({'roi_head' : roi_box_full_head })

        return out