import numpy as np
from copy import copy
from typing import Callable, Set, Sequence, Union, List, Tuple, Dict, Optional, NamedTuple
from functools import partial
import enum

import torch
from torch import Tensor
import torch.nn.functional as F

from trackertraincode.pipelines import Batch
from trackertraincode.datasets.batch import Metadata
from trackertraincode.datasets import preprocessing
from trackertraincode.neuralnets.affine2d import Affine2d
from trackertraincode.datasets.dshdf5pose import FieldCategory, imagelike_categories
from trackertraincode.neuralnets.math import random_choice, random_uniform
from trackertraincode.datatransformation.core import get_category

from trackertraincode.datatransformation.affinetrafo import (
    apply_affine2d, 
    croprescale_image_torch,
    croprescale_image_cv2,
    affine_transform_image_cv2,
    affine_transform_image_torch,
    position_normalization,
    position_unnormalization)
from trackertraincode.facemodel.bfm import ScaledBfmModule, BFMModel
from trackertraincode.neuralnets.modelcomponents import PosedDeformableHead

class RoiFocusRandomizationParameters(NamedTuple):
    scales : torch.Tensor # Shape B
    angles : torch.Tensor # Shape B
    translations : torch.Tensor # Shape (B, 2)


def RandomFocusRoi(new_size, roi_variable='roi', rotation_aug_angle : float = 30., extension_factor = 1.1, insert_backtransform=False):
    return GeneralFocusRoi(
        MakeRoiRandomizationParameters(rotation_aug_angle=rotation_aug_angle, extension_factor=extension_factor),
        new_size, 
        roi_variable,
        insert_backtransform)


def FocusRoi(new_size, extent_factor, roi_variable='roi', insert_backtransform=False):
    return GeneralFocusRoi(
        NoRoiRandomization(extent_factor),
        new_size, 
        roi_variable,
        insert_backtransform)


class MakeRoiRandomizationParameters(object):
    def __init__(self, rotation_aug_angle, extension_factor):
        self.rotation_aug_angle = rotation_aug_angle
        self.extension_factor = extension_factor

    def __call__(self, B : tuple) -> RoiFocusRandomizationParameters:
        scales = torch.randn(size=B).mul(0.1).clip(-0.5,0.5).add(self.extension_factor)
        translations = torch.randn(size=B+(2,)).mul(0.5).clip(-1., 1.)
        angles = self._pick_angles(B, self.rotation_aug_angle) if self.rotation_aug_angle else torch.zeros(size=B)
        return RoiFocusRandomizationParameters(
            scales = scales,
            angles = angles,
            translations = translations)

    def _pick_angles(self, B : tuple, angle : float):
        angles = torch.full(B, fill_value=np.pi*angle/180.)
        angles *= torch.from_numpy(np.random.choice([-1.,1.], size=B, replace=False))
        angles *= np.random.choice([0.,1.], size=B, replace=False, p = [2./3, 1./3])
        return angles


class NoRoiRandomization(object):
    def __init__(self, extent_factor):
        self.extent_factor = extent_factor

    def __call__(self, B) -> RoiFocusRandomizationParameters:
        return RoiFocusRandomizationParameters(
            scales = torch.full(B, self.extent_factor),
            angles = torch.zeros(B),
            translations = torch.zeros(B+(2,)))


class GeneralFocusRoi(object):
    def __init__(self, make_randomization_parameters, new_size, roi_variable, insert_backtransform):
        self.new_size = new_size
        self.roi_variable = roi_variable
        self.insert_backtransform = insert_backtransform
        self._max_beyond_border_shift = 0.3
        self.make_randomization_parameters = make_randomization_parameters


    def _compute_view_roi(self, face_roi : torch.Tensor, enlargement_factor : torch.Tensor, translation_distribution : torch.Tensor, beyond_border_shift : float):
        '''
        enlargement_factor: By how much the current roi is scaled up
        translation_distribution: Random number between -1 and 1 indicating the movement of the face roi within the expanded roi
        beyond_border_shift: Fraction of the original roi by which the face can be moved beyond the border of the expanded roi
        '''
        x0, y0, x1, y1 = torch.moveaxis(face_roi, -1, 0)
        rx, ry = translation_distribution.moveaxis(-1,0)
        roi_w = x1-x0
        roi_h = y1-y0
        cx = 0.5*(x1+x0)
        cy = 0.5*(y1+y0)
        size = torch.maximum(roi_w, roi_h)*enlargement_factor
        wiggle_room_x = F.relu(size-roi_w)
        wiggle_room_y = F.relu(size-roi_h)
        tx = (wiggle_room_x * 0.5 + roi_w * beyond_border_shift) * rx
        ty = (wiggle_room_y * 0.5 + roi_h * beyond_border_shift) * ry
        x0 = cx - size*0.5 + tx
        x1 = cx + size*0.5 + tx
        y0 = cy - size*0.5 + ty
        y1 = cy + size*0.5 + ty
        new_roi = torch.stack([x0, y0, x1, y1], dim=-1)
        new_roi = torch.round(new_roi).to(torch.int32)
        return new_roi


    def _compute_point_transform_from_roi(self, B : Tuple[int], new_roi : torch.Tensor, new_size : int):
        return Affine2d.range_remap_2d(
            inmin = new_roi[...,:2].to(torch.float32),
            inmax = new_roi[...,2:].to(torch.float32),
            outmin = torch.zeros(B+(2,), device=new_roi.device, dtype=torch.float32),
            outmax = torch.full(B+(2,), fill_value = new_size, device=new_roi.device, dtype=torch.float32))


    def _center_rotation_tr(self, rotations : Tensor):
        # TODO: Complete implementation for batches
        tr_norm = position_normalization(self.new_size, self.new_size).to(device=rotations.device)
        tr_rot = Affine2d.trs(angles = rotations)
        tr_denorm = position_unnormalization(self.new_size, self.new_size).to(device=rotations.device)
        return tr_denorm @ tr_rot @ tr_norm
    

    def _maybe_account_for_video(self, meta : Metadata, params : RoiFocusRandomizationParameters):
        # The simplest thing to do is overwrite the trafos of each sequence
        # with the trafo of the first frame. Thus we have augmentation and
        # stable frame crops around the heads.
        if meta.seq is None:
            return params
        for a, b in meta.sequence_start_end:
            params.translations[a:b,...] = params.translations[a:a+1,...]
            params.scales[a:b] = params.scales[a:a+1]
            if params.angles is not None:
                params.angles[a:b] = params.angles[a:a+1]
        return params


    def __call__(self, sample : Batch):
        W, H = sample.meta.image_wh
        B = sample.meta.prefixshape
        roi = sample[self.roi_variable]

        params : RoiFocusRandomizationParameters = self.make_randomization_parameters(B)

        self._maybe_account_for_video(sample.meta, params)

        view_roi = self._compute_view_roi(roi, params.scales, params.translations, self._max_beyond_border_shift)
        tr = self._compute_point_transform_from_roi(B, view_roi, self.new_size)
        tr = self._center_rotation_tr(params.angles) @ tr

        if params.angles.item() != 0.: 
            image_transform_function = lambda img: affine_transform_image_cv2(img, tr, self.new_size)
        else:
            image_transform_function = lambda img: croprescale_image_cv2(img, view_roi, self.new_size)

        for k, v in sample.items():
            c = get_category(sample, k)
            if c == FieldCategory.image:
                sample[k] = image_transform_function(v)
            else:
                sample[k] = apply_affine2d(tr, k, v, c)

        if self.insert_backtransform:
            sample['image_backtransform'] = tr.inv().tensor()
            sample['image_original_size'] = torch.tensor((W, H), dtype=torch.int32)

        sample.meta._imagesize = self.new_size
        return sample


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


class StabilizeRoi(object):
    def __init__(self, alpha=0.01, destination='roi'):
        self.roi_filter_alpha = alpha
        self.last_roi = None
        self.last_id = None
        self.destination = destination

    def filter_roi(self, sample):
        roi = sample['roi']
        id_ = sample['individual'] if 'individual' in sample else None
        if id_ == self.last_id and self.last_roi is not None:
            roi = self.roi_filter_alpha*roi + (1.-self.roi_filter_alpha)*self.last_roi
        #     print (f"Filt: {id_}")
        # else:
        #     print (f"Raw: {id_}")
        self.last_roi = roi
        self.last_id = id_
        return roi
        
    def __call__(self, batch):
        batch[self.destination] = self.filter_roi(batch)
        return batch


def horizontal_flip_and_rot_90(p_rot : float, sample : Batch):
    assert sample.meta.batchsize == 0
    do_flip = np.random.randint(0,2) == 0
    rot_dir = np.random.choice([-1,0,1],p=[p_rot/2., (1.-p_rot), p_rot/2.])
    if not do_flip and rot_dir == 0:
        return sample
    sample = copy(sample)
    w, h = sample.meta.image_wh
    tr = Affine2d.identity()
    if rot_dir != 0:
        tr = tr @ Affine2d.range_remap_2d([-1., -1.],[1., 1.], [0.,0.],[w, h]) @ Affine2d.trs(angles=torch.tensor(rot_dir*np.pi*0.5,dtype=torch.float32)) @ Affine2d.range_remap_2d([0.,0.],[w, h], [-1., -1.],[1., 1.]) 
    if do_flip:
        tr = tr @ Affine2d.range_remap_2d([0.,0.],[w, h], [w,0], [0, h])

    for k, v in sample.items():
        c = get_category(sample, k)
        if c in imagelike_categories:
            if do_flip:
                v = torch.flip(v,(-1,))
            if rot_dir != 0:
                v = v.swapaxes(-1, -2)
            if rot_dir == 1:
                v = torch.flip(v,(-1,))
            elif rot_dir == -1:
                v = torch.flip(v,(-2,))
            sample[k] = v
        else:
            sample[k] = apply_affine2d(tr, k, v, c)
    return sample
