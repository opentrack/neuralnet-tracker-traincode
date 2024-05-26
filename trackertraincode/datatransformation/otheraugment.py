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
    transform_image_opencv, 
    transform_image_pil, 
    transform_image_torch,
    position_normalization,
    transform_keypoints)


class ScalingMode(enum.Enum):
    OPENCV_AREA = 1
    TORCH_GRID_SAMPLE_ALIGN_CORNERS = 2,
    TORCH_GRID_SAMPLE_NO_ALIGN_CORNERS = 3,
    PIL_HAMMING_WINDOW = 4


class RoiFocusRandomizationParameters(NamedTuple):
    scales : torch.Tensor # Shape B
    angles : Optional[torch.Tensor] # Shape B
    translations : torch.Tensor # Shape B2
    scaling_mode : ScalingMode


def RandomFocusRoi(new_size, roi_variable='roi', largeangles=False, insert_backtransform=False, align_corners=False):
    return GeneralFocusRoi(
        MakeRoiRandomizationParameters(largeangles, align_corners),
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
    def __init__(self, largeangles, align_corners):
        self.largeangles = largeangles
        self.align_corners = align_corners

    def __call__(self, B : tuple) -> RoiFocusRandomizationParameters:
        # In the original the scaling was more like 0.6 to 1.1
        # And the roi was moved to the edges of the image.
        # Then I used 0.8 to 1.1 for a while.
        scales = torch.randn(size=B).mul(0.1).clip(-0.5,0.5).add(1.1)
        translations = torch.randn(size=B+(2,)).mul(0.5).clip(-1., 1.)
        angles = (self._pick_angles_both_ways(B, 15.) if self.largeangles else self._pick_angles_uniform(B, 30.))
        if self.align_corners:
            scaling_mode = ScalingMode.TORCH_GRID_SAMPLE_ALIGN_CORNERS
        else:
            scaling_mode = [
                ScalingMode.OPENCV_AREA,
                ScalingMode.PIL_HAMMING_WINDOW,
                ScalingMode.TORCH_GRID_SAMPLE_NO_ALIGN_CORNERS
            ][np.random.randint(0,3)]
        return RoiFocusRandomizationParameters(
            scales = scales,
            angles = angles,
            translations = translations,
            scaling_mode = scaling_mode)

    def _pick_angles_uniform(self, B : tuple, deg : float):
        return random_uniform(B,-np.pi*deg/180.,np.pi*deg/180. )

    def _pick_large_angles(self, B: tuple):
        weights = torch.tensor([98,1,1], dtype=torch.float)
        angles = torch.tensor([0., 70., -70.], dtype=torch.float32)*np.pi/180.
        return random_choice(B, angles, weights, replacement=True)

    def _pick_angles_both_ways(self, B : tuple, deg : float):
        uniform_angles = self._pick_angles_uniform(B, deg)
        angles = self._pick_large_angles(B)
        return angles + uniform_angles


class NoRoiRandomization(object):
    def __init__(self, extent_factor):
        self.extent_factor = extent_factor

    def __call__(self, B) -> RoiFocusRandomizationParameters:
        return RoiFocusRandomizationParameters(
            scales = torch.full(B, self.extent_factor),
            angles = torch.zeros(B),
            translations = torch.zeros(B+(2,)),
            scaling_mode = ScalingMode.OPENCV_AREA)


class GeneralFocusRoi(object):
    def __init__(self, make_randomization_parameters, new_size, roi_variable, insert_backtransform):
        self.new_size = new_size
        self.roi_variable = roi_variable
        self.insert_backtransform = insert_backtransform
        self._max_beyond_border_shift = 0.3
        self.make_randomization_parameters = make_randomization_parameters


    def _compute_view_roi(self, face_roi : torch.Tensor, enlargement_factor : torch.Tensor, translation_distribution : torch.Tensor, beyond_border_shift : float):
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
        new_roi = torch.round(new_roi)
        return new_roi


    def _compute_point_transform_from_roi(self, B : Tuple[int], new_roi : torch.Tensor, new_size : int):
        return Affine2d.range_remap_2d(
            inmin = new_roi[...,:2],
            inmax = new_roi[...,2:],
            outmin = torch.zeros(B+(2,), device=new_roi.device, dtype=torch.float32),
            outmax = torch.full(B+(2,), fill_value = new_size, device=new_roi.device, dtype=torch.float32))


    def _compute_point_transform_from_params(self, B : Tuple[int], roi : torch.Tensor, params : RoiFocusRandomizationParameters, new_size : int, align_corners : bool):
        ones = torch.ones(B+(2,), device=roi.device, dtype=torch.float32)
        view_roi = self._compute_view_roi(roi, params.scales, params.translations, self._max_beyond_border_shift)
        tr_norm = Affine2d.range_remap_2d(
            inmin = view_roi[...,:2],
            inmax = view_roi[...,2:],
            outmin = -ones,
            outmax =  ones)
        tr_rot = Affine2d.trs(angles = params.angles)
        output_domain_size = new_size-1 if align_corners else new_size
        tr_denorm = Affine2d.range_remap(-1., 1., 0., output_domain_size).to(device=roi.device).repeat(B)
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

        if params.scaling_mode in (ScalingMode.OPENCV_AREA, ScalingMode.PIL_HAMMING_WINDOW):
            view_roi = self._compute_view_roi(roi, params.scales, params.translations, 0.3)
            tr = self._compute_point_transform_from_roi(B, view_roi, self.new_size)
            if params.scaling_mode == ScalingMode.OPENCV_AREA:
                transform_function = partial(transform_image_opencv, view_roi=view_roi, new_size=self.new_size)
            else:
                transform_function = partial(transform_image_pil, view_roi=view_roi, new_size=self.new_size)
        
        elif params.scaling_mode in (ScalingMode.TORCH_GRID_SAMPLE_ALIGN_CORNERS, ScalingMode.TORCH_GRID_SAMPLE_NO_ALIGN_CORNERS):
            align_corners=params.scaling_mode==ScalingMode.TORCH_GRID_SAMPLE_ALIGN_CORNERS
            tr = self._compute_point_transform_from_params(B, roi, params, self.new_size, align_corners=align_corners)
            transform_function = partial(transform_image_torch, tr=tr, new_size=self.new_size, align_corners=align_corners)

        for k, v in sample.items():
            c = get_category(sample, k)
            if c in imagelike_categories:
                sample[k] = transform_function(v, category=c)
            else:
                # WARNING `apply_affine2d` also applies to `equivariance_affine2d`! Hence, what
                # actually comes out of this function is (tr @ offset @ tr.inv()), which is what
                # we want. Because applied to the transformed original pose (tr @ pose), we get
                # tr @ offset @ pose, matching the transformed image `equivariance_image`.
                sample[k] = apply_affine2d(tr, k, v, c)

        if self.insert_backtransform:
            sample['image_backtransform'] = tr.inv().tensor()
            sample['image_original_size'] = torch.tensor((W, H), dtype=torch.int32)

        sample.meta._imagesize = self.new_size
        return sample


class PutRoiFromLandmarks(object):
    def __init__(self, extend_to_forehead = False):
        self.extend_to_forehead = extend_to_forehead

    def _create_roi(self, landmarks3d):
        is_tensor = isinstance(landmarks3d, torch.Tensor)
        if is_tensor:
            landmarks3d = landmarks3d.numpy()
        if self.extend_to_forehead:
            roi = preprocessing.head_bbox_from_keypoints(landmarks3d)
        else:
            min_ = np.amin(landmarks3d[...,:2], axis=-2)
            max_ = np.amax(landmarks3d[...,:2], axis=-2)
            roi = np.concatenate([min_, max_], axis=0).astype(np.float32)
        if is_tensor:
            roi = torch.from_numpy(roi)
        return roi

    def __call__(self, sample : Batch):
        if 'pt3d_68' in sample:
            roi = self._create_roi(sample['pt3d_68'])
            sample['roi'] = roi
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


def horizontal_flip_and_rot_90(align_corners : bool, p_rot : float, sample : Batch):
    assert sample.meta.batchsize == 0
    do_flip = np.random.randint(0,2) == 0
    rot_dir = np.random.choice([-1,0,1],p=[p_rot/2., (1.-p_rot), p_rot/2.])
    if not do_flip and rot_dir == 0:
        return sample
    sample = copy(sample)
    w, h = sample.meta.image_wh
    if align_corners:
        w, h = w-1, h-1
    
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
