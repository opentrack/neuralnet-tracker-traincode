import glob
import os
from os.path import basename, join
import numpy as np
import cv2
import re
import scipy.io
from scipy.spatial.transform import Rotation
from copy import copy
import pickle
import math
import enum
import functools

import torch
from torch.utils.data import Dataset, DataLoader, Subset, IterableDataset, get_worker_info
import torch.nn.functional

import utils
import datasets.preprocessing as preprocessing


class PostprocessingDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        self.postprocess = kwargs.pop('postprocess')
        super(PostprocessingDataLoader, self).__init__(*args, **kwargs)
    
    def __iter__(self):
        for item in super(PostprocessingDataLoader, self).__iter__():
            yield self.postprocess(item)


def _maybe_apply(fun):
    def wrapper(sample, name, *args, **kwargs):
        if name in sample:
            sample[name] = fun(sample[name][None,...], *args, **kwargs)[0]
    return wrapper

maybe_normalize_point_data = _maybe_apply(utils.normalize_points)
maybe_normalize_box_data = _maybe_apply(utils.normalize_boxes)


def maybe_transform_point_data(sample, name, offset, scale):
    try:
        p = sample[name]
    except KeyError:
        return
    p[0] *= scale[0]
    p[1] *= scale[1]
    if p.shape[0]==3:
        p[2] *= scale[2]
    else:
        assert p.shape[0]==2
    p[0] += offset[0]
    p[1] += offset[1]


def maybe_transform_box_data(sample, name, offset, scale):
    try:
        x0, y0, x1, y1 = sample[name]
    except KeyError:
        return
    x0 = x0 * scale[0] + offset[0]
    x1 = x1 * scale[0] + offset[0]
    y0 = y0 * scale[1] + offset[1]
    y1 = y1 * scale[1] + offset[1]
    if scale[0] < 0.:
        x1, x0 = x0, x1
    if scale[1] < 0.:
        y1, y0 = y0, y1
    sample[name] = np.array([x0, y0, x1, y1])


class ApplyRoi(object):
    def __init__(self, padding_fraction=0.1, square=True):
        self.padding_fraction = padding_fraction
        self.square = square

    def __call__(self, sample):
        sample = copy(sample)
        image = sample['image']
        roi = sample['roi']
        image, o = preprocessing.extract_image_roi(image,roi,self.padding_fraction,self.square, return_offset=True)
        sample['image'] = image
        s = [ 1., 1., 1. ]
        maybe_transform_point_data(sample, 'coord', o, s)
        maybe_transform_point_data(sample, 'pt3d_68', o, s)
        maybe_transform_point_data(sample, 'pt2d_68', o, s)
        maybe_transform_box_data(sample, 'roi', o, s)
        return sample


class ApplyRoiRandomized(object):
    def __call__(self, sample):
        sample = copy(sample)
        image = sample['image']
        h, w = image.shape[:2]
        roi = sample['roi']
        roi_w = roi[2]-roi[0]
        roi_h = roi[3]-roi[1]

        scale = np.random.uniform(-0.05, 0.4)
        scaled_roi = preprocessing.extend_rect(roi, w, h, scale, 0.)
        scaled_roi = np.array(preprocessing.squarize_roi(scaled_roi))
        scaled_roi_w = scaled_roi[2]-scaled_roi[0]
        scaled_roi_h = scaled_roi[3]-scaled_roi[1]

        woffset = max(0, scaled_roi_w-roi_w)/2
        hoffset = max(0, scaled_roi_h-roi_h)/2

        scaled_roi[[0,2]] += np.random.uniform(-woffset,woffset)
        scaled_roi[[1,3]] += np.random.uniform(-hoffset,hoffset)

        image, o = preprocessing.extract_image_roi(image,scaled_roi,0,False, return_offset=True)
        sample['image'] = image
        s = [ 1., 1., 1. ]
        maybe_transform_point_data(sample, 'coord', o, s)
        maybe_transform_point_data(sample, 'pt3d_68', o, s)
        maybe_transform_point_data(sample, 'pt2d_68', o, s)
        maybe_transform_box_data(sample, 'roi', o, s)
        return sample


class Flip(object):
    @staticmethod
    def flip(sample):
        img = sample['image']
        h, w = img.shape[:2]
        sample['image'] = preprocessing.flip_image(img)
        if 'pose' in sample:
            sample['pose'] = preprocessing.flip_rot(sample['pose'])
        Flip.maybe_flip_keypoints(sample, 'pt3d_68', w)
        Flip.maybe_flip_keypoints(sample, 'pt2d_68', w)
        o = [ w, 0 ]
        s = [-1, 1, 1]
        maybe_transform_point_data(sample, 'coord', o, s)
        maybe_transform_box_data(sample, 'roi', o, s)

    @staticmethod
    def maybe_flip_keypoints(sample, name, width):
        if name in sample:
            sample[name] = preprocessing.flip_keypoints_68(width, sample[name])

    def __call__(self, sample):
        if np.random.randint(2):
            return sample
        else:
            self.flip(sample)
            return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        sample['image'] = cv2.resize(image, (new_w, new_h))

        s = [ new_w / w, new_h/h, 0.5*(new_w / w + new_h / h) ]
        o = [ 1., 1.]
        maybe_transform_point_data(sample, 'coord', o, s)
        maybe_transform_point_data(sample, 'pt3d_68', o, s)
        maybe_transform_point_data(sample, 'pt2d_68', o, s)
        maybe_transform_box_data(sample, 'roi', o, s)

        return sample


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        h, w = sample['image'].shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        sample['image'] = sample['image'][top: top + new_h,
                      left: left + new_w,...]

        s = [ 1., 1., 1. ]
        o = [ -left, -top ]
        maybe_transform_point_data(sample, 'coord', o, s)
        maybe_transform_point_data(sample, 'pt3d_68', o, s)
        maybe_transform_point_data(sample, 'pt2d_68', o, s)
        maybe_transform_box_data(sample, 'roi', o, s)

        return sample


class CenterCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        h, w = sample['image'].shape[:2]
        new_h, new_w = self.output_size

        top = (h - new_h)//2
        left = (w - new_w)//2

        sample['image'] = sample['image'][top: top + new_h,left: left + new_w,...]
        
        s = [ 1., 1., 1. ]
        o = [ -left, -top ]
        maybe_transform_point_data(sample, 'coord', o, s)
        maybe_transform_point_data(sample, 'pt3d_68', o, s)
        maybe_transform_point_data(sample, 'pt2d_68', o, s)
        maybe_transform_box_data(sample, 'roi', o, s)

        return sample


class RescaleMaintainAspect(object):
    '''
    Rescales the image while maintaining its aspect ratio.
    The larger dimension is cropped in order to make the
    result match the desired output size.
    '''
    def __init__(self, output_size):
        self.output_size = output_size
    
    def __call__(self, sample):
        h, w = sample['image'].shape[:2]
        new_h, new_w = self.output_size
        new_aspect = new_w/new_h
        if w/h >= new_aspect:
            w_crop = round(new_aspect*h)
            sample = CenterCrop((h, w_crop))(sample)
        else:
            h_crop = round(w / new_aspect)
            sample = CenterCrop((h_crop, w))(sample)
        return Rescale((new_h, new_w))(sample)


class ImageDistortion(object):
    def __init__(self, func):
        self.func = func
    
    def __call__(self, sample):
        sample['image'] = self.func(sample['image'])
        return sample


def faceboxes_style_color_distortion(image):
    # Adapted from https://github.com/zisianw/FaceBoxes.PyTorch/blob/master/data/data_augment.py
    # Increased max contrast from 1.5 to 2.
    
    assert image.dtype == np.uint8
    
    def _convert(image, alpha=1, beta=0):
        image[...] = np.clip(image * alpha + beta, 0., 255.9999).astype(np.uint8)

    image = image.copy()

    if np.random.randint(2):

        #brightness distortion
        if np.random.randint(2):
            _convert(image, beta=np.random.uniform(-32, 32))

        #contrast distortion
        if np.random.randint(2):
            _convert(image, alpha=np.random.uniform(0.5, 2.))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if np.random.randint(2):
            _convert(image[:, :, 1], alpha=np.random.uniform(0.5, 1.5))

        #hue distortion
        if np.random.randint(2):
            tmp = image[:, :, 0].astype(int) + np.random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:

        #brightness distortion
        if np.random.randint(2):
            _convert(image, beta=np.random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if np.random.randint(2):
            _convert(image[:, :, 1], alpha=np.random.uniform(0.5, 1.5))

        #hue distortion
        if np.random.randint(2):
            tmp = image[:, :, 0].astype(int) + np.random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        #contrast distortion
        if np.random.randint(2):
            _convert(image, alpha=np.random.uniform(0.5, 2.))

    return image


def blur_distortion(image):
    if isinstance(image, np.ndarray):
        assert len(image.shape)==3 # Single sample

        image = image.copy()

        if np.random.randint(2):
            image[...] = cv2.GaussianBlur(image,(3,3),0)

        stddev = np.random.choice([16., 8., 0.])
        if stddev > 0.:
            image[...] = np.clip(image + np.random.normal(0, stddev, size=image.shape, ), 0., 255.9999).astype(np.uint8)

    elif isinstance(image, torch.Tensor):
        with torch.no_grad():
            assert len(image.shape)==4 # Batch
            B, C, H, W = image.shape
            image = image.cuda()
            kernel = torch.from_numpy(np.array([
                [ 0., 1., 0.],
                [ 1., 4., 1.],
                [ 0., 1., 0.]
            ], np.float32))
            kernel /= torch.sum(kernel)
            kernel = kernel[None,None,...].repeat(C,1,1,1)
            kernel = kernel.cuda()
            noise = noise = torch.cuda.FloatTensor(C,H,W)
            for i in range(B):
                if np.random.randint(2):
                    image[i:i+1] = torch.nn.functional.conv2d(image[i:i+1], kernel, bias=None, padding=1, groups=C)
                stddev = np.random.choice([16./255, 8./255, 0.])
                if stddev>0:
                    noise.normal_(0., stddev)
                    image[i] += noise
            image = torch.clip(image, -0.5, 0.5)
            del kernel
            del noise
    return image


def _value_distortion(image):
    alpha = np.random.uniform(0.5, 1.5)
    beta = np.random.uniform(-16, 16)
    return np.clip(image.astype(np.float32) * np.float32(alpha) + np.float32(beta), np.float32(0.), np.float32(255.9999)).astype(np.uint8)


def ImageValueDistortion():
    return ImageDistortion(_value_distortion)


def ImageColorDistort():
    return ImageDistortion(faceboxes_style_color_distortion)


def BlurNoiseDistortion():
    return ImageDistortion(blur_distortion)


def FullImageDistortion():
    return ImageDistortion(lambda img: blur_distortion(faceboxes_style_color_distortion(img)))


class AdaptiveBrightnessContrastDistortion(object):
    def __call__(self, sample):
        img = sample['image']
        assert len(img.shape)==3 and img.shape[2]==3
        if 'roi' in sample:
            roi = sample['roi']
            if roi[0]>roi[2] and roi[1]>roi[2]:
                crop = preprocessing.extract_image_roi(img, roi, 0, False, False)
            else:
                crop = img
        else:
            crop = img
        crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        a, b = np.quantile(crop.ravel(), [0.2, 0.8])

        scale = np.random.uniform(min(1.,32./(a+1.e-6)), 255./(b+1))
        img = scale*img
        
        sample['image'] = np.clip(img, 0, 255).astype(np.uint8)
        
        return sample


class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __init__(self, 
        monochrome=True):
        self.monochrome = monochrome

    @staticmethod
    def _normalized_bool(x, smooth=0.):
        # Label smoothing!
        return np.array(1.0-smooth if x else smooth, dtype=np.float32)

    def __call__(self, sample):
        image = sample['image']
        if self.monochrome:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = image[...,None] # Add the channel back.
        image=  image/255.0 - 0.5
        sample['image'] = image.astype(np.float32)

        try:
            sample['pose'] = utils.convert_from_rot(sample['pose'])
        except KeyError:
            pass

        try:
            sample['hasface'] = self._normalized_bool(sample['hasface'], 0.1)
        except KeyError:
            pass

        shape = image.shape[:2]

        maybe_normalize_point_data(sample, 'coord', shape)
        maybe_normalize_point_data(sample, 'pt3d_68', shape)
        maybe_normalize_point_data(sample, 'pt2d_68', shape)
        maybe_normalize_box_data(sample, 'roi', shape)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def _do_image(self, img):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        assert len(img.shape)==3
        img = np.transpose(img, [2, 0, 1])
        return torch.from_numpy(img)

    def _do_other(self, data):
        return torch.from_numpy(data)

    def __call__(self, sample):
        sample = { k:(self._do_image(v) if k == 'image' else self._do_other(v))
            for k, v in sample.items() }
        return sample


class InjectZeroKeypoints3d(object):
    def __init__(self, dim=3):
        self.points = np.zeros((dim,68), np.float32)
    def __call__(self, sample):
        sample['pt3d_68'] = self.points.copy()
        sample['pt3d_68_enable'] = np.array(0.,dtype=np.float32)
        return sample


class InjectZeroPose(object):
    def __call__(self, sample):
        sample.update({
            'pose' : Rotation.identity(),
            'coord' : np.array([0.,0.,0.,], dtype=np.float32),
            'pose_enable' : np.array(0.,dtype=np.float32),
        })
        return sample


class InjectPoseEnable(object):
    def __call__(self, sample):
        sample['pose_enable'] = np.array(1., dtype=np.float32)
        return sample


class InjectPt3d68Enable(object):
    def __call__(self, sample):
        sample['pt3d_68_enable'] = np.array(1., dtype=np.float32)
        return sample