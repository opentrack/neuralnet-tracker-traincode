import numpy as np
import cv2
import functools
import enum
import imghdr
import io
from PIL import Image
from numpy.typing import NDArray
from typing import Union, Tuple

from trackertraincode.facemodel.bfm import BFMModel


class ImageFormat(enum.IntEnum):
    JPG = 1
    PNG = 2


def which_image_format(buffer : np.ndarray) -> ImageFormat:
    f = io.BytesIO(buffer.tobytes())
    kind = imghdr.what(f)
    return {
        'jpeg' : ImageFormat.JPG,
        'png' : ImageFormat.PNG
    }[kind]


def imencode(img : NDArray[np.uint8], format = ImageFormat.JPG, quality=None) -> np.ndarray:
    '''
    Convert image to JPEG. Ingests numpy arrays. Returns numpy array of uint8's.
    '''
    cv_format = {
        ImageFormat.JPG : '.JPEG',
        ImageFormat.PNG : '.PNG'
    }[format]
    assert format == ImageFormat.JPG or quality is None
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if format == ImageFormat.JPG:
        quality = 99 if quality is None else quality
        _, img = cv2.imencode(cv_format, img,  (cv2.IMWRITE_JPEG_QUALITY, quality))
    else:
        _, img = cv2.imencode(cv_format, img)
    return np.frombuffer(img, dtype='uint8')


def imdecode(blob, color=False):
    '''
        color arguments:
            False -> single channel grayscale
            True -> rgb
    '''
    if isinstance(blob, bytes):
        blob = np.frombuffer(blob, dtype='B')
    img = cv2.imdecode(blob, cv2.IMREAD_COLOR if color else 0)
    assert img is not None
    if color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imread(fn):
    img = cv2.imread(fn)
    assert img is not None, f"Failed to load image {fn}!"
    return img


def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def imrescale(img : Union[NDArray[np.uint8],Image.Image], factor : float):
    '''Rescales the original size by a factor.

    PIL Images are resampled with the HAMMING window. Numpy arrays are upsampled with
    Opencv's bilinear interpolation and downsampled with the area mode.
    '''
    h, w = img.shape[:2] if isinstance(img,np.ndarray) else (img.height, img.width)
    new_w = round(w * factor)
    new_h = round(h * factor)
    if isinstance(img, np.ndarray):
        out : NDArray[np.uint8] = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if factor<1. else cv2.INTER_BILINEAR)
        return out
    elif isinstance(img, Image.Image):
        return img.resize((new_w,new_h),resample=Image.HAMMING, reducing_gap=3.)
    assert False, "Unsupported input"


def imshape(img : Union[NDArray[np.uint8],Image.Image]) -> Tuple[int,int]:
    '''Beware returns (height, width) according to numpy conventions'''
    assert isinstance(img, Image.Image) or (len(img.shape)<=3 and img.shape[2] in (1,3))
    return tuple(map(int,img.shape[:2])) if isinstance(img,np.ndarray) else (img.height, img.width)


def extend_rect(roi, padding_fraction, abs_padding):
    x0, y0, x1, y1 = roi
    roi_w = x1-x0
    roi_h = y1-y0
    border = max(roi_w,roi_h)*padding_fraction + abs_padding
    x0b = x0 - border
    x1b = x1 + border
    y0b = y0 - border
    y1b = y1 + border
    return np.array([x0b, y0b, x1b, y1b])


def squarize_roi(roi, crop=False):
    x0, y0, x1, y1 = roi
    roi_w = x1-x0
    roi_h = y1-y0
    cx = 0.5*(x1+x0)
    cy = 0.5*(y1+y0)
    roi_h = roi_w = (min(roi_w, roi_h) if crop else max(roi_w, roi_h))
    x0 = cx - roi_w*0.5
    x1 = cx + roi_w*0.5
    y0 = cy - roi_h*0.5
    y1 = cy + roi_h*0.5
    return (x0, y0, x1, y1)


def compute_padding(roi, w, h):
    x0, y0, x1, y1 = roi
    assert all(isinstance(v, int) for v in roi)
    return max(
        max(-x0, 0),
        max(-y0, 0),
        max(x1-w, 0),
        max(y1-h, 0)
    )


def roi_to_ints(roi):
    x0, y0, x1, y1 = roi
    # This code has to preserve the property that width==height if present.
    roi_w = round(x1-x0)
    roi_h = round(y1-y0)
    x0 = round(x0)
    y0 = round(y0)
    return (x0, y0, x0+roi_w, y0+roi_h)


def extract_image_roi(image, roi, padding_fraction, square=False, return_offset = False):
    '''
    The returned offset is the vector to add to landmarks in order to 
    make them match the returned image
    '''
    h, w = image.shape[:2]
    roi = extend_rect(roi, padding_fraction, 0)
    offset = np.array([0., 0.])
    if square:
        roi = squarize_roi(roi)
    roi = roi_to_ints(roi)
    padding = compute_padding(roi, w, h)
    if padding > 0:
        image = cv2.copyMakeBorder(
            image, padding, padding, padding, padding, 
            cv2.BORDER_CONSTANT, value=(0,0,0))
        roi = tuple((v+padding) for v in roi)
        offset[0] = padding
        offset[1] = padding
    x0, y0, x1, y1 = roi
    image = np.ascontiguousarray(image[y0:y1,x0:x1,...])
    offset[0] -= x0
    offset[1] -= y0
    if return_offset:
        return image, offset
    else:
        return image


@functools.lru_cache(1)
def load_shape_components():
    bfm = BFMModel()
    keypts = bfm.scaled_vertices[bfm.keypoints,:]
    w_exp = bfm.scaled_exp_base[:,bfm.keypoints,:]
    w_shp = bfm.scaled_shp_base[:,bfm.keypoints,:]
    return keypts, w_shp, w_exp


def get_3ddfa_shape_parameters(params):
    """ Modified for a subset of rescaled shape vectors. 
        Also restricted to the first 40 and 10 parameters, respectively."""
    f_shp = params['Shape_Para'][:40,0]/20./1.e5
    f_exp = params['Exp_Para'][:10,0]/5.
    return f_shp, f_exp


def compute_keypoints(f_shp, f_exp, head_size, rotation, tx, ty):
    keypts, w_shp, w_exp = load_shape_components()
    pts3d = keypts + \
        np.sum(f_shp[:40,None,None]*w_shp, axis=0) + \
        np.sum(f_exp[:10,None,None]*w_exp, axis=0)
    pts3d *= head_size     
    pts3d = rotation.apply(pts3d)
    pts3d = pts3d.T
    pts3d[0] += tx
    pts3d[1] += ty
    return pts3d


def sanity_check_landmarks(coord, rotation, pt3d_68, params=None, reltol=0.4, img = None):
    if params is None:
        f_shp, f_exp = np.zeros((40,1)), np.zeros((10,1))
    else:
        f_shp, f_exp = params
    expected = compute_keypoints(
        f_shp, f_exp,
        coord[2],
        rotation,
        coord[0],coord[1])
    if not np.allclose(expected, pt3d_68, rtol=0., atol=coord[2]*reltol):
        print ("Large deviation between base shape and point labels detected. Check for coordinate flips. Click the window away if the sample is fine.")
        import matplotlib.pyplot as pyplot
        from mpl_toolkits.mplot3d import Axes3D
        fig = pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*pt3d_68, cmap='rainbow', c=np.linspace(0.,1.,68), label='labels')
        ax.scatter(*expected, cmap='rainbow', c=np.linspace(0.,1.,68), marker='x', label='model')
        ax.legend()
        if img is not None:
            ax = fig.add_axes([0.,0., 0.33, 0.33])
            ax.imshow(img)
        pyplot.show()


def depth_centered_keypoints(kpts):
    eye_corner_indices = [45, 42, 39, 36]
    center = np.average(kpts[:,eye_corner_indices], axis=1)
    kpts = np.array(kpts, copy=True)
    kpts[2] -= center[2]
    return kpts


def move_aflw_head_center_to_between_eyes(coords, rot):
    offset_my_mangled_shape_data = np.array([0., -0.26, -0.9])
    offset = rot.apply(offset_my_mangled_shape_data)*coords[2]
    coords = np.array(coords, copy=True)
    coords[0:2] += offset[:2]
    return coords


def head_bbox_from_keypoints(keypts):
    assert keypts.shape[-2] == 68
    assert keypts.shape[-1] in (2,3), f"Bad shape {keypts.shape}"
    eye_corner_indices = [45, 42, 39, 36]
    jaw_right_idx = [0, 1]
    jaw_left_idx = [16,15]
    chin_idx = [ 7,8,9 ]
    point_between_eyes = np.average(keypts[...,eye_corner_indices,:], axis=-2)
    upvec = point_between_eyes - np.average(keypts[...,chin_idx,:], axis=-2)
    upvec /= np.linalg.norm(upvec)
    jaw_center_point = np.average(keypts[...,jaw_right_idx + jaw_left_idx,:],axis=-2)
    radius = np.linalg.norm(jaw_center_point - point_between_eyes, axis=-1, keepdims=True)
    center = 0.5*(jaw_center_point + point_between_eyes) + 0.25*radius*upvec
    def mkpoint(cx, cy):
        return center[...,:2] +  radius * np.array([cx, cy])
    corners = np.asarray([ mkpoint(cx,cy) for cx,cy in [
        (-1,1), (1,1), (1,-1), (-1,-1)
    ] ]).swapaxes(0,-2)
    allpoints = np.concatenate([keypts[...,:2], corners], axis=-2)
    min_ = np.amin(allpoints[...,:2], axis=-2)
    max_ = np.amax(allpoints[...,:2], axis=-2)
    roi = np.concatenate([min_, max_], axis=-1).astype(np.float32)
    return roi


# Adapted from 
# https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
def box_iou(box1, box2):
    '''Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Args:
      box1: (tensor) bounding boxes, sized [...,4].
      box2: (tensor) bounding boxes, sized [...,4].
    Return:
      (tensor) iou, sized [N,M].
    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    shape1 = box1.shape[:-1]
    shape2 = box2.shape[:-1]
    box1 = np.reshape(box1, (-1,4))
    box2 = np.reshape(box2, (-1,4))

    lt = np.maximum(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = np.minimum(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = np.maximum(rb-lt, 0)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
    area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
    iou = inter / (area1[:,None] + area2 - inter)
    return np.reshape(iou,shape1+shape2)


if __name__ == '__main__':
    head_bbox_from_keypoints(np.random.rand(68,3))
    head_bbox_from_keypoints(np.random.rand(7,68,3))