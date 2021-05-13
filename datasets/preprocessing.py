from os.path import dirname, join
import numpy as np
import math
import cv2
from contextlib import contextmanager
from scipy.spatial.transform import Rotation
import functools


def get_base_seed32(worker_info):
    '''
        Returns seed which is the same for all workers.
        This is an evil hack but there is no other way I'd know.
        With that we can make a random permutation which is the
        same for all workers and then each worker can pick
        its subset. Thus every sample is processed exactly once
        per epoch. That's the idea at least ...
    '''
    seed = worker_info.seed - worker_info.id
    seed = seed & ((2**32)-1)
    return seed


def imencode(img, quality=99):
    _, img = cv2.imencode('.JPEG', img,  (cv2.IMWRITE_JPEG_QUALITY, quality))
    return np.frombuffer(img, dtype='uint8')


def imdecode(blob, color=False):
    if isinstance(blob, bytes):
        blob = np.frombuffer(blob, dtype='B')
    img = cv2.imdecode(blob, cv2.IMREAD_COLOR if color else 0)
    if color == 'rgb':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imread(fn):
    img = cv2.imread(fn)
    assert img is not None, f"Failed to load image {fn}!"
    return img


class PinholeCam(object):
    def __init__(self, fov, w, h):
        self.fov = fov
        self.f = 1. / math.tan(fov*np.pi/180.*0.5)
        self.w = w
        self.h = h
        self.aspect = w/h

    def project_to_screen(self, p):
        # xscr / f = x / z
        # f = tan(fov * 0.5)
        x, y, z = p
        xscr = self.f * x / z
        yscr = self.f * y / z * self.aspect
        return xscr, yscr

    def project_to_image(self, p):
        x, y = self.project_to_screen(p)
        x = (x + 1.)*0.5
        y = (y + 1.)*0.5
        x *= self.w
        y *= self.h
        x, y = x, y
        return x, y

    def project_size_to_image(self, depth, scale):
        screen_space_scale = self.f * scale / depth
        return self.w * screen_space_scale * 0.5


def extend_rect(roi, w, h, padding_fraction, abs_padding):
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
    x0 = int(math.floor(x0))
    y0 = int(math.floor(y0))
    x1 = int(math.ceil(x1))
    y1 = int(math.ceil(y1))
    return (x0, y0, x1, y1)


def extract_image_roi(image, roi, padding_fraction, square=False, return_offset = False):
    # The returned offset is the vector to add to landmarks in order to 
    # make them match the returned image
    h, w = image.shape[:2]
    roi = extend_rect(roi, w, h, padding_fraction, 0)
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


def flip_keypoints_68(width, pts):
    flip_map = [
        16,
        15,
        14,
        13,
        12,
        11,
        10,
        9,
        8,
        7,
        6,
        5,
        4,
        3,
        2,
        1,
        0,
        26,
        25,
        24,
        23,
        22,
        21,
        20,
        19,
        18,
        17,
        27,
        28,
        29,
        30,
        35, # 31
        34,
        33,
        32,
        31,
        45, # 36
        44,
        43,
        42, # 39
        47,
        46, # 41
        39,
        38,
        37,
        36,
        41, 
        40, # 47
        54,
        53,
        52,
        51,
        50,
        49,
        48,
        59,
        58,
        57,
        56,
        55,
        64, # 60
        63,
        62,
        61,
        60,
        67, # 65
        66,
        65,
    ]
    assert pts.shape[0] in (2,3)
    pts = pts[:,flip_map].copy()
    pts[0,:] = width - pts[0,:]
    return pts


def flip_image(img):
    return np.ascontiguousarray(img[:,::-1,:])


def flip_rot(rot):
    m = rot.as_matrix()
    m[2,:] *= -1
    m[:,2] *= -1
    rot = Rotation.from_matrix(m)
    return rot


def rotation_conversion_from_hell(pitch, yaw, roll):
    # For AFLW and 300W-LP

    # This makes stuff make sense w.r.t to my visualization
    # and my euler convention.
    rot = Rotation.from_euler('XYZ', [pitch,-yaw,roll])
    M = rot.as_matrix()
    x, y, z = M
    M = np.array([z, -y, -x])
    x, y, z = M.T
    M = np.array([z, -y, -x]).T
    rot = Rotation.from_matrix(M)
    # Now at zero rotation, we have the local X/forward 
    # axis pointing to the viewer. Y goes up. And Z goes left.
    # It is a right-handed coordinate system.
    # The notion of pointing to the viewer depends on my visualization
    # ofc. Still it is what makes sense with a kinda-aerospace Euler
    # angle convention. I use 'YZX' with the scipy rotation lib. 
    # The YZX are the yaw pitch and roll angles. What I require from
    # this is:
    # Pitching up makes the X axes get a positive y component.
    # In the visualization the Y axis points up. Have to be careful here
    # because pixel coordinates are counted from top down.
    # Finally, rolling should leave the forward axis invariant.
    return rot


@functools.lru_cache(1)
def _load_shape_components():
    # With arbitrary scaling factors that match those used when creating the 
    # keypoint arrays from the original deformable model.
    # This scaling was done in order to bring typical shape parameters
    # into a reasonable range between -1 and 1
    path = join(dirname(__file__),'..','neuralnets')
    keypts = np.load(join(path,'face_keypoints_base_68_3d.npy'))
    w_shp = np.load(join(path,'./face_keypoints_base_68_3d_shp.npy'))
    w_exp = np.load(join(path,'./face_keypoints_base_68_3d_exp.npy'))
    return keypts, w_shp, w_exp


def get_3ddfa_shape_parameters(params):
    """ Modified for a subset of rescaled shape vectors. 
        Also restricted to the first 40 and 10 parameters, respectively."""
    f_shp = params['Shape_Para'][:40,0]/20./1.e5
    f_exp = params['Exp_Para'][:10,0]/5.
    return f_shp, f_exp


def compute_keypoints_from_3ddfa_shape_params(params, head_size, rotation, tx, ty):
    f_shp, f_exp = get_3ddfa_shape_parameters(params)
    keypts, w_shp, w_exp = _load_shape_components()
    pts3d = keypts + \
        np.sum(f_shp[:40,None,None]*w_shp, axis=0) + \
        np.sum(f_exp[:10,None,None]*w_exp, axis=0)
    pts3d *= head_size     
    pts3d = rotation.apply(pts3d)
    pts3d = pts3d[:,[2,1,0]]
    pts3d = pts3d.T
    pts3d[0] *= -1
    pts3d[1] *= -1
    pts3d[0] += tx
    pts3d[1] += ty
    return pts3d


def depth_centered_keypoints(kpts):
    eye_corner_indices = [45, 42, 39, 36]
    center = np.average(kpts[:,eye_corner_indices], axis=1)
    kpts = np.array(kpts, copy=True)
    kpts[2] -= center[2]
    return kpts


def move_aflw_head_center_to_between_eyes(coords, rot):
    offset_my_mangled_shape_data = np.array([-0.9, -0.26, 0.])
    offset = rot.apply(offset_my_mangled_shape_data)*coords[2]
    # world to screen transform:
    offset = offset[[2,1]]
    offset[0] *= -1
    offset[1] *= -1
    coords = np.array(coords, copy=True)
    coords[0:2] -= offset
    return coords


def extended_key_points_for_bounding_box(keypts):
    assert keypts.shape[1] in (2,3)
    eye_corner_indices = [45, 42, 39, 36]
    center = np.average(keypts[eye_corner_indices,:], axis=0)
    chin_idx = [ 7,8,9 ]
    chin = np.average(keypts[chin_idx,:], axis=0) - center
    eye_right_idx = [ 39, 36]
    eye_left_idx = [45, 42 ]
    right = 0.5*(np.average(keypts[eye_right_idx,:],axis=0) - np.average(keypts[eye_left_idx,:],axis=0))
    right = right / np.linalg.norm(right)
    jaw_right_idx = [0, 1, 2, 3]
    jaw_left_idx = [16,15,14,13]
    radius = np.linalg.norm(0.5*(np.average(keypts[jaw_right_idx,:],axis=0) - np.average(keypts[jaw_left_idx,:],axis=0)))
    #back = np.cross(right, chin/np.linalg.norm(chin))
    back = 0.5*(np.average(keypts[jaw_right_idx,:],axis=0) + np.average(keypts[jaw_left_idx,:],axis=0)) - center
    back = back / np.linalg.norm(back)
    head_top_distance_scale = 0.7
    width_scale = 0.7
    length_back_scale = 0.7
    side_points_back = 0.4
    pt1 = center -chin*head_top_distance_scale
    pt2 = pt1 + width_scale*radius*right + back*radius*side_points_back
    pt3 = pt1 - width_scale*radius*right + back*radius*side_points_back
    pt4 = pt1 + length_back_scale*np.linalg.norm(radius)*back
    extended_keypts = np.concatenate([keypts, pt1[None,...], pt2[None, ...], pt3[None, ...], pt4[None, ...]], axis=0)
    return extended_keypts