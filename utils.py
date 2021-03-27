import numpy as np
import enum
import pickle
from scipy.spatial.transform import Rotation


def as_hpb(rot):
    return rot.as_euler('YZX')


def from_hpb(hpb):
    return Rotation.from_euler('YZX', hpb)


def unnormalize_images(images):
    return ((images + 0.5)*255).astype(np.uint8)


rad2deg = 180./np.pi
deg2rad = np.pi/180.


def unnormalize_points(size_input, pts):
    h, w = size_input
    pts = pts.copy()
    pts[...,0,:] += 1.
    pts[...,1,:] += 1.
    pts[...,0,:] *= w*0.5
    pts[...,1,:] *= h*0.5
    if pts.shape[-2]==3:
        pts[...,2,:] *= 0.25*(h+w)
    else:
        assert pts.shape[-2]==2
    return pts


def unnormalize_coords(size_input, coords):
    h, w = size_input
    coords = coords.copy()
    coords[...,0] += 1.
    coords[...,0] *= w*0.5
    coords[...,1] += 1.
    coords[...,1] *= h*0.5
    coords[...,2] *= 0.25*(h+w)
    return coords


def normalize_points(pts, size_input):
    h, w = size_input
    out = np.empty_like(pts)
    out[:,0,...] = pts[:,0,...]/w*2.-1.
    out[:,1,...] = pts[:,1,...]/h*2.-1.
    if pts.shape[1]==3:
        out[:,2,...] = pts[:,2,...]/(w+h)*4.
    else:
        assert pts.shape[1]==2
    return out


def unnormalize_boxes(size_input, boxes):
    h, w = size_input
    boxes = np.array(boxes)
    out = np.empty_like(boxes)
    out[...,[0,2]] = w*(boxes[...,[0,2]]+1.)*0.5
    out[...,[1,3]] = h*(boxes[...,[1,3]]+1.)*0.5
    return out


def normalize_boxes(boxes, size_input):
    h, w = size_input
    boxes = np.array(boxes)
    out = np.empty_like(boxes)
    out[...,[0,2]] = boxes[...,[0,2]]/w*2.-1.
    out[...,[1,3]] = boxes[...,[1,3]]/h*2.-1.
    return out


def convert_to_rot(net_output):
    return Rotation.from_quat(net_output)


def convert_from_rot(rot):
    return rot.as_quat().astype(np.float32)


def undo_collate(batch):
    """
        Generator that takes apart a batch. Oposite of collate_fn.
    """
    assert isinstance(batch, dict), f"Got {type(batch)}"
    keys = [*batch.keys()]
    assert keys
    N = batch[keys[0]].shape[0]  # First dimension should be the batch size
    for i in range(N):
        yield { k:batch[k][i] for k in keys }