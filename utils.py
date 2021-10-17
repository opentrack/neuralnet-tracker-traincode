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
    quat = rot.as_quat().astype(np.float32)
    # q and -q represent the same rotation. So we arbitrarily demand
    # the real component to be positive. Hopefully this prevents the
    # network from getting confused.
    if len(quat.shape) == 2:
        quat[quat[:,-1]<0] *= -1
    else:
        quat *= -1
    # TODO: When using this, maybe apply exp function to neural net output to force positivity
    return quat


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


def inv_rotation_conversion_from_hell(rot):
    x, y, z = rot.as_matrix().T
    x, y, z = np.array([z, -y, -x]).T
    m = np.array([z, -y, -x])
    p,y,r = Rotation.from_matrix(m).as_euler('XYZ')
    return np.asarray([-p,-y,-r])


def angle_errors(euler1, euler2):
    v1 = np.concatenate([np.cos(euler1)[...,None],np.sin(euler1)[...,None]],axis=-1)
    v2 = np.concatenate([np.cos(euler2)[...,None],np.sin(euler2)[...,None]],axis=-1)
    angles = np.arccos(np.sum(v1*v2, axis=-1))
    return angles


def affine3d_chain(Ta, Tb):
    Ra, ta = Ta
    Rb, tb = Tb
    return Ra*Rb, Ra.as_matrix().dot(tb) + ta


def affine3d_inv(Ta):
    Ra, ta = Ta
    RaInv = Ra.inv()
    return RaInv, -RaInv.as_matrix().dot(ta)


def iter_batched(iterable, batchsize):
    it = iter(iterable)
    while True:
        ret = [*zip(range(batchsize),it)]
        ret = [ x for _,x in ret ]
        if not ret:
            break
        yield ret


if __name__ == '__main__':
    R1 = Rotation.random()
    t1 = np.random.rand(3)
    inv = affine3d_inv((R1,t1))
    R2, t2 = affine3d_chain((R1,t1), inv)
    print(R2.as_matrix(),t2)