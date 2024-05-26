from os.path import splitext
from typing import List,Any,Dict
import os
import numpy as np
import enum
import pickle
import collections
from scipy.spatial.transform import Rotation
import h5py
import fnmatch

def identity(arg):
    return arg

def as_hpb(rot):
    return rot.as_euler('YXZ')


def from_hpb(hpb):
    return Rotation.from_euler('YXZ', hpb)


rad2deg = 180./np.pi
deg2rad = np.pi/180.


def convert_to_rot(net_output):
    return Rotation.from_quat(net_output)


def aflw_rotation_conversion(pitch, yaw, roll) -> Rotation:
    '''Euler angles to Rotation objects, used for AFLW and 300W-LP data'''
    # It's the result of endless trial and error. Don't ask ...
    # Euler angles suck.
    rot : Rotation = Rotation.from_euler('XYZ', [pitch,-yaw,roll])
    M = rot.as_matrix()
    P = np.asarray([
        [ 1, 0, 0 ],
        [ 0, 1, 0 ],
        [ 0, 0, -1 ]
    ])
    M = P @ M @ P.T
    rot = Rotation.from_matrix(M)
    return rot


def inv_aflw_rotation_conversion(rot : Rotation):
    '''Rotation object to Euler angles for AFLW and 300W-LP data'''
    # Flip x-axis
    x, y, z = rot.as_matrix().T
    x, y, z = np.array([-x, y, z]).T
    m = np.array([-x, y, z])
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
    if isinstance(iterable, (h5py.Dataset, np.ndarray)):
        for i in range(0, iterable.shape[0], batchsize):
            yield iterable[i:i+batchsize,...]
    else:
        it = iter(iterable)
        while True:
            ret = [ x for _,x in zip(range(batchsize),it) ]
            if not ret:
                break
            yield ret


def cycle(iterable):
    # itertools.cycle does unfortunately store the outputs of the iterator.
    # See another poor soul who ran into this https://github.com/pytorch/pytorch/issues/23900
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def replace_ext(filename, replacement):
    basename, _ = splitext(filename)
    return basename+replacement


def num_workers():
    return int(os.environ.get('NUM_WORKERS', 4))


def copy_attributes(src : h5py.HLObject, dst : h5py.HLObject):
    for k,v in src.attrs.items():
        dst.attrs[k] = v


def iter_hdf_datasets(x : h5py.HLObject):
    '''Creates iterator over contained datasets.'''
    if isinstance(x,h5py.Group):
        for v in x.values():
            yield from iter_hdf_datasets(v)
    else:
        yield x

        
def glob_hdf_datasets(f : h5py.File, patterns : List[str]):
    '''Creates iterator over contained datasets whose names match any of the patterns.
    
    Patterns are matched using the fnmatch library.
    '''
    it = iter_hdf_datasets(f)
    matcher = lambda ds: any(fnmatch.fnmatch(ds.name, pattern) for pattern in patterns)
    yield from filter(matcher, it)


def list_of_dicts_to_dict_of_lists(lod : List[Dict[Any,Any]]) -> Dict[Any, List[Any]]:
    if not lod:
        return {}
    first = next(iter(lod))
    return { k:[items[k] for items in lod] for k in first.keys() }

# TODO: Tests for this and the aflw rotation conversion
if __name__ == '__main__':
    R1 = Rotation.random()
    t1 = np.random.rand(3)
    inv = affine3d_inv((R1,t1))
    R2, t2 = affine3d_chain((R1,t1), inv)
    print(R2.as_matrix(),t2)