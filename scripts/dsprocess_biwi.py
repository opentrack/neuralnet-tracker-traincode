import os
import sys
import zipfile
import numpy as np
from os.path import join, dirname, basename, splitext
import re
import math
import tqdm
import tarfile
from scipy.spatial.transform import Rotation
import scipy.io
import cv2
import trackertraincode.utils as utils
import h5py
import argparse
import io
from collections import defaultdict
from zipfile import ZipFile
from typing import Tuple

from trackertraincode.datasets.preprocessing import imdecode, imencode, extract_image_roi
from trackertraincode.datasets.dshdf5pose import create_pose_dataset, FieldCategory

C = FieldCategory


PROJ_FOV = 65.
HEAD_SIZE_MM = 100.
PREFIX='faces_0/'


def get_pose_from_mat(f):
    lines = f.readlines()
    matrix = np.array([[*map(float,row.split(' ')[:3])] for row in lines[:3]])
    rot = Rotation.from_matrix(matrix)
    pos = np.array([*map(float, lines[4].split(' ')[:3])])
    return rot, pos


def get_camera_extrinsics(zf : ZipFile, fn) -> Tuple[Rotation,np.ndarray]:
    with io.StringIO(zf.read(fn).decode('ascii')) as f:
        lines = f.readlines()
    intrin, intrin, intrin, _, _, _, m1, m2, m3, _, pos, _, res = lines
    matrix = np.array([[*map(float,row.split(' ')[:3])] for row in [m1, m2, m3]])
    rot = Rotation.from_matrix(matrix)
    pos = np.array([*map(float, pos.split(' ')[:3])])
    return rot, pos


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


def compute_rotation_to_vector(pos):
    # Computes a rotation which aligns 
    # the x axes with the given head position.
    z = pos / np.linalg.norm(pos)
    x = np.cross([0.,1.,0.],z)
    y = -np.cross(x, z)
    M = np.array([x,y,z]).T
    rot = Rotation.from_matrix(M)
    return rot


def apply_local_head_origin_offset(rot, sz, offset):
    offset = rot.apply(offset)*sz
    # world to screen transform:
    offset = offset[:2]
    return offset


def find_image_file_names(zf : ZipFile):
    """
        Returns dict of filename lists. Keys are person numbers.
    """
    regex = re.compile(PREFIX+r'(\d\d)/frame_(\d\d\d\d\d)_rgb.png')
    samples = defaultdict(list)
    for f in zf.filelist:
        m = regex.match(f.orig_filename)
        if m is None:
            continue
        person = int(m.group(1))
        frame = m.group(2)
        samples[person].append((frame, f.orig_filename))
    for k, v in samples.items():
        # Sort by frame number then discard frame number
        v = sorted(v, key = lambda t: t[0])
        samples[k] = [ fn for (_,fn) in v ]
    return samples


def find_cal_files(zf : ZipFile):
    regex = re.compile(PREFIX+r'(\d\d)/rgb.cal')
    cal_files = {}
    for f in zf.filelist:
        m = regex.match(f.orig_filename)
        if m is None:
            continue
        person = int(m.group(1))
        cal_files[person] = f.orig_filename
    return cal_files



def read_data(zf, imagefile, cal):
    posefile = imagefile[:-len('_rgb.png')]+'_pose.txt'

    imgbuffer = zf.read(imagefile)
    img = imdecode(imgbuffer, True) # BGR
    h, w, _ = img.shape

    with io.StringIO(zf.read(posefile).decode('ascii')) as f:
        rot, pos = get_pose_from_mat(f)
    
    cam_inv = cal #utils.affine3d_inv(cal)
    rot, pos = utils.affine3d_chain(cam_inv, (rot, pos))

    rot_correction = compute_rotation_to_vector(pos)
    rot = rot_correction.inv() * rot

    cam = PinholeCam(PROJ_FOV, w, h)
    x, y = cam.project_to_image(pos)
    size = cam.project_size_to_image(pos[2], HEAD_SIZE_MM)
    roi = np.array([x-size, y-size, x+size, y+size])

    img, offset = extract_image_roi(img, roi, 0.5, return_offset=True)
    roi[[0,1]] += offset
    roi[[2,3]] += offset
    x += offset[0]
    y += offset[1]

    # Offset in local frame is given as argument.
    # It was found by eyemeasure. It could perhaps be improved by optimizing it during the training.
    offset = apply_local_head_origin_offset(rot, size, np.array([0.03,-0.35,-0.2]))
    x += offset[0]
    y += offset[1]

    return { 
        'pose' :  rot.as_quat(),
        'coord' : np.array([x, y, size]),
        'roi' : roi, 
        'image' : img,
    }


def generate_hdf5_dataset(source_file, outfilename, count=None):
    every = 1
    with ZipFile(source_file,mode='r') as zf:
        calibration_data = { 
            k:get_camera_extrinsics(zf,fn) for k,fn in find_cal_files(zf).items() }
        print ("Found calibration files: ", calibration_data.keys())
        sequence_frames = find_image_file_names(zf)
        if count or every:
            for k, v in sequence_frames.items():
                sequence_frames[k] = v[slice(0,count,every)]
        sequence_lengths = [len(v) for v in sequence_frames.values()]
        print ([(k,len(v)) for k,v in sequence_frames.items()])
        N = sum(sequence_lengths)
        with h5py.File(outfilename, mode='w') as f:
            ds_img = create_pose_dataset(f, C.image, count=N)
            ds_roi = create_pose_dataset(f, C.roi, count=N)
            ds_quats = create_pose_dataset(f, C.quat, count=N)
            ds_coords = create_pose_dataset(f, C.xys, count=N)
            f.create_dataset('sequence_starts', data = np.cumsum([0]+sequence_lengths))
            i = 0
            with tqdm.tqdm(total=N) as bar:
                for ident, frames in sequence_frames.items():
                    for fn in frames:
                        sample = read_data(zf, fn, calibration_data[ident])
                        ds_img[i] = sample['image']
                        ds_quats[i] = sample['pose']
                        ds_coords[i] = sample['coord']
                        ds_roi[i] = sample['roi']
                        i += 1
                        bar.update(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert dataset")
    parser.add_argument('source', help="source file", type=str)
    parser.add_argument('destination', help='destination file', type=str, nargs='?', default=None)
    parser.add_argument('-n', dest = 'count', type=int, default=None)
    args = parser.parse_args()
    dst = args.destination if args.destination else \
        splitext(args.source)[0]+'.h5'
    generate_hdf5_dataset(args.source, dst, args.count)