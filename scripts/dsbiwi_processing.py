import os
import sys
import numpy as np
from os.path import join, dirname, basename, splitext
import re
import math
import progressbar
import zipfile
from scipy.spatial.transform import Rotation
import scipy.io
import cv2
import utils
import h5py
import argparse
import io
from collections import defaultdict

from datasets.preprocessing import PinholeCam, imdecode, imencode

PROJ_FOV = 65.
HEAD_SIZE_MM = 100.


dt = h5py.special_dtype(vlen=np.dtype('uint8'))


def get_pose_from_mat(f):
    lines = f.readlines()
    matrix = np.array([[*map(float,row.split(' ')[:3])] for row in lines[:3]])
    P = np.array([
        [ 0, 0, 1 ],
        [ 0, 1, 0  ],
        [ 1, 0, 0  ]
    ], dtype=np.float64)
    matrix = np.dot(P, np.dot(matrix, P))
    rot = Rotation.from_matrix(matrix)
    pos = np.array([*map(float, lines[4].split(' ')[:3])])
    return rot, pos


def compute_rotation_to_vector(pos):
    # Computes a rotation which aligns 
    # the x axes with the given head position.
    x = pos / np.linalg.norm(pos)
    # Here is swap X and Z because in biwi the
    # depth direction is z, but for me it is x.
    x = x[[2,1,0]]
    z = np.cross(x, [0.,1.,0.])
    y = -np.cross(x, z)
    M = np.array([x,y,z]).T
    rot = Rotation.from_matrix(M)
    return rot


def find_image_file_names(zf):
    """
        Returns dict of filename lists. Keys are person numbers.
    """
    regex = re.compile(r'(\d\d)/frame_(\d\d\d\d\d)_rgb.png')
    samples = defaultdict(list)
    for f in zf.filelist:
        m = regex.match(f.filename)
        if m is None:
            continue
        person = int(m.group(1))
        frame = m.group(2)
        samples[person].append((frame, f.filename))
    for k, v in samples.items():
        # Sort by frame number then discard frame number
        v = sorted(v, key = lambda t: t[0])
        samples[k] = [ fn for (_,fn) in v ]
    return samples


def read_data(zf, imagefile):
    posefile = imagefile[:-len('_rgb.png')]+'_pose.txt'

    imgbuffer = zf.read(imagefile)
    img = imdecode(imgbuffer, True) # BGR
    h, w, _ = img.shape

    with io.StringIO(zf.read(posefile).decode('ascii')) as f:
        rot, pos = get_pose_from_mat(f)
    rot_correction = compute_rotation_to_vector(pos)
    rot = rot_correction.inv() * rot

    cam = PinholeCam(PROJ_FOV, w, h)
    x, y = cam.project_to_image(pos)
    # FIXME: Biwi has a different way to represent the head size than
    #        AFLW & 300W-LP. This must be accounted for!
    size = cam.project_size_to_image(pos[2], HEAD_SIZE_MM)
    roi = np.array([(x-size, y-size, x+size, y+size)])

    return { 
        'pose' :  rot.as_quat(),
        'coord' : np.array([x, y, size]),
        'roi' : roi, 
        'image' : imencode(img),
        'file' : ('biwi/'+imagefile[:-len('_rgb.png')]).encode('ascii'),
    }


def generate_hdf5_dataset(source_file, outfilename, count=None):
    # FIXME: Add a dataset id
    with zipfile.ZipFile(source_file) as zf:
        sequence_frames = find_image_file_names(zf)
        if count:
            for k, v in sequence_frames.items():
                sequence_frames[k] = v[:count]
        sequence_lengths = [len(v) for v in sequence_frames.values()]
        N = sum(sequence_lengths)
        cs = min(N, 1024)
        with h5py.File(outfilename, mode='w') as f:
            ds_img = f.create_dataset('images', (N,), chunks=(cs,), maxshape=(N,), dtype=dt)
            ds_quats = f.create_dataset('quats', (N,4), chunks=(cs,4), maxshape=(N,4), dtype='f4')
            ds_coords = f.create_dataset('coords', (N,3), chunks=(cs,3), maxshape=(N,3), dtype='f4')
            ds_file = f.create_dataset('files', (N,), chunks=(cs,), maxshape=(N,), dtype='S40')
            ds_roi = f.create_dataset('rois', (N,4), chunks=(cs,4), maxshape=(N,4), dtype='f4')
            f.create_dataset('sequence_starts', data = np.cumsum([0]+sequence_lengths))
            i = 0
            with progressbar.ProgressBar() as bar:
                for fn in bar(sum(sequence_frames.values(), [])):
                    sample = read_data(zf, fn)
                    ds_img[i] = sample['image']
                    ds_quats[i] = sample['pose']
                    ds_coords[i] = sample['coord']
                    ds_file[i] = sample['file']
                    ds_roi[i] = sample['roi']
                    i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert dataset")
    parser.add_argument('source', help="source file", type=str)
    parser.add_argument('destination', help='destination file', type=str, nargs='?', default=None)
    parser.add_argument('-n', dest = 'count', type=int, default=None)
    args = parser.parse_args()
    dst = args.destination if args.destination else \
        splitext(args.source)[0]+'.h5'
    generate_hdf5_dataset(args.source, dst, args.count)