import os
import sys
import zipfile
import pandas
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
from typing import Tuple, Sequence, Any, Optional, Dict
from numpy.typing import NDArray

from facenet_pytorch import MTCNN

from trackertraincode.datasets.preprocessing import imdecode, imencode, extract_image_roi
from trackertraincode.datasets.dshdf5pose import create_pose_dataset, FieldCategory

from dsprocess_lapa import improve_roi_with_face_detector
from filter_dataset import filter_file_by_frames

C = FieldCategory

# See "FSA-Net: Learning Fine-Grained Structure Aggregation for Head Pose Estimation from a Single Image"
# as reference for the evaluation protocol. Uses MTCNN to generate bounding boxes.
# https://github.com/shamangary/FSA-Net/blob/master/data/TYY_create_db_biwi.py
# Differences:
# * Projection using the camera matrix
# * Keeping the aspect ratio in the face crop. (The bbox is extended along the shorter side.)
# * Skip frames where MTCNN predicts a box far off the projected head center. (In FSA frames where skipped
#   based on the inter-frame movement difference.)
# * Where multiple detections: Take the one closest to the projected head center. (In FSA, the detection
#   closest to a fixed position in the image was picked, approximating the heads locations)
# * When generating the dataset with cropped images, a rotation correction is applied which is due to perspective.
#   Thereby the angle spanned between the forward direction and the head position is added to the head orientation.
#   Currently, this assumes a prescribed FOV.
# * Only small number of images is affected by failed detections. 15074 of 15678 are good.

# Update:
# This script can now load the anotations from
# https://github.com/pcr-upm/opal23_headpose/blob/main/annotations/biwi_ann.txt
# for best reproducability and fair comparison.


PROJ_FOV = 65.
HEAD_SIZE_MM = 100.
PREFIX1='faces_0/'
PREFIX2='kinect_head_pose_db/'


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


Affine = Tuple[Rotation,NDArray[Any]]


def affine_to_str(affine : Affine):
    rot, pos = affine
    return f"R={rot.as_matrix()}, T={pos}"


def assert_extrinsics_have_identity_rotation(extrinsics : Sequence[Tuple[Any,Affine]]):
    for id_, (rot,_) in extrinsics:
        assert np.allclose(rot.as_matrix(), np.eye(3),atol=0.04, rtol=0.), f"Rotation {rot.as_matrix()} of {id_} is far from identity"


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
    # Computes a rotation which aligns the z axes with the given position.
    # It's done in terms of a rotation matrix which represents the new aligned
    # coordinate frame, i.e. it's z-axis will point towards "pos". This leaves
    # a degree of rotation around the this axis. This is resolved by constraining
    # the x axis to the horizonal plane (perpendicular to the global y-axis).
    z = pos / np.linalg.norm(pos)
    x = np.cross([0.,1.,0.],z)
    x = x / np.linalg.norm(x)
    y = -np.cross(x, z)
    y = y / np.linalg.norm(y)
    M = np.array([x,y,z]).T
    rot = Rotation.from_matrix(M)
    return rot


def transform_local_to_screen_offset(rot, sz, offset):
    offset = rot.apply(offset)*sz
    # world to screen transform:
    offset = offset[:2]
    return offset


def find_image_file_names(filelist : Sequence[str]):
    """
        Returns dict of filename lists. Keys are person numbers.
    """
    regex = re.compile(PREFIX1+r'(\d\d)/frame_(\d\d\d\d\d)_rgb.png')
    samples = defaultdict(list)
    for f in filelist:
        m = regex.match(f)
        if m is None:
            continue
        person = int(m.group(1))
        frame = m.group(2)
        samples[person].append((frame, f))
    for k, v in samples.items():
        # Sort by frame number then discard frame number
        v = sorted(v, key = lambda t: t[0])
        samples[k] = [ fn for (_,fn) in v ]
    return samples


def find_cal_files(zf : ZipFile):
    regex = re.compile(PREFIX1+r'(\d\d)/rgb.cal')
    cal_files = {}
    for f in zf.filelist:
        m = regex.match(f.orig_filename)
        if m is None:
            continue
        person = int(m.group(1))
        cal_files[person] = f.orig_filename
    return cal_files


def read_data(zf, imagefile, cam_extrinsics_inv, mtcnn : MTCNN | None, box_annotation : tuple[int,int,int,int] | None):
    posefile = imagefile[:-len('_rgb.png')]+'_pose.txt'

    imgbuffer = zf.read(imagefile)
    img = imdecode(imgbuffer, True) # BGR
    h, w, _ = img.shape

    with io.StringIO(zf.read(posefile).decode('ascii')) as f:
        rot, pos = get_pose_from_mat(f)
    
    rot, pos = utils.affine3d_chain(cam_extrinsics_inv, (rot, pos))

    cam = PinholeCam(PROJ_FOV, w, h)
    x, y = cam.project_to_image(pos)
    size = cam.project_size_to_image(pos[2], HEAD_SIZE_MM)

    if box_annotation:
        roi = np.asarray(box_annotation)
    else:
        roi = np.array([x-size, y-size, x+size, y+size])

    if mtcnn is not None:
        roi, ok = improve_roi_with_face_detector(img, roi, mtcnn, iou_threshold=0.01, confidence_threshold=0.9)
        if not ok:
            print (f"WARNING: MTCNN didn't find a face that overlaps with the projected head position. Frame {imagefile}.")
    else:
        ok = True

    # Offset in local frame is given as argument.
    # It was found by eyemeasure. It could perhaps be improved by optimizing it during the training.
    # It does not affect the rotation, and thus also not the benchmarks, so I'm free to do this.
    offset = transform_local_to_screen_offset(rot, size, np.array([0.03,-0.35,-0.2]))
    x += offset[0]
    y += offset[1]

    return { 
        'pose' :  rot.as_quat(),
        'coord' : np.array([x, y, size]),
        'roi' : roi, 
        'image' : img,
    }, ok


def generate_hdf5_dataset(source_file, outfilename, opal_annotation : str | None, count : int | None =None):
    mtcnn = None
    sequence_frames = None
    box_annotations = None
    if opal_annotation:
        dataframe = pandas.read_csv(opal_annotation, header=0, sep=';')
        dataframe.columns = dataframe.columns[1:].append(pandas.Index([ 'dummy']))
        filelist : list[str] = dataframe['image'].values.tolist()
        filelist = [ f.replace(PREFIX2,PREFIX1) for f in filelist ]
        box_annotations = dataframe[list('tl_x;tl_y;br_x;br_y'.split(';'))].values.tolist()
        box_annotations = dict(zip(filelist, box_annotations))
        sequence_frames = find_image_file_names(filelist)
        assert sum(len(frames) for frames in sequence_frames.values()) == len(filelist)
    else:
        mtcnn = MTCNN(keep_all=True, device='cpu')
    every = 1
    with ZipFile(source_file,mode='r') as zf:
        calibration_data = { 
            k:get_camera_extrinsics(zf,fn) for k,fn in find_cal_files(zf).items() }
        #print ("Found calibration files: ", calibration_data.keys())
        print ("Sample camera params: ", affine_to_str(next(iter(calibration_data.values()))))
        assert_extrinsics_have_identity_rotation(calibration_data.items())
        if opal_annotation is None:
            sequence_frames = find_image_file_names([ f.orig_filename for f in zf.filelist])
        if count or every:
            for k, v in sequence_frames.items():
                sequence_frames[k] = v[slice(0,count,every)]
        max_num_frames = sum(len(v) for v in sequence_frames.values())
        print ("Found videos (id,length): ",[(k,len(v)) for k,v in sequence_frames.items()])
        with h5py.File(outfilename, mode='w') as f:
            ds_img = create_pose_dataset(f, C.image, count=max_num_frames)
            ds_roi = create_pose_dataset(f, C.roi, count=max_num_frames)
            ds_quats = create_pose_dataset(f, C.quat, count=max_num_frames)
            ds_coords = create_pose_dataset(f, C.xys, count=max_num_frames)
            i = 0
            sequence_starts = [0]
            with tqdm.tqdm(total=max_num_frames) as bar:
                for ident, frames in sequence_frames.items():
                    for fn in frames:
                        sample, ok = read_data(zf, fn, calibration_data[ident], mtcnn, box_annotations[fn] if box_annotations else None)
                        if ok:
                            ds_img[i] = sample['image']
                            ds_quats[i] = sample['pose']
                            ds_coords[i] = sample['coord']
                            ds_roi[i] = sample['roi'].tolist()
                            i += 1
                        bar.update(1)
                    assert i != sequence_starts[-1], "Every sequence should have at least one good frame!"
                    sequence_starts.append(i)
            for ds in [ds_img, ds_roi, ds_quats, ds_coords]:
                ds.resize(i, axis=0)
            f.create_dataset('sequence_starts', data = sequence_starts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert dataset")
    parser.add_argument('source', help="source file", type=str)
    parser.add_argument('destination', help='destination file', type=str, nargs='?', default=None)
    parser.add_argument('-n', dest = 'count', type=int, default=None)
    parser.add_argument('--opal-annotation', help='Use annotations from opal paper', type=str, nargs='?', default=None)
    args = parser.parse_args()
    dst = args.destination if args.destination else \
        splitext(args.source)[0]+'.h5'
    generate_hdf5_dataset(args.source, dst, args.opal_annotation, args.count)