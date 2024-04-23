'''
Conversion for the LaPa dataset.

Beware, images intersect with 300W-LP and Megaface.
'''
import os
import numpy as np
from os.path import join, dirname, basename, splitext
import argparse
import zipfile
import io
import h5py
import cv2
import io
from typing import List, Optional, Tuple, Dict, NamedTuple
import re
import tqdm
import torch
from pathlib import Path
from scipy.interpolate import interp1d
from trackertraincode.datasets.preprocessing import box_iou, imdecode
from trackertraincode.datasets.preprocessing import imread, rgb2gray, extend_rect, imencode, imrescale, imshape, ImageFormat
from trackertraincode.neuralnets.affine2d import Affine2d
from trackertraincode.datatransformation.affinetrafo import transform_roi, transform_points
from trackertraincode import vis
from trackertraincode.datasets.dshdf5pose import create_pose_dataset, FieldCategory
from dsprocess_wflw import cropped


C = FieldCategory

# TODO: Use my own localizer
from facenet_pytorch import MTCNN


class DatasetInfo(NamedTuple):
    imagedir : Path
    lmkdir : Path
    itemnames : List[str]


def discover_items(source_dir):
    root = Path(source_dir)/'train'/'images'
    items = [ p.relative_to(root).stem for p in Path.glob(root,'*.jpg') ]
    return DatasetInfo(
        root,
        Path(source_dir)/'train'/'landmarks',
        items)


def filter_megaface(info : DatasetInfo):
    # Megaface consists of the files without alphabetical prefix. There's numbers only in the filename. So ...
    regex = re.compile(r"^(\d|\_)+$")
    megaface = [ x for x in info.itemnames if regex.match(x) ]
    return info._replace(itemnames = megaface)


def read_annotation(f):
    lines = f.readlines()
    assert lines[0].strip() == '106'
    lines = lines[1:]
    assert len(lines) == 106, "Expected one landmark per line for a total of 106"
    def _cvt_pt(line):
        a, b = (float(s.strip()) for s in line.split())
        return a, b
    return np.asarray([ _cvt_pt(l) for l in lines ]).astype(np.float32)


def poor_mans_roi(points: np.ndarray):
    x0, y0 = np.amin(points, axis=0)
    x1, y1 = np.amax(points, axis=0)
    return np.array([x0, y0, x1, y1])


def improve_roi_with_face_detector(img, roi, mtcnn : MTCNN):
    new_roi, _ = mtcnn.detect(img)
    if new_roi is not None:
        iou = box_iou(roi, new_roi)
        i = np.argmax(iou)
        new_roi = new_roi[i]
        iou = iou[i]
        if iou > 0.25:
            return new_roi, True
    return roi, False


def maybe_downscale(img, roi, landmarks):
    rw = roi[2]-roi[0]
    rh = roi[3]-roi[1]
    h, w = imshape(img)
    desired_roi_size = 224
    if rw > desired_roi_size and rh > desired_roi_size:
        scale = desired_roi_size/min(rh, rw)
        img = imrescale(img, scale)
        scale = imshape(img)[1] / w
        landmarks = scale*landmarks
        roi = scale*roi
        return (img, roi, landmarks)
    return None


def read_data(info : DatasetInfo, itemname, mtcnn : MTCNN):
    with open(info.imagedir / (itemname+'.jpg'), 'rb') as f:
        rawjpg = f.read()
        img = imdecode(rawjpg, 'rgb')
    with open(info.lmkdir / (itemname+'.txt'), 'r') as f:
        landmarks = read_annotation(f)
    roi = poor_mans_roi(landmarks)
    roi, _ = improve_roi_with_face_detector(img, roi, mtcnn)
    if (scaled_stuff := maybe_downscale(img, roi, landmarks)) is not None:
        img, roi, landmarks = scaled_stuff
        rawjpg = imencode(img)
    return rawjpg, roi, landmarks


def cvt_landmarks_68pt(lmk, improved_chin = False):
    lmk = lmk.swapaxes(-1,-2)
    assert lmk.shape == (2,106)
    if not improved_chin:
        chin = lmk[...,:33:2]
    else:
        # The endpoints start too far above the eyes. It's better to throw them away.
        # But then the number of points doesn't match for furhter throwing away every
        # second point, so we have to interpolate.
        chin = lmk[...,:33]
        xs = np.linspace(0.,32.,33)
        chin = interp1d(xs,chin,kind='quadratic',axis=-1,fill_value='extrapolate')(np.linspace(1.5,32.-1.5,17))
    assert chin.shape == (2,17)
    brows_pairs_left = [
        (34, 41), (35, 40), (36,39), (37,38) ]
    brows_pairs_right = [
        (42, 50), (43, 49), (44,48), (45,47) ]
    def avg(*pairs):
        a, b = zip(*pairs)
        return np.average([lmk[...,a], lmk[...,b]], axis=0)
    def rng(start, end=None):
        if end is None:
            end = start+1
        return lmk[...,start:end]
    lmk68 = np.concatenate([ 
        chin,
        rng(33), # Brow
        avg(*brows_pairs_left),
        avg(*brows_pairs_right),
        rng(46),  # Brow
        rng(51,55), # Nose
        rng(57), avg((58,59)), rng(60), avg((61,62)), rng(63),
        rng(66), # Eye Left
        avg((67, 68), (68,69)), 
        rng(70),
        avg((71,72),(72,73)),
        rng(75), # Right Eye
        avg((76,77), (77,78)),
        rng(79),
        avg((80,81), (81,82)), 
        rng(84,104) # Mouth
    ], axis=-1)
    lmk68 = lmk68.swapaxes(-1,-2)
    assert lmk68.shape[-2:] == (68,2), f"Bad shape {lmk68.shape}"
    return lmk68


def do_conversion(source_dir : str, f : h5py.File, mtcnn : MTCNN, max_count : int, only_megaface : bool):
    dsinfo = discover_items(source_dir)


    if only_megaface:
        dsinfo = filter_megaface(dsinfo)

    if max_count is not None:
        dsinfo = dsinfo._replace(itemnames = dsinfo.itemnames[:max_count])
        
    N = len(dsinfo.itemnames)
    ds_img = create_pose_dataset(f, C.image, count=N)

    pt2ds_68 = []
    rois = []
    trafos = []

    for i,name in enumerate(tqdm.tqdm(dsinfo.itemnames)):
        rawjpg, roi, points = read_data(dsinfo, name, mtcnn)
        points = cvt_landmarks_68pt(points, improved_chin=True)

        img = imdecode(rawjpg, cv2.IMREAD_COLOR)
        img, trafo = cropped(img, roi, desired_roi_size=224, padding_factor=0.5, abs_padding=10)
        
        pt2ds_68.append(points)
        rois.append(roi)
        trafos.append(trafo.tensor())
        
        ds_img[i] = imencode(img, ImageFormat.JPG, quality=95)
        #ds_img[i] = np.frombuffer(rawjpg, dtype='B')

    trafos = Affine2d(torch.stack(trafos))
    pt2ds_68 = np.asarray(pt2ds_68).astype(np.float32)
    rois = np.asarray(rois).astype(np.float32)
    pt2ds_68 = transform_points(trafos, torch.from_numpy(pt2ds_68)).numpy()
    rois = transform_roi(trafos, torch.from_numpy(rois)).numpy()

    create_pose_dataset(f, C.points, 'pt2d_68', data = pt2ds_68, dtype='f2')
    create_pose_dataset(f, C.roi, data = rois, dtype='f2')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert dataset")
    parser.add_argument('source', help="source dir", type=str)
    parser.add_argument('destination', help="destination file", type=str)
    parser.add_argument('--only-megaface', help="Only generate dataset from the megaface part of the original", default=False, action='store_true')
    parser.add_argument('-n', dest = 'count', type=int, default=None)
    args = parser.parse_args()

    mtcnn = MTCNN(keep_all=True, device='cpu')
    with h5py.File(args.destination,'w') as f:
        do_conversion(args.source, f, mtcnn, args.count, args.only_megaface)