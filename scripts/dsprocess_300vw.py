import os
import numpy as np
from os.path import join, dirname, basename, splitext
import argparse
import zipfile
import io
import h5py
import scipy.io
import itertools
import cv2
import io
import collections
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from collections import defaultdict
import re
import tqdm
import tempfile

from trackertraincode.neuralnets.affine2d import Affine2d
from trackertraincode.datatransformation.tensors.affinetrafo import transform_roi, transform_keypoints
from scripts.dsprocess_wflw import cropped
from trackertraincode.datasets.preprocessing import imencode, box_iou
from trackertraincode.datasets.dshdf5pose import create_pose_dataset, FieldCategory

C = FieldCategory

# TODO: Use my own localizer
from facenet_pytorch import MTCNN


@dataclass
class VideoInfo(object):
    annot : List[Tuple[int,str]] = field(default_factory=list) # Annotation filenames
    video : Optional[str] = None


def discover_items(zf):
    match_annotation = re.compile(r'.*(\d\d\d)/annot/(\d\d\d\d\d\d)\.pts')
    match_video = re.compile(r'.*(\d\d\d)/(.+)\.avi')
    annotations = defaultdict(VideoInfo)
    for f in zf.filelist:
        m = match_annotation.match(f.filename)
        if m is not None:
            vid_num, frame_num = m.group(1), m.group(2)
            annotations[vid_num].annot.append((int(frame_num),f.filename))
        elif (m := match_video.match(f.filename)) is not None:
            vid_num = m.group(1)
            annotations[vid_num].video = f.filename
    return annotations


def read_annotation(f : io.StringIO):
    lines = f.readlines()
    lines = lines[3:-1]
    assert len(lines) == 68, "Expected one landmark per line for a total of 68"
    def _cvt_pt(line):
        a, b = (float(s.strip()) for s in line.split())
        return a, b
    return np.asarray([ _cvt_pt(l) for l in lines ])


def iter_annotation_files(zf : zipfile.ZipFile, vi : VideoInfo):
    '''
    Points only
    '''
    for i,fn in sorted(vi.annot, key=lambda x: x[0]):
        f = io.StringIO(zf.read(fn).decode('ascii'))
        yield read_annotation(f)


def iter_frames(zf : zipfile.ZipFile, vi : VideoInfo):
    with tempfile.TemporaryDirectory() as tmp:
        tmpfilename = join(tmp,'video_oh_why_1000.avi')
        with open(tmpfilename,'wb') as f:
            # Cannot use `zf.extract` because it will create directories
            f.write(zf.read(vi.video))
        vidcap = cv2.VideoCapture(tmpfilename)
        while True:
            success,image = vidcap.read()
            if not success:
                break
            yield image


def roi_from_points(points: np.ndarray):
    tl = np.amin(points, axis=-2)
    br = np.amax(points, axis=-2)
    return np.concatenate([tl, br], axis=-1)


def compute_padding_from_rois(rois : np.ndarray):
    tl = rois[...,:2]
    br = rois[...,2:]
    diag = np.linalg.norm(br - tl, axis=-1)
    return max(10, np.amax(diag)*0.5)


def compute_scaling_from_rois(rois : np.ndarray, desired_roi_size : int):
    tl = rois[...,:2]
    br = rois[...,2:]
    maxlen = np.amax(br - tl)
    tolerance_factor = 1.5
    return min(1, desired_roi_size*tolerance_factor / maxlen)


def apply_face_detector(img, roi, mtcnn : MTCNN):
    '''
    The original roi might be returned if the overlap with the detection is low, indicating maybe a bad detection.
    '''
    new_roi, _ = mtcnn.detect(img)
    if new_roi is not None:
        iou = box_iou(roi, new_roi)
        i = np.argmax(iou)
        new_roi = new_roi[i]
        iou = iou[i]
        if iou > 0.25:
            return new_roi, True
    return roi, False


def improve_roi_with_face_detector(img, roi, mtcnn : MTCNN):
    '''
    Applies face detector on cropped image. Then roi is back-transformed to original image.
    '''
    temp_img, trafo = cropped(img, roi, desired_roi_size=224)
    temp_roi = transform_roi(trafo, roi)
    temp_roi, roi_ok = apply_face_detector(temp_img, temp_roi, mtcnn)
    if roi_ok:
        # Use the corrected roi instead of the original one
        roi = transform_roi(trafo.inv(), temp_roi)
    return roi, roi_ok


def process_video(zf : zipfile.ZipFile, vi : VideoInfo, mtcnn : MTCNN):
    landmarks = np.asarray(list(iter_annotation_files(zf, vi)), dtype='f4')
    
    rois = roi_from_points(landmarks)

    roi_across_frames = roi_from_points(landmarks.reshape(-1,2))

    scaling = compute_scaling_from_rois(rois, desired_roi_size=129)
    abs_padding = scaling*compute_padding_from_rois(rois)

    for roi, landmark, img in zip(rois, landmarks, iter_frames(zf, vi)):
        large_enough_to_disable_downscaling = 1<<16
        
        h, w = img.shape[:2]
        myscale = int(w * scaling)/w
        new_w = int(w * myscale)
        new_h = int(h * myscale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        img, trafo = cropped(img, myscale*roi_across_frames, desired_roi_size=large_enough_to_disable_downscaling, padding_factor=0, abs_padding=abs_padding)
        
        landmark = transform_keypoints(trafo, myscale*landmark)
        roi = transform_roi(trafo, myscale*roi)

        roi, roi_ok = improve_roi_with_face_detector(img, roi, mtcnn)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        yield img, landmark, roi, roi_ok


def do_conversion(zf : zipfile.ZipFile, videoinfos : List[VideoInfo], f : h5py.File, max_count = None):
    mtcnn = MTCNN(keep_all=True, device='cpu')
    if max_count is not None:
        videoinfos = videoinfos[:max_count]
    sequence_starts = np.cumsum([0]+[len(vi.annot) for vi in videoinfos])
    
    N = sequence_starts[-1]
    
    ds_img = create_pose_dataset(f, C.image, count=N)
    f.create_dataset('sequence_starts', data = sequence_starts)

    pt2ds_68 = []
    rois = []

    i = 0
    with tqdm.tqdm(total=N) as bar:
        for vi in videoinfos:
            for frame, points, roi, roi_ok in process_video(zf, vi, mtcnn):
                if not roi_ok:
                    print (f"face detection failure frame {i}, original {vi.video}")
                pt2ds_68.append(points)
                rois.append(roi)
                ds_img[i] = imencode(frame, quality=95)
                i += 1
                bar.update(1)

    create_pose_dataset(f, C.points, name='pt2d_68', data = pt2ds_68, dtype=np.float16)
    create_pose_dataset(f, C.roi, data = rois, dtype=np.float16)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert dataset")
    parser.add_argument('source', help="source file", type=str)
    parser.add_argument('destination', help="destination file", type=str)
    parser.add_argument('-n', dest = 'count', type=int, default=None)
    args = parser.parse_args()

    with zipfile.ZipFile(args.source) as zf:
        with h5py.File(args.destination,'w') as f:
            directories = discover_items(zf)
            do_conversion(zf, list(directories.values()), f, max_count=args.count)