import os
import numpy as np
from os.path import join, dirname, basename, splitext
import cv2
import h5py
import argparse
import itertools
import zipfile
import tqdm

from trackertraincode.datasets.dshdf5pose import create_pose_dataset, FieldCategory

C = FieldCategory


def iterfiles(zf : zipfile.ZipFile):
    contents = frozenset(zf.namelist())
    for i in itertools.count():
        img = f'{i:06d}.png'
        if img not in contents:
            break
        lmk = f'{i:06d}_ldmks.txt'
        assert lmk in contents
        yield img, lmk

def convert(zf : zipfile.ZipFile, filename):
    with zf.open(filename, 'r') as f:
        lines = f.readlines()
    cvtline = lambda line: tuple(float(u.strip()) for u in line.split())
    lmks = np.asarray(list(map(cvtline, lines)))
    assert lmks.shape == (70,2), f"Bad shape {lmks.shape}"
    return lmks

def generate_roi(landmarks):
    min_ = np.amin(landmarks[...,:2], axis=-2)
    max_ = np.amax(landmarks[...,:2], axis=-2)
    roi = np.concatenate([min_, max_], axis=-1).astype(np.float32)
    return roi

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert dataset")
    parser.add_argument('source', help="source file", type=str)
    parser.add_argument('destination', help="Destination file", type=str)
    parser.add_argument('-n', dest = 'count', type=int, default=None)
    args = parser.parse_args()

    with zipfile.ZipFile(args.source, 'r') as zf:
        # Test
        convert(zf, '000000_ldmks.txt')

        files = list(iterfiles(zf))

        if args.count:
            files = files[:args.count]

        relative_image_paths = [ a for a,b in files ]

        data = np.asarray([ convert(zf,b) for a,b in files ])
        roi = generate_roi(data)

        data = data[:,:68,:]
        data = np.concatenate([data, np.zeros((data.shape[0],68,1))], axis=-1)

        assert roi.shape[0] == data.shape[0] and roi.shape[1]==4 and data.shape[1:]==(68,3)

        with h5py.File(args.destination, 'w') as f:
            create_pose_dataset(f, C.points, 'pt3d_68', data = data)
            create_pose_dataset(f, C.roi, data = roi)
            ds_img = create_pose_dataset(f, C.image, count = len(relative_image_paths), lossy=False)
            for i, name in tqdm.tqdm(enumerate(relative_image_paths)):
                jpgbuffer = zf.read(name)
                ds_img[i] = np.frombuffer(jpgbuffer, dtype='uint8')