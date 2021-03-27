import os
import sys
import numpy as np
import glob
import random
import zipfile
from copy import copy
from os.path import join
from collections import namedtuple
from typing import Union
import h5py
import argparse
import cv2
import progressbar
import itertools

import utils
from datasets.preprocessing import imdecode, imencode

Annotation = namedtuple('Annotation', 'filename boxes')

class WiderFace(object):
    def __init__(self, root_dir, validation):
        self.root_dir = root_dir
        self.validation = validation
        self.subset = 'wider_face_val_bbx_gt.txt' if validation else 'wider_face_train_bbx_gt.txt'
        self.annotation_file = join(self.root_dir,'wider_face_split.zip')
        self.trainimage_file = join(self.root_dir,'WIDER_val.zip' if validation else 'WIDER_train.zip')
        self.trainimage_zip = zipfile.ZipFile(self.trainimage_file)
        self.annotations = self._read_annotation()
        
    def _read_annotation(self):
        zf = self.trainimage_zip
        imagenames = frozenset([
            f.filename for f in zf.filelist if f.external_attr==0x20
        ])
        with zipfile.ZipFile(self.annotation_file) as zf:
            annolines = zf.read('wider_face_split/'+self.subset).decode('ascii').splitlines()
        annos = []
        it = iter(enumerate(annolines))
        while True:
            try:
                lineno, fn = next(it)
            except StopIteration:
                break
            fn = 'WIDER_'+('val' if self.validation else 'train')+'/images/' + fn
            if not fn in imagenames:
                #print (f"Warning line {lineno}: {fn} is not in the file list")
                continue
            a = Annotation(fn, [])
            lineno, numboxes = next(it)
            numboxes = int(numboxes)
            for _, (lineno,boxline) in zip(range(numboxes), it):
                x1, y1, w, h = boxline.split()[:4]
                x0, y0, w, h = map(int, [x1, y1, w, h])
                if w==0 or h==0:
                    #print(f"Warning line {lineno}: {fn} has zero size box")
                    continue
                x1, y1 = x0+w, y0+h
                a.boxes.append((x0,y0,x1,y1))
            annos.append(a)
        return annos
    
    def image(self, a : Union[Annotation,int]):
        if isinstance(a, int):
            a = self.annotations[a]
        file = self.trainimage_zip.read(a.filename)
        img = imdecode(file, 'rgb')
        return img
    
    def close(self):
        if self.trainimage_zip is not None:
            self.trainimage_zip.close()
            self.trainimage_zip = None
        
    def __enter__(self):
        return self
    
    def __exit__(self, *args, **kwargs):
        self.close()


def compute_max_crop_size(boxwidth, imgwidth, size_fraction):   
    size = boxwidth / size_fraction
    size = min(imgwidth, size)
    return size


def face_crop(imgshape, box, target_aspect, target_face_size_frac, rnd):
    x0, y0, x1, y1 = box
    h, w, _ = imgshape
    max_crop_w = compute_max_crop_size(x1-x0, w, target_face_size_frac)
    max_crop_h = max_crop_w / target_aspect
    if max_crop_h > h:
        max_crop_w *= h/max_crop_h
        max_crop_h = h
    #print (max_crop_w, max_crop_h)
    xmax = x0 - max(0, x0 + max_crop_w - w)
    xmin = x1 - max_crop_w - min(0, x1 - max_crop_w)
    ymax = y0 - max(0, y0 + max_crop_h - h)
    ymin = y1 - max_crop_h - min(0, y1 - max_crop_h)
    #print (xmin, xmax, ymin, ymax)
    rx, ry = rnd.uniform(0., 1., size=2)
    xc = xmin + rx * (xmax - xmin)
    yc = ymin + ry * (ymax - ymin)
    return (xc,yc,xc+max_crop_w, yc+max_crop_h), (xmin, ymin, xmax, ymax, max_crop_w, max_crop_h)


def no_face_crop(imgshape, box, aspect, rnd):
    h, w, _ = imgshape
    x0, y0, x1, y1 = box
    if x0 < w-x1:
        # Take from right of box
        u0 = x1
        u1 = w
    else:
        u0 = 0
        u1 = x0
    dv = (u1-u0)/aspect
    if dv > h:
        du = h * aspect
        u0 = u0 + rnd.randint(0, max(0,u1-u0-du)+1)
        u1 = u0 + du
        dv = h
    r = rnd.randint(0,h-dv+1)
    v0 = r
    v1 = r + dv
    return (u0, v0, u1, v1)


class SingleWiderFaces(object):
    def __init__(self, root, validation):
        self.rnd = np.random.RandomState(seed=123)
        self.validation = validation
        self.root = root
        with WiderFace(root, validation) as wf:
            self.singleface_annos = [a for a in wf.annotations if len(a.boxes)==1]

    def __len__(self):
        return len(self.singleface_annos)*2

    def _cropimg(self, img, cropbox, box):
        h, w, _ = img.shape
        x0, y0, x1, y1 = map(int, cropbox)
        x1 = min(w, x1)
        x0 = max(0, x0)
        y1 = min(h, y1)
        y0 = max(0, y0)
        img = img[y0:y1,x0:x1,...]
        u0, v0, u1, v1 = box
        u0 -= x0
        u1 -= x0
        v0 -= y0
        v1 -= y0
        return (img, (u0,v0,u1,v1))
    
    def _maybe_scale(self, img, box):
        h, w, _ = img.shape
        if h > 640 or w > 640:
            f = 640. / max(h,w)
            w = round(w*f)
            h = round(h*f)
            img = cv2.resize(img, (w, h), interpolation = cv2.INTER_AREA)
            x0, y0, x1, y1 = box
            x0 *= f
            y0 *= f
            x1 *= f
            y1 *= f
            box = (x0, y0, x1, y1)
        return img, box

    def _make_sample(self, img, cropbox, box, hasface):
        img, box = self._cropimg(img, cropbox, box)
        img, box = self._maybe_scale(img, box)
        return {
            'image' : img,
            'roi' : box if hasface else (0,0,0,0),
            'hasface' : hasface
        }

    def __iter__(self):
        with WiderFace(self.root, self.validation) as wf:
            for a in self.singleface_annos:
                box = a.boxes[0]
                img = wf.image(a)
                h, w, _ = img.shape
                size_frac = self.rnd.uniform(0.1, 0.33)
                (fx0, fy0, fx1, fy1), _ = face_crop(img.shape, box, 4./3., size_frac, self.rnd)
                (ex0, ey0, ex1, ey1) = no_face_crop(img.shape, box, 4./3., self.rnd)
                yield self._make_sample(img, (fx0, fy0, fx1, fy1), box, True)
                yield self._make_sample(img, (ex0, ey0, ex1, ey1), box, False)


def generate_hdf5_dataset(source_dir, outfilename, count=None):
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    wfval = SingleWiderFaces(source_dir, validation=True)
    wftrain = SingleWiderFaces(source_dir, validation=False)
    N = len(wftrain) + len(wfval)
    if count is not None:
        N = min(count, N)
    with h5py.File(outfilename, 'w') as f:
        cs = min(N, 1024)
        ds_img = f.create_dataset('images', (N,), chunks=(cs,), maxshape=(N,), dtype=dt)
        ds_roi = f.create_dataset('rois', (N,4), chunks=(cs,4), maxshape=(N,4), dtype='f4')
        ds_hasface = f.create_dataset('hasface', (N,), chunks=(cs,), maxshape=(N,), dtype='?')
        indices = np.random.permutation(N)
        with progressbar.ProgressBar(max_value=N) as bar:
            for i, sample in bar(zip(indices,itertools.chain(wftrain, wfval))):
                ds_img[i] = imencode(cv2.cvtColor(sample['image'], cv2.COLOR_RGB2BGR))
                ds_roi[i] = sample['roi']
                ds_hasface[i] = sample['hasface']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert dataset")
    parser.add_argument('source', help="source file", type=str)
    parser.add_argument('destination', help='destination file', type=str, nargs='?', default=None)
    parser.add_argument('-n', dest = 'count', type=int, default=None)
    args = parser.parse_args()
    dst = args.destination if args.destination else \
        args.source+'.h5'
    generate_hdf5_dataset(args.source, dst, args.count)