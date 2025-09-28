import sys
import os
from os.path import join, splitext, sep, dirname
from collections import defaultdict
import argparse
import sys
import numpy as np
from numpy.typing import NDArray
import cv2

cv2.setNumThreads(0)
import os
import itertools
import functools
import h5py
from scipy.spatial.transform import Rotation
import zipfile
import io
import scipy.io
import tqdm
import csv
import re
import torch
from typing import Callable, Tuple, Any, List, NamedTuple, Optional
from PIL import Image, ImageDraw, ImageOps
from matplotlib import pyplot
from pathlib import Path

from trackertraincode.datasets.dshdf5pose import create_pose_dataset, FieldCategory
from trackertraincode.datatransformation.tensors.affinetrafo import (
    transform_roi,
    transform_keypoints,
)
from dsprocess_wflw import cropped

C = FieldCategory

from facenet_pytorch import MTCNN


def detect_one(mtcnn: MTCNN, image: Image.Image) -> Tuple[NDArray[Any], str]:
    myboxes, probs = mtcnn.detect(image)
    if myboxes is None or len(myboxes) == 0:
        return None, "No faces detected"
    if len(myboxes) > 1:
        return myboxes[np.argmax(probs)], f"{len(myboxes)} faces detected"
    return myboxes[0], ""


def convert_unlabeled_sequences(directory: Path, outputfile, max_sample_count):
    """Directory must contain single frames named by the scheme: <prefix><frame>.<ext>"""

    class SampleFile(NamedTuple):
        filename: str
        ident: Optional[str]
        number: Optional[int]

    regex = re.compile(f"(.+?)(\d+).(jpg|png|jpeg|bmp)")

    def make_sample(filename):
        m = regex.match(filename)
        return SampleFile(directory / filename, m.group(1), int(m.group(2)))

    mtcnn = MTCNN(keep_all=True, device="cuda", min_face_size=16)

    sample_files = [make_sample(fn) for fn in os.listdir(directory)]
    sample_files = [x for x in sample_files if x.number is not None and x.ident is not None]
    # Python sorts tuples lexicographically, so here the identity first, and then the frame number
    sample_files = sorted(sample_files, key=lambda x: (x.ident, x.number))

    if max_sample_count is not None:
        sample_files = sample_files[:max_sample_count]

    indexed_by_ident = defaultdict(list)
    for sf in sample_files:
        indexed_by_ident[sf.ident].append(sf)

    del sample_files

    sequence_starts = np.cumsum([0] + [len(v) for v in indexed_by_ident.values()])
    N = sequence_starts[-1]

    print(f"Found {len(sequence_starts)-1} sequences.")

    boxes = []

    with h5py.File(outputfile, "w") as f:
        f.create_dataset("sequence_starts", data=sequence_starts)
        ds_roi = create_pose_dataset(f, C.roi, count=N, dtype=np.float16)
        ds_img = create_pose_dataset(f, C.image, count=N)

        i = 0
        for ident, sample_files in tqdm.tqdm(indexed_by_ident.items(), postfix="Sequence"):
            boxes = []
            images = []

            for sf in tqdm.tqdm(sample_files, "Loading ..."):
                image = Image.open(sf.filename)

                # Given 3 channel images, we can store 1000 of these in ca 1.5 GB of memory.
                # My image sequences are shorter than that, so no problem.
                if image.width > 720 and image.height > 720:
                    image.thumbnail((640, 640), Image.Resampling.HAMMING)

                box, error = detect_one(mtcnn, image)
                if box is None:
                    # Failsafe but don't use this dataset now ...
                    box = (0, 0, image.width, image.height)
                if error:
                    print(f"Det issue frame {sf.filename}: {error}")

                images.append(image)
                boxes.append(box)

            boxes = np.asarray(boxes)

            combined_box = np.concatenate(
                [np.amin(boxes[:, :2], axis=0), np.amax(boxes[:, 2:], axis=0)]
            )

            for img, box in tqdm.tqdm(zip(images, boxes), "Writing ..."):
                img, trafo = cropped(
                    np.asarray(img),
                    combined_box,
                    desired_roi_size=224,
                    padding_factor=0.25,
                    abs_padding=10,
                )
                box = transform_roi(trafo, torch.from_numpy(box)).numpy()
                ds_img[i] = img
                ds_roi[i] = box
                i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset")
    parser.add_argument("source", help="source file", type=str)
    parser.add_argument("destination", help="destination file", type=str)
    parser.add_argument("-n", dest="count", type=int, default=None)
    args = parser.parse_args()

    convert_unlabeled_sequences(Path(args.source), args.destination, args.count)
