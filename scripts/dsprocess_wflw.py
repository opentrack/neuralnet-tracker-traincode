"""
WFLW is from "Look at Boundary: A Boundary-Aware Face Alignment Algorithm"

Looks like these images weren't used before.
"""

import os
import numpy as np
from os.path import join, dirname, basename, splitext
import cv2
import h5py
import argparse
import tqdm
import itertools
import torch
from typing import Tuple, Union
import torch
from PIL import Image

from trackertraincode.datasets.dshdf5pose import Hdf5PoseDataset
from trackertraincode.datasets.preprocessing import extend_rect, imencode, imrescale, imshape
from trackertraincode.neuralnets.affine2d import Affine2d
from trackertraincode.datatransformation.tensors.affinetrafo import transform_roi, transform_points
from trackertraincode.datasets.dshdf5pose import create_pose_dataset, FieldCategory

C = FieldCategory


def cvt_landmarks_68pt(lmk):
    assert lmk.shape[-2:] == (2, 98)
    chin = lmk[..., :33:2]
    assert chin.shape[-2:] == (2, 17)
    brows_pairs_left = [(34, 41), (35, 40), (36, 39), (37, 38)]
    brows_pairs_right = [(42, 50), (43, 49), (44, 48), (45, 47)]

    def avg(*pairs):
        a, b = zip(*pairs)
        return np.average([lmk[..., a], lmk[..., b]], axis=0)

    def rng(start, end=None):
        if end is None:
            end = start + 1
        return lmk[..., start:end]

    lmk68 = np.concatenate(
        [
            chin,
            rng(33),  # Brow
            avg(*brows_pairs_left),
            avg(*brows_pairs_right),
            rng(46),  # Brow
            rng(51, 60),  # Nose
            rng(60),  # Eye
            avg((61, 62), (62, 63)),
            rng(64),
            avg((66, 65), (67, 66)),
            rng(68),  # Right Eye
            avg((69, 70), (70, 71)),
            rng(72),
            avg((74, 73), (75, 74)),
            rng(76, 96),  # Mouth
        ],
        axis=-1,
    )
    lmk68 = lmk68.swapaxes(-1, -2)
    assert lmk68.shape[-2:] == (68, 2), f"Bad shape {lmk68.shape}"
    return lmk68


def convert(f):
    def cvtline(line):
        vals = [s.strip() for s in line.split(" ")]
        landmarks = vals[: 98 * 2]
        (
            x_min_rect,
            y_min_rect,
            x_max_rect,
            y_max_rect,
            pose,
            expression,
            illumination,
            makeup,
            occlusion,
            blur,
            image_name,
        ) = vals[98 * 2 :]
        # Note: landmarks come as floating point numbers
        landmarks = np.array(list(map(float, landmarks)))
        landmarks = np.stack([landmarks[::2], landmarks[1::2]], axis=-1).T
        assert landmarks.shape == (2, 98)
        roi = np.array(list(map(float, [x_min_rect, y_min_rect, x_max_rect, y_max_rect])))
        image_name = join("WFLW_images", image_name)
        return image_name, landmarks, roi

    paths, landmarks, rois = map(np.asarray, zip(*[cvtline(l) for l in f.readlines()]))
    landmarks = cvt_landmarks_68pt(landmarks)
    landmarks = landmarks.astype(np.float32)
    rois = rois.astype(np.float32)
    return paths, landmarks, rois


def cropped(
    img: Union[np.ndarray, Image.Image],
    roi: np.ndarray,
    desired_roi_size=129,
    padding_factor=0.5,
    abs_padding=10,
) -> Tuple[np.ndarray, Affine2d]:
    tr = Affine2d.trs(torch.zeros((2,)))

    rw = roi[2] - roi[0]
    rh = roi[3] - roi[1]
    h, w = imshape(img)

    # Only ever perform downscaling to save storage
    # No point in upscaling. That can be done as part of augmentation.
    alpha = 1.5  # How much bigger the ROI must be than the target roi size in order for rescaling to happen
    beta = 1.0  # And how much larger it is actually made??
    if rw > alpha * desired_roi_size and rh > alpha * desired_roi_size:
        scale = beta * desired_roi_size / min(rh, rw)
        img = imrescale(img, scale)
        scale = imshape(img)[1] / w
        h, w = imshape(img)
        tr = Affine2d.trs(scales=torch.tensor(scale))
        roi = scale * roi

    cropbox = extend_rect(roi, padding_factor, abs_padding)
    cropbox[0] = max(cropbox[0], 0)
    cropbox[1] = max(cropbox[1], 0)
    cropbox[2] = min(cropbox[2], w)
    cropbox[3] = min(cropbox[3], h)

    x0, y0, x1, y1 = cropbox.astype(int)

    img = np.asarray(img)

    img = np.ascontiguousarray(img[y0:y1, x0:x1, ...])

    tr = Affine2d.trs(torch.tensor([-x0, -y0])) @ tr

    # print (f"Store image size: {img.shape}")
    return img, tr


def mask_for_good_boxes(boxes, min_width):
    return (boxes[:, 2] - boxes[:, 0]) >= min_width


def generate_hdf5_dataset(sourcedir, outdir, count=None, min_box_width=129):
    assert os.path.exists(
        join(
            sourcedir,
            "WFLW_annotations",
            "list_98pt_rect_attr_train_test",
            "list_98pt_rect_attr_train.txt",
        )
    )
    assert os.path.exists(
        join(
            sourcedir,
            "WFLW_annotations",
            "list_98pt_rect_attr_train_test",
            "list_98pt_rect_attr_test.txt",
        )
    )
    assert os.path.exists(
        join(sourcedir, "WFLW_images", "0--Parade", "0_Parade_marchingband_1_116.jpg")
    )

    for split in ["test", "train"]:
        annotationfile = join(
            sourcedir,
            "WFLW_annotations",
            "list_98pt_rect_attr_train_test",
            f"list_98pt_rect_attr_{split}.txt",
        )

        with open(annotationfile, "r", encoding="utf-8") as f:
            paths, landmarks, rois = convert(f)

        if count is not None:
            paths = paths[:count]
            landmarks = landmarks[:count]
            rois = rois[:count]

        good_samples_mask = mask_for_good_boxes(rois, min_box_width)
        if not np.all(good_samples_mask):
            paths = paths[good_samples_mask]
            landmarks = landmarks[good_samples_mask]
            rois = rois[good_samples_mask]

        N = len(paths)

        with h5py.File(join(outdir, f"wflw_{split}.h5"), "w") as f:
            ds_img = create_pose_dataset(f, C.image, count=N)

            trafos = []

            for i, path, roi in tqdm.tqdm(zip(itertools.count(), paths, rois), total=N):
                imgs = Image.open(join(sourcedir, path))
                imgs, trafo = cropped(
                    imgs, roi, desired_roi_size=224, padding_factor=0.5, abs_padding=10
                )
                ds_img[i] = imgs
                trafos.append(trafo.tensor())

            trafos = Affine2d(torch.stack(trafos))

            landmarks = transform_points(trafos, torch.from_numpy(landmarks)).numpy()
            rois = transform_roi(trafos, torch.from_numpy(rois)).numpy()

            create_pose_dataset(f, C.points, name="pt2d_68", dtype=np.float16, data=landmarks)
            create_pose_dataset(f, C.roi, dtype=np.float16, data=rois)

        # Test it
        ds = Hdf5PoseDataset(join(outdir, f"wflw_{split}.h5"))
        assert next(iter(ds)) is not None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset")
    parser.add_argument("source", help="source directory", type=str)
    parser.add_argument(
        "destination", help="destination directory", type=str, nargs="?", default=None
    )
    parser.add_argument("-n", dest="count", type=int, default=None)
    args = parser.parse_args()

    generate_hdf5_dataset(args.source, args.destination, args.count)
