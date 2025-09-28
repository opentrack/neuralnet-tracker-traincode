import os
import numpy as np
from os.path import join, dirname, basename, splitext
import cv2
import h5py
import argparse
import itertools
import zipfile
import tqdm
from matplotlib import pyplot

from trackertraincode.datasets.dshdf5pose import create_pose_dataset, FieldCategory
from trackertraincode.datasets.preprocessing import imdecode, imencode

C = FieldCategory


class ClassIds:
    BACKGROUND = 0
    SKIN = 1
    NOSE = 2
    RIGHT_EYE = 3
    LEFT_EYE = 4
    RIGHT_BROW = 5
    LEFT_BROW = 6
    RIGHT_EAR = 7
    LEFT_EAR = 8
    MOUTH_INTERIOR = 9
    TOP_LIP = 10
    BOTTOM_LIP = 11
    NECK = 12
    HAIR = 13
    BEARD = 14
    CLOTHING = 15
    GLASSES = 16
    HEADWEAR = 17
    FACEWEAR = 18
    IGNORE = 255


def iterfiles(zf: zipfile.ZipFile):
    contents = frozenset(zf.namelist())
    for i in itertools.count():
        img = f"{i:06d}.png"
        if img not in contents:
            break
        seg = f"{i:06d}_seg.png"
        assert seg in contents, f"not in archive: {seg}"
        lmk = f"{i:06d}_ldmks.txt"
        assert lmk in contents
        yield img, lmk, seg


def convert(zf: zipfile.ZipFile, lmk_filename: str):
    with zf.open(lmk_filename, "r") as f:
        lines = f.readlines()
    cvtline = lambda line: tuple(float(u.strip()) for u in line.split())
    lmks = np.asarray(list(map(cvtline, lines)))
    assert lmks.shape == (70, 2), f"Bad shape {lmks.shape}"
    return lmks


def generate_roi_from_points(landmarks):
    min_ = np.amin(landmarks[..., :2], axis=-2)
    max_ = np.amax(landmarks[..., :2], axis=-2)
    roi = np.concatenate([min_, max_], axis=-1).astype(np.float32)
    return roi


def generate_roi_from_seg(zf: zipfile.ZipFile, seg_filename: str):
    seg = imdecode(zf.read(seg_filename), color=False)
    mask = np.logical_or(seg == ClassIds.SKIN, seg == ClassIds.NOSE).astype(np.uint8)
    points = cv2.findNonZero(mask)
    if points is None:
        print(f"Warning ROI fallback activated for {seg_filename}")
        mask = (seg != ClassIds.BACKGROUND).astype(np.uint8)
        points = cv2.findNonZero(mask)
    # fig, ax = pyplot.subplots(1,2)
    # ax[0].imshow(mask)
    # ax[1].imshow(seg)
    # pyplot.show()
    assert points.ndim == 3 and points.shape[1] == 1 and points.shape[2] == 2
    bbox = generate_roi_from_points(points[:, 0, :])
    return bbox


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset")
    parser.add_argument("source", help="source file", type=str)
    parser.add_argument("destination", help="Destination file", type=str)
    parser.add_argument("-n", dest="count", type=int, default=None)
    args = parser.parse_args()

    with zipfile.ZipFile(args.source, "r") as zf:
        # Test
        convert(zf, "000000_ldmks.txt")
        generate_roi_from_seg(zf, "045332_seg.png")

        files = list(iterfiles(zf))

        if args.count:
            files = files[: args.count]

        relative_image_paths = np.array([a for a, _, _ in files], dtype=object)

        data = np.asarray([convert(zf, b) for _, b, _ in tqdm.tqdm(files, desc="LMK conv")])
        roi = np.asarray(
            [generate_roi_from_seg(zf, c) for _, _, c in tqdm.tqdm(files, desc="ROI gen")]
        )
        w, h = (roi[:, 2:] - roi[:, :2]).T
        ok_mask = (w > 32) & (h > 32)

        data = data[ok_mask]
        roi = roi[ok_mask]
        relative_image_paths = relative_image_paths[ok_mask]

        # Omit pupils
        data = data[:, :68, :]
        # Pad z-dimension with zeros
        data = np.concatenate([data, np.zeros((data.shape[0], 68, 1))], axis=-1)

        assert roi.shape[0] == data.shape[0] and roi.shape[1] == 4 and data.shape[1:] == (68, 3)

        with h5py.File(args.destination, "w") as f:
            create_pose_dataset(f, C.points, "pt3d_68", data=data)
            create_pose_dataset(f, C.roi, data=roi)
            ds_img = create_pose_dataset(f, C.image, count=len(relative_image_paths), lossy=True)
            for i, name in tqdm.tqdm(enumerate(relative_image_paths), desc="IMG conversion"):
                imgbuffer = imencode(imdecode(zf.read(name), color=True), quality=95)
                ds_img[i] = imgbuffer
