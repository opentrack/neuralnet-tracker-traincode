import numpy as np
import h5py
import argparse
import tqdm
from pathlib import Path
import contextlib
from scipy.spatial.transform import Rotation
import functools
from contextlib import closing
from pprint import pprint
from functools import lru_cache
from numpy.typing import NDArray
import cv2

from trackertraincode.datasets.dshdf5pose import create_pose_dataset, FieldCategory
from trackertraincode.datasets.preprocessing import depth_centered_keypoints, imread

C = FieldCategory

COLOR_FACE = (220, 57, 33)  # For segmentation mask
COLOR_BEARD = (118, 190, 70)
COLOR_CLOTHES = (135, 198, 199)
COLOR_BG = (0, 0, 0)


def map_indices(full_head_points, subset_indices):
    m = np.full(np.amax(full_head_points) + 1, fill_value=-1, dtype=np.int64)
    m[full_head_points] = np.arange(len(full_head_points))
    new_subset_indices = m[subset_indices]
    assert np.all(new_subset_indices >= 0)
    return new_subset_indices


@functools.lru_cache()
def get_landmark_indices(dataset_root: Path):
    with closing(np.load(dataset_root / "head_indices.npz")) as f:
        # Overall head. All vertices.
        head_indices = f["indices"]
    with closing(np.load(dataset_root / "landmark_indices.npz")) as f:
        # Landmarks roughly matching the 68 points scheme from 300W-LP and others.
        landmark_indices = f["indices"]
    with closing(np.load(dataset_root / "face_indices.npz")) as f:
        # Only the face area. Dense vertices. Used for creating the bounding box.
        face_indices = f["indices"]
    # Map landmark indices to reduced set of head vertices.
    new_landmark_indices = map_indices(head_indices, landmark_indices)
    new_face_indices = map_indices(head_indices, face_indices)
    return new_landmark_indices, new_face_indices


def _screen_to_image(p, img_size):
    return (1.0 - p) / 2.0 * img_size


@lru_cache(maxsize=2)
def get_image(image_filename: Path):
    return imread(str(image_filename))


@lru_cache(maxsize=2)
def get_segmentation(filename: Path) -> NDArray[np.uint8]:
    """Return RGB, uint8, HWC segmentation image."""
    seg_array = imread(str(filename))
    assert len(seg_array.shape) == 3 and seg_array.shape[-1] == 3 and seg_array.dtype == np.uint8
    return seg_array


def check_valid(image_filename: Path):
    # seg_filename = image_filename.parent / (image_filename.stem + '_mask.png')

    # seg_array = imread(str(seg_filename))
    # assert len(seg_array.shape)==3 and seg_array.shape[-1]==3 and seg_array.dtype==np.uint8
    # h, w, _ = seg_array.shape
    # num_face = np.count_nonzero(np.amax(np.abs(seg_array.astype(np.int32) - np.asarray(COLOR_FACE)),axis=-1) < 50)

    # if str(seg_filename).endswith('01792_mask.png') or str(seg_filename).endswith('00000_mask.png'):
    #     print (num_face, h*w, num_face/(h*w))
    #     from matplotlib import pyplot
    #     pyplot.imshow(seg_array)
    #     pyplot.show()

    # if num_face < h*w*0.01:
    #     # Insufficient face area
    #     return False

    image_array = get_image(image_filename)
    avg_brightness = np.average(image_array)
    if avg_brightness < 20 and np.percentile(np.ravel(np.average(image_array, axis=-1)), 98) < 20:
        # Too dark and no bright areas
        return False

    return True


def _calc_mask_for_class(seg_array, class_colors):
    return np.amax(np.abs(seg_array.astype(np.int32) - np.asarray(class_colors)), axis=-1) < 20


def generate_roi_from_points(landmarks):
    min_ = np.amin(landmarks[..., :2], axis=-2)
    max_ = np.amax(landmarks[..., :2], axis=-2)
    roi = np.concatenate([min_, max_], axis=-1).astype(np.float32)
    return roi


def roi_intersection(roi1, roi2):
    """x0y0x1y1 format. Shapes: (...,4)"""
    xymin = np.maximum(roi1[..., :2], roi2[..., :2])
    xymax = np.minimum(roi1[..., 2:], roi2[..., 2:])
    roi = np.concatenate([xymin, xymax], axis=-1)
    return roi


def generate_roi_from_seg(image_filename: Path) -> NDArray[np.float32] | None:
    seg_array = get_segmentation(image_filename)
    h, w, _ = seg_array.shape

    mask = _calc_mask_for_class(seg_array, COLOR_FACE)
    points = cv2.findNonZero(mask.astype(np.uint8))

    if points is None:
        print(f"Warning ROI fallback activated for {image_filename}")
        mask = ~(
            _calc_mask_for_class(seg_array, COLOR_CLOTHES)
            | _calc_mask_for_class(seg_array, COLOR_BG)
        )
        points = cv2.findNonZero(mask.astype(np.uint8))

    if 0:
        from matplotlib import pyplot

        fig, ax = pyplot.subplots(1, 2)
        ax[0].imshow(mask)
        ax[1].imshow(seg_array)
        pyplot.show()

    assert points.ndim == 3 and points.shape[1] == 1 and points.shape[2] == 2
    bbox = generate_roi_from_points(points[:, 0, :])

    bw, bh = bbox[2:] - bbox[:2]
    if (bw < 32 or bh < 32) or (bw > 2 * w // 3 or bh > 2 * h // 3):
        return np.zeros((4,), dtype=np.int64)
    return bbox


def convert(filename: Path):
    with contextlib.closing(np.load(filename)) as f:
        modelview = f["modelview"]
        projection = f["projection"]
        vertices = f["vertices"]
        resolution = f["resolution"]
    assert np.isclose(projection[0, 0], projection[1, 1]), "FOV should be symmetric"
    # Rotation to compensate different axis choices between blender and this project.
    rx = Rotation.from_rotvec([np.pi, 0.0, 0.0]).as_matrix()
    rx44 = np.eye(4)
    rx44[:3, :3] = rx
    # Position and size
    headbone_to_eye_center = np.asarray([0.0, -0.064, -0.086, 1.0])
    facepos3d = rx44.T @ modelview @ rx44 @ headbone_to_eye_center
    # TODO: Should the headsize be accurate to every individual sample? How would I calculate that?
    #       Currently the head shape doesn't vary a lot though.
    headradius3d = 0.1  # Hardcoded approximation for all heads. (meters)
    img_size = float(resolution)
    p = projection @ facepos3d
    p = p / p[3]
    depth = facepos3d[2]
    p[:2] = _screen_to_image(p[:2], img_size)
    # Weak perspective approximation for the image-space size of the head.
    # Note the 0.5 comes from the screen to image mapping because the image
    # spans the range [-1,1].
    p[2] = headradius3d * projection[0, 0] / depth * img_size * 0.5
    # Rotation
    quat = Rotation.from_matrix(rx.T @ modelview[:3, :3] @ rx).as_quat()
    # Vertices and bounding box
    landmark_indices, face_indices = get_landmark_indices(filename.parent)
    vertices = np.pad(vertices, [(0, 0), (0, 1)], constant_values=1.0)
    proj_vertices = (projection @ rx44.T @ modelview) @ vertices[face_indices].T
    proj_vertices /= proj_vertices[3, :]
    proj_vertices = _screen_to_image(proj_vertices[:2], img_size).T

    assert proj_vertices.ndim == 2 and proj_vertices.shape[1] == 2
    bbox = generate_roi_from_points(proj_vertices)

    # Landmarks
    landmarks = vertices[landmark_indices]  # - headbone_to_eye_center
    landmarks = (rx44.T @ modelview @ landmarks.T).T
    if 1:
        # Weak perspective projection
        landmarks = -projection[0, 0] / depth * landmarks
    else:
        landmarks = landmarks @ projection.T
        landmarks = landmarks / landmarks[:, 3:]
    landmarks = _screen_to_image(landmarks[:, :3], img_size)
    landmarks = depth_centered_keypoints(landmarks.T).T
    # print (landmarks.shape)
    # print ('mproj\n',projection)
    # print ('modelview', np.linalg.det(modelview[:3,:3]))
    # print ('p', p)
    return quat, p[:3], bbox, landmarks


def npz_to_other_files(f: Path):
    return (f.with_name(f.stem + "_img.jpg"), f.with_name(f.stem + "_mask.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset")
    parser.add_argument("source", help="source file", type=str)
    parser.add_argument("destination", help="Destination file", type=str)
    parser.add_argument("-n", dest="count", type=int, default=None)
    args = parser.parse_args()

    label_files = sorted(list(Path(args.source).glob("face_[0-9]*.npz")))

    if args.count:
        label_files = label_files[: args.count]

    label_files = np.asarray(label_files, dtype=object)

    print("processing: ", len(label_files))

    is_valid_mask = np.asarray(
        [
            check_valid(npz_to_other_files(fn)[0])
            for fn in tqdm.tqdm(label_files, desc="validity checking")
        ]
    )

    seg_rois = np.asarray(
        [
            generate_roi_from_seg(npz_to_other_files(fn)[1])
            for fn in tqdm.tqdm(label_files, desc="mask analysis")
        ]
    )

    quats, xys, pts_rois, landmarks = map(
        np.asarray, zip(*[convert(lbl) for lbl in tqdm.tqdm(label_files, desc="label conversion")])
    )

    # Points based ROI, which covers the face, just doesn't seem to be as good
    # as the segmentation bases ROI which covers visible skin of the entire head.
    # rois = roi_intersection(seg_rois, pts_rois)
    rois = seg_rois
    del seg_rois
    del pts_rois
    rw, rh = (rois[:, 2:] - rois[:, :2]).T

    is_valid_mask = is_valid_mask & (rw > 32) & (rh > 32)

    if 1:
        invalid_images = [str(fn) for fn in label_files[~is_valid_mask]]
        print(f"Invalid images: {len(invalid_images)/len(label_files)*100:0.3f}%")
        pprint(invalid_images)

    label_files = label_files[is_valid_mask]
    rois = rois[is_valid_mask]
    quats = quats[is_valid_mask]
    xys = xys[is_valid_mask]
    landmarks = landmarks[is_valid_mask]

    with h5py.File(args.destination, "w") as f:
        create_pose_dataset(f, C.quat, data=quats)
        create_pose_dataset(f, C.xys, data=xys, dtype=np.float16)
        create_pose_dataset(f, C.roi, data=rois, dtype=np.float16)
        create_pose_dataset(f, C.points, name="pt3d_68", data=landmarks, dtype=np.float16)
        ds_img = create_pose_dataset(f, C.image, count=len(label_files), lossy=True)
        for i, name in tqdm.tqdm(enumerate(label_files), desc="image conversion"):
            img_filename, _ = npz_to_other_files(name)
            with open(img_filename, "rb") as f:
                img_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            ds_img[i] = img_bytes
