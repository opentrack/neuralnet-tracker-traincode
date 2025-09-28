import itertools
import subprocess
from typing import Any, Sequence, Iterable
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import functools
from scipy.spatial.transform import Rotation
import dataclasses
from numpy.typing import NDArray
import matplotlib.patches as mpatches
from PIL import Image
import os
import copy
from collections import defaultdict
import tqdm
from concurrent.futures import (
    ProcessPoolExecutor,
)
from multiprocessing import Lock
import diskcache
import re
import ffmpeg
import h5py

from trackertraincode.facemodel.bfm import BFMModel
from trackertraincode.datasets.preprocessing import extract_image_roi
from trackertraincode.datasets.dshdf5pose import create_pose_dataset, FieldCategory

plt.rcParams["image.interpolation"] = "nearest"


body_edges = (
    np.array(
        [
            [1, 2],
            [1, 4],
            [4, 5],
            [5, 6],
            [1, 3],
            [3, 7],
            [7, 8],
            [8, 9],
            [3, 13],
            [13, 14],
            [14, 15],
            [1, 10],
            [10, 11],
            [11, 12],
        ]
    )
    - 1
)
colors = plt.cm.hsv(np.linspace(0, 1, 30)).tolist()
HDCAM_ID = 0
NUM_HDCAMS = 31
NOSE = 1
LEYE = 15
REYE = 17
LEAR = 16
REAR = 18
FACE_SIZE_FACTOR = 1.4
FACE_LEYE = [36, 37, 38, 39, 41, 40]
FACE_REYE = [42, 43, 44, 45, 47, 46]
FACE_NOT_CHIN = list(range(17, 68))
MIN_BBOX_SIZE = 64
PADDING_FRACTION = 0.25

FACE_VERTICES = BFMModel().scaled_vertices
FACE_VERTICES = Rotation.from_rotvec([np.pi, 0.0, 0.0]).apply(FACE_VERTICES)
rnd = np.random.RandomState(seed=123456)
FACE_VERTICES = np.ascontiguousarray(FACE_VERTICES[rnd.choice(len(FACE_VERTICES), size=5000)])
SPHERE_POINTS = rnd.normal(
    loc=(
        0.0,
        0.0,
        0.0,
    ),
    scale=1.0,
    size=(1000, 3),
)
SPHERE_POINTS = SPHERE_POINTS / np.linalg.norm(SPHERE_POINTS, axis=1, keepdims=True)
del rnd

VIDEOS_DIR = "hdVideos"
SHRINKED_VIDEOS_DIR = "hdVideosShrinked"


def as_hpb(rot):
    """This uses an aeronautic-like convention.

    Rotation are applied (in terms of extrinsic rotations) as follows in the given order:
    Roll - around the forward direction.
    Pitch - around the world lateral direction
    Heading - around the world vertical direction
    """
    return rot.as_euler("YXZ")


def projectPoints(X, K, R, t, Kd):
    # Copy pasted from panutils.py
    """Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].

    Roughly, x = K*(R*X + t) + distortion

    See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    or cv2.projectPoints
    """

    x = np.asarray(R * X + t)

    x[0:2, :] = x[0:2, :] / x[2, :]

    r = x[0, :] * x[0, :] + x[1, :] * x[1, :]

    x[0, :] = (
        x[0, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r)
        + 2 * Kd[2] * x[0, :] * x[1, :]
        + Kd[3] * (r + 2 * x[0, :] * x[0, :])
    )
    x[1, :] = (
        x[1, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r)
        + 2 * Kd[3] * x[0, :] * x[1, :]
        + Kd[2] * (r + 2 * x[1, :] * x[1, :])
    )

    x[0, :] = K[0, 0] * x[0, :] + K[0, 1] * x[1, :] + K[0, 2]
    x[1, :] = K[1, 0] * x[0, :] + K[1, 1] * x[1, :] + K[1, 2]

    return x


def project_points_weak_perspective(X, Xref, K, R, t, Kd):
    # Adapted from panutils.py
    """Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].

    Uses weak perspective approximation and preserves z-coordinate
    """

    x = np.asarray(R * X + t)
    xref = np.asarray(R * Xref[:, None] + t)[:, 0]

    x = x / xref[2]
    xref = xref / xref[2]

    r = xref[0] * xref[0] + xref[1] * xref[1]

    x[0, :] = (
        x[0, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r)
        + 2 * Kd[2] * xref[0] * xref[1]
        + Kd[3] * (r + 2 * xref[0] * xref[0])
    )
    x[1, :] = (
        x[1, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r)
        + 2 * Kd[3] * xref[0] * xref[1]
        + Kd[2] * (r + 2 * xref[1] * xref[1])
    )
    x[2, :] = x[2, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r)

    x[0, :] = K[0, 0] * x[0, :] + K[0, 1] * x[1, :] + K[0, 2]
    x[1, :] = K[1, 0] * x[0, :] + K[1, 1] * x[1, :] + K[1, 2]
    x[2, :] = np.sqrt(np.linalg.det(K[:2, :2])) * x[2, :]

    return x


@dataclasses.dataclass
class Pose:
    rot: Rotation
    t: NDArray[Any]
    size: float | NDArray[Any]
    valid: bool | NDArray[Any] = True

    @staticmethod
    def dummy(ndims=3):
        return Pose(Rotation.identity(), np.zeros((ndims,)), 0.0, valid=False)

    @staticmethod
    def concatenate(items: Sequence["Pose"]):
        return Pose(
            rot=Rotation.concatenate([x.rot for x in items]),
            t=np.stack([x.t for x in items]),
            size=np.asarray([x.size for x in items]),
            valid=np.asarray([x.valid for x in items], dtype="?"),
        )


class Camera:
    def __init__(self, json_data: dict[Any, Any]):
        self.json_data = json_data

    def project(self, points):
        cam = self.json_data
        shape_prefix = points.shape[:-1]
        points = np.reshape(points[..., :3], (-1, 3))
        proj = projectPoints(points.T, cam["K"], cam["R"], cam["t"], cam["distCoef"])[:2, ...].T
        return np.reshape(proj, (*shape_prefix, 2))

    def project_weak_perspective(self, points, ref_point):
        cam = self.json_data
        shape_prefix = points.shape[:-1]
        points = np.reshape(points[..., :3], (-1, 3))
        proj = project_points_weak_perspective(
            points.T, ref_point, cam["K"], cam["R"], cam["t"], cam["distCoef"]
        ).T
        return np.reshape(proj, (*shape_prefix, 3))

    def project_pose(self, pose: Pose) -> Pose:
        if not pose.valid:
            return Pose.dummy(ndims=2)
        cam = self.json_data
        # Beware, "R" is the transform from world to camera space.
        crot = cam["R"]
        eps = 1.0e-3
        # Pad with zero vector to also transform the original center.
        # The other vectors are slightly offset in the direction of the camera axes.
        p = pose.t[None, :] + eps * np.pad(crot.T, [(0, 0), (0, 1)]).T
        p = projectPoints(p.T, cam["K"], cam["R"], cam["t"], cam["distCoef"]).T
        mask = (
            (p[..., 0] > 0) & (p[..., 1] > 0) & (p[..., 0] < self.width) & (p[..., 1] < self.height)
        )
        pose_z_coord = (crot @ pose.t[:, None] + cam["t"])[2, 0]
        is_in_frustum = np.all(mask) & (pose_z_coord > pose.size)
        proj_center = p[-1, :]
        proj_delta = (p[:-1, :] - proj_center[None, :]) / eps
        # Determinant of the transformed trapezoid gives approximately the projected size squared.
        avg_scale = np.sqrt(np.abs(np.linalg.det(proj_delta[:2, :2])))
        rotation = Rotation.from_matrix(crot) * pose.rot
        return Pose(rotation, proj_center[:2], avg_scale * pose.size, valid=is_in_frustum)

    def perspective_corrected_rotation(self, world_position: NDArray[Any], rot: Rotation):
        r"""
            Explanation though top view
                                       ^ face-local z-axis
                         z-axis ^      |   ^ direction under which the CNN "sees" the face through it's crop
                                |     _|__/
                                |    /    \
                                |   | face |
                                |    \ __ /
                                |     /         Note: <----> marks the face crop
                                |    /
           -----------------------<-x->-------------- screen
                                |  / xy_normalized
                              f | /
                                |/
                        camera  x ------> x-axis

            Thus, it is apparent that the CNN sees the face approximately under an angle spanned by the forward
            direction and the 3d position of the face. The more wide-angle the lense is the stronger the effect.
            As usual perspective distortion within the crop is neglected.
            Hence, we assume that the detected rotation is given w.r.t to a coordinate system whose z-axis is
            aligned with the position vector as illustrated. Consequently, the resulting pose is simply the
            cnn-output transformed into the world coordinate system.

            Beware, position correction is handled in the evaluation scripts. It's much simpler as we only have
            to consider the offset and scaling due to the cropping and resizing to the CNN input size.

        Returns:
            Updated rotations.
        """
        cam = self.json_data
        cam_position = ((cam["R"] @ world_position[:, None]) + cam["t"]).A[:, 0]
        m = _make_look_at_matrix(cam_position)
        # print("cp = ", cam_position)
        # print("m = \n", m)
        # print(np.linalg.det(m))
        # print(m @ m.T)
        out = Rotation.from_matrix(m).inv() * rot
        return out

    @property
    def width(self):
        return self.json_data["resolution"][0]

    @property
    def height(self):
        return self.json_data["resolution"][1]

    @property
    def id(self):
        return self.json_data["node"]


def _make_look_at_matrix(pos: NDArray[Any]):
    """Computes a rotation matrix where the z axes is aligned with the argument vector.

    This leaves a degree of rotation around the this axis. This is resolved by constraining
    the x axis to the horizonal plane (perpendicular to the global y-axis).
    """
    z = pos / np.linalg.norm(pos, axis=-1, keepdims=True)
    x = np.cross(*np.broadcast_arrays(np.asarray([0.0, 1.0, 0.0]), z), axis=-1)
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    y = np.cross(z, x, axis=-1)
    y = y / np.linalg.norm(x, axis=-1, keepdims=True)
    M = np.stack([x, y, z], axis=-1)
    return M


class PoseUnreliable(Exception):
    pass


@dataclasses.dataclass
class Body:
    id: str
    # xyz and confidence
    points: NDArray[np.float64]
    face_points: NDArray[np.float64]
    # Shape (#hd cams, #points)
    face_points_visibility: NDArray[Any]
    _rot: dataclasses.InitVar[Rotation | None]
    head_pose: Pose = dataclasses.field(init=False)

    def __post_init__(self, _rot: Rotation | None):
        assert self.face_points is not None
        assert _rot is not None
        self.head_pose = self.__head_pose(_rot)
        self.head_pose.valid = self.__head_pose_is_confident()

    def __head_pose_is_confident(self):
        ref_points = self.points[[LEYE, REYE, LEAR, REAR], :3]
        skull_center = np.average(ref_points, axis=0)
        skull_radius = 0.5 * np.average(np.linalg.norm(ref_points - skull_center, axis=-1))
        face_points_visible = np.all(np.any(self.face_points_visibility[:, FACE_NOT_CHIN], axis=0))
        points_in_face_area = np.all(
            np.linalg.norm(self.face_points[FACE_NOT_CHIN, :] - skull_center, axis=-1)
            < 3 * skull_radius
        )

        skeleton_points_are_confident = np.all(self.points[[LEYE, REYE, LEAR, REAR, NOSE], 3] > 0.1)

        lear, rear = self.points[[LEAR, REAR], :3]
        x_axis_by_landmarks = lear - rear
        x_axis = self.head_pose.rot.as_matrix()[:, 0]
        x_axis_is_aligned = np.inner(x_axis, x_axis_by_landmarks) > 0.8 * np.linalg.norm(
            x_axis_by_landmarks
        )

        return (
            face_points_visible
            and points_in_face_area
            and skeleton_points_are_confident
            and x_axis_is_aligned
        )

    def __head_pose(self, rot: Rotation):
        l, r = self.points[[LEYE, REYE], :3]
        center = 0.5 * (l + r)
        l, r = self.points[[LEAR, REAR], :3]
        size = 0.5 * FACE_SIZE_FACTOR * np.linalg.norm(l - r)
        # center -= rot.apply(np.array([0., -0.26, -0.9])) * size
        return Pose(rot, center, size)

    def face_vertices_for_bbox(self):
        l, r = self.points[[LEAR, REAR], :3]
        center = 0.5 * (l + r)
        l, r = self.points[[LEAR, REAR], :3]
        size = 0.5 * np.linalg.norm(l - r)
        # size = np.asarray([size,size*FACE_SIZE_FACTOR,size*FACE_SIZE_FACTOR])
        vertices2 = (
            size * self.head_pose.rot.apply(SPHERE_POINTS + np.asarray([0.0, 0.25, 0.0])) + center
        )
        vertices1 = self.head_pose.size * self.head_pose.rot.apply(FACE_VERTICES) + self.head_pose.t
        return np.concatenate([vertices1, vertices2])

    def guestimate_head_bounding_box(self, cam: Camera) -> NDArray[np.float64]:
        """xyxy format."""
        # if not self.head_pose_is_confident():
        #     raise PoseUnreliable()
        # leye, reye, lear, rear = self.points[[LEYE, REYE, LEAR, REAR], :3]
        # skull_center = 0.5 * (lear + rear)
        # skull_radius = 0.5 * np.linalg.norm(lear - rear)
        # up_vec = self._rot.as_matrix()[:, 1]
        # skull_center += 0.75 * skull_radius * up_vec
        # point_visible = np.any(self.face_points_visibility, axis=0)
        # point_in_face_area = np.linalg.norm(self.face_points - skull_center, axis=-1) < 3 * skull_radius
        # good_points = self.face_points[point_visible & point_in_face_area]
        # proj_face = cam.project(good_points)
        # proj_skull = cam.project_pose(Pose(rot=Rotation.identity(), t=skull_center, size=skull_radius))
        # skull_points = proj_skull.t + proj_skull.size * np.asarray(
        #     [[-1, 1], [1, 1], [1, -1], [-1, -1]], dtype=np.float64
        # )
        # pts = np.concatenate([proj_fawce, skull_points], axis=0)
        pts = self.face_vertices_for_bbox()
        pts = cam.project(pts)
        xy_min = np.amin(pts, axis=0)
        xy_max = np.amax(pts, axis=0)
        return np.concatenate([xy_min, xy_max], axis=-1)


class Bodies:
    def __init__(self, directory: Path, frame_num: int):
        fn = directory / "hdPose3d_stage1_coco19" / f"body3DScene_{frame_num:08}.json"
        with open(fn) as f:
            skeletons = dict(self.__parse_skeleton(json.load(f)))
        fn = directory / "meshTrack_face" / f"meshTrack_{frame_num:08}.txt"
        with open(fn) as f:
            face_fits = dict(self.__parse_face_raw_fit(f.read()))
        fn = directory / "hdFace3d" / f"faceRecon3D_hd{frame_num:08d}.json"
        with open(fn) as f:
            landmarks = dict(self.__parse_face(json.load(f)))

        self.individuals: list[str] = list(
            set(skeletons.keys()) & set(face_fits.keys()) & set(landmarks.keys())
        )
        self.bodies = {
            id: Body(
                id,
                points=skeletons[id],
                face_points=landmarks[id][0],
                _rot=face_fits[id],
                face_points_visibility=landmarks[id][1],
            )
            for id in self.individuals
        }

    def __parse_skeleton(self, json_skel: dict[Any, Any]):
        for body in json_skel["bodies"]:
            individual = body["id"]
            points = np.array(body["joints19"]).reshape((-1, 4))
            yield individual, points

    def __parse_face_raw_fit(self, face_raw: str):
        lines = face_raw.splitlines()
        # Skip header
        lines = lines[2:]
        for i, maybe_face_hdr in enumerate(lines):
            if not maybe_face_hdr.startswith("Face"):
                continue
            individual = int(lines[i - 5].strip())
            mrot = Rotation.from_rotvec(
                np.asarray([float(v.strip()) for v in lines[i + 2].split()])
            )
            yield individual, mrot

    def __parse_face(self, json_face: dict[Any, Any]):
        for face in json_face["people"]:
            individual = face["id"]
            # There is some dummy data
            if individual < 0:
                continue
            lmks = np.array(face["face70"]["landmarks"]).reshape((-1, 3))
            visibility_mask = np.zeros((NUM_HDCAMS, lmks.shape[0]), dtype="?")
            for point_idx, cam_ids in enumerate(face["face70"]["visibility"]):
                visibility_mask[cam_ids, point_idx] = True
            yield individual, (lmks, visibility_mask)


class PanopticSequence:
    re_extract_body_frame_num = re.compile(r"body3DScene_(\d*).json")
    re_extract_face_track_frame_num = re.compile(r"meshTrack_(\d*).txt")
    re_extract_landmark_frame_num = re.compile(r"faceRecon3D_hd(\d*).json")

    def __init__(self, directory: str | Path):
        directory = Path(directory)
        with open(next(iter(directory.glob("calibration_*.json")))) as cfile:
            calib = json.load(cfile)
        # Cameras are identified by a tuple of (panel#,node#)
        cameras = {(int(cam["panel"]), int(cam["node"])): cam for cam in calib["cameras"]}

        # Convert data into numpy arrays for convenience
        for k, cam in cameras.items():
            cam["K"] = np.matrix(cam["K"])
            cam["distCoef"] = np.array(cam["distCoef"])
            cam["R"] = np.matrix(cam["R"])
            cam["t"] = np.array(cam["t"]).reshape((3, 1))

        self.cameras = {k: Camera(v) for k, v in cameras.items() if k[0] == HDCAM_ID}
        self.directory = directory
        self.frames_nums = self.__discover_frames()

    def __discover_frames(self):
        path = self.directory / "hdPose3d_stage1_coco19"
        if not path.is_dir():
            raise ValueError(f"Sequence {self.directory} is missing the hd pose dir")
        body_frames = set(
            int(self.re_extract_body_frame_num.match(abspath.name).group(1))
            for abspath in path.iterdir()
        )
        path = self.directory / "meshTrack_face"
        if not path.is_dir():
            raise ValueError(f"Sequence {self.directory} is missing the meshTrack_face dir")
        face_track_frames = set(
            int(self.re_extract_face_track_frame_num.match(abspath.name).group(1))
            for abspath in path.iterdir()
        )
        path = self.directory / "hdFace3d"
        if not path.is_dir():
            raise ValueError(f"Sequence {self.directory} is missing the hdFace3d dir")
        landmark_frames = set(
            int(self.re_extract_landmark_frame_num.match(abspath.name).group(1))
            for abspath in path.iterdir()
        )
        framelist = list(body_frames.intersection(face_track_frames).intersection(landmark_frames))
        assert framelist, f"Label files missing in {self.directory}"
        return framelist

    @functools.lru_cache(maxsize=10000)
    def get_body_pose_data(self, frame_num: int):
        return Bodies(self.directory, frame_num)


@functools.lru_cache()
def CachedPanopticSequence(sequence_dir: str | Path):
    return PanopticSequence(sequence_dir)


def visu_body(ax, body: Body, cam: Camera):
    skel = body.points.T
    # Show only points detected with confidence
    valid = skel[3, :] > 0.1

    pt = cam.project(body.points).T

    ax.plot(pt[0, valid], pt[1, valid], ".", color=colors[body.id])

    # Plot edges for each bone
    for edge in body_edges:
        if valid[edge[0]] or valid[edge[1]]:
            ax.plot(pt[0, edge], pt[1, edge], color=colors[body.id])

    # Show the joint numbers
    for ip in range(pt.shape[1]):
        if pt[0, ip] >= 0 and pt[0, ip] < cam.width and pt[1, ip] >= 0 and pt[1, ip] < cam.height:
            ax.text(pt[0, ip], pt[1, ip] - 5, "{0}".format(ip), color=colors[body.id])

    fps = cam.project(body.face_points).T
    visible = body.face_points_visibility[cam.id]

    ax.scatter(fps[0, visible], fps[1, visible], color=colors[body.id], s=1.0)
    ax.scatter(fps[0, ~visible], fps[1, ~visible], color="r", s=1.0)

    if body.head_pose.valid:
        pose = body.head_pose
        rot, xy, scale, _ = dataclasses.astuple(cam.project_pose(pose))

        circle = mpatches.Circle(xy, scale, ec="w", fc="none")
        ax.add_artist(circle)

        axis_scale = 10.0  # cm?
        xyz_proj = cam.project(pose.t[None, :] + axis_scale * pose.rot.as_matrix().T)
        for e, c in zip(xyz_proj, "rgb"):
            ax.plot([xy[0], e[0]], [xy[1], e[1]], color=c)

        bbox = body.guestimate_head_bounding_box(cam)

        rect = mpatches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], ec="r", fc="none")
        ax.add_artist(rect)

        boxpoints = cam.project(body.face_vertices_for_bbox())
        ax.scatter(boxpoints.T[0], boxpoints.T[1], color=colors[body.id], s=0.1)


def vis_one(sequence: Path | str, frame_num: int, cam_id: int):
    sequence = Path(sequence)
    panseq = PanopticSequence(sequence)
    images = ImageExtractor(sequence.parent)
    pic = images.get(sequence.name, frame_num, cam_id)
    frame = panseq.get_body_pose_data(frame_num)
    cam = panseq.cameras[HDCAM_ID, cam_id]
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.set(xlim=(0, cam.width), ylim=(cam.height, 0))
    ax.imshow(pic)
    ax.set_autoscale_on(False)
    for id, body in frame.bodies.items():
        visu_body(ax, body, cam)
    plt.show()


class ImageExtractor:
    def __init__(self, root: str | Path, use_shrinked_videos: bool = False):
        self._root = Path(root)
        self._extract_single_frame_mutex = Lock()
        self._viddir = SHRINKED_VIDEOS_DIR if use_shrinked_videos else VIDEOS_DIR

    def _ensure_frames_dir(self, sequence_dir: str | Path, camera_id: int):
        path = self._root / sequence_dir / self._viddir / f"frames_00_{camera_id:02}"
        os.makedirs(path, exist_ok=True)
        return path

    @functools.lru_cache(maxsize=128)
    def get(self, sequence_dir: str | Path, hdframe_id: int, camera_id: int):
        with self._extract_single_frame_mutex:
            frame_fn = self._ensure_frames_dir(sequence_dir, camera_id) / f"{hdframe_id:08}.jpg"
            if not frame_fn.exists():
                self._extract_single_to_file(sequence_dir, hdframe_id, camera_id)
            return Image.open(frame_fn)

    def _extract_single_to_file(self, sequence_dir: str | Path, hdframe_id: int, camera_id: int):
        panel = 0
        video_fn = (
            self._root / Path(sequence_dir) / self._viddir / f"hd_{panel:02}_{camera_id:02}.mp4"
        )
        frame_fn = self._ensure_frames_dir(sequence_dir, camera_id) / f"{hdframe_id:08}.jpg"
        out, _ = (
            ffmpeg.input(video_fn)
            .filter("select", "eq(n,{})".format(hdframe_id))
            .output(str(frame_fn), vframes=1, qscale=2, pix_fmt="rgb24")
            .run(capture_stdout=True)
        )

    def probe_video_info(self, sequence_dir: str | Path, camera_id: int):
        panel = 0
        video_fn = (
            self._root / Path(sequence_dir) / self._viddir / f"hd_{panel:02}_{camera_id:02}.mp4"
        )
        if not video_fn.exists():
            raise RuntimeError(f"Video missing: {video_fn}")
        probe = ffmpeg.probe(video_fn)
        video_stream = next(
            (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
            None,
        )
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        nb_frames = int(video_stream["nb_frames"])
        return nb_frames, width, height

    def stream_frames(self, sequence_dir: str | Path, camera_id: int, max_num_frames: int | None):
        nb_frames, width, height = self.probe_video_info(sequence_dir, camera_id)
        if max_num_frames is None:
            max_num_frames = nb_frames

        panel = 0
        video_fn = (
            self._root / Path(sequence_dir) / self._viddir / f"hd_{panel:02}_{camera_id:02}.mp4"
        )
        print("Streaming video: ", video_fn)
        process1 = (
            ffmpeg.input(video_fn)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24", vframes=max_num_frames)
            .run_async(pipe_stdout=True)
        )
        while True:
            in_bytes = process1.stdout.read(width * height * 3)
            if not in_bytes:
                print("Done")
                break
            in_frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
            yield in_frame
        process1.stdout.close()
        process1.wait()


@dataclasses.dataclass
class CropLabel:
    body: Body
    world_pose: Pose
    pose: Pose
    rect: NDArray[Any]
    landmarks: NDArray[Any]


def extract_crop(
    img,
    lbl: CropLabel,
):
    patch, offset = extract_image_roi(
        np.asarray(img),
        lbl.rect,
        padding_fraction=PADDING_FRACTION,
        square=True,
        return_offset=True,
    )

    lbl = copy.deepcopy(lbl)

    lbl.pose.t += offset
    lbl.rect[:2] += offset
    lbl.rect[2:] += offset
    lbl.landmarks[:, :2] += offset
    return patch, lbl


class FaceCropAnalysis:
    def __init__(self, bodies: Sequence[Bodies]):
        self._bodies = list(bodies)

    def compute_projections(self, cam: Camera) -> list[CropLabel]:
        def process_one(body: Body) -> CropLabel | None:
            pose = body.head_pose
            ppose = cam.project_pose(pose)
            if not ppose.valid:
                return None
            ppose.rot = cam.perspective_corrected_rotation(pose.t, ppose.rot)
            bbox = body.guestimate_head_bounding_box(cam)
            landmarks = cam.project_weak_perspective(body.face_points, pose.t)
            return CropLabel(body, pose, ppose, bbox, landmarks)

        return list(
            filter(
                lambda x: x is not None,
                (process_one(b) for b in self._bodies),
            )
        )

    def compute(self, cam: Camera):
        labels = self.compute_projections(cam)
        valid_visibility = FaceCropAnalysis._guestimate_not_self_occlusion(labels, cam.id)
        valid_visibility = valid_visibility & FaceCropAnalysis._compute_valid_bounding_boxes(labels)
        return labels, valid_visibility

    @staticmethod
    def _guestimate_not_self_occlusion(labels: list[CropLabel], cam_id: int) -> NDArray[np.bool_]:
        ANGLE_THRESHOLD = 45.0 / 180.0 * np.pi
        MIN_VIS_POINTS = (68 * 1) // 3
        COS_ANGLE_THRESHOLD = np.cos(ANGLE_THRESHOLD)
        z_vec = np.asarray([0.0, 0.0, 1.0])
        if labels:
            cos_angles: NDArray[np.float64] = -np.dot(
                # Dot is (N,3) x (3) matrix.
                Rotation.concatenate([p.pose.rot for p in labels]).as_matrix()[:, :, 2],
                z_vec,
            )
            num_vis_points = np.asarray(
                [np.count_nonzero(label.body.face_points_visibility[cam_id]) for label in labels]
            )
            return (cos_angles < COS_ANGLE_THRESHOLD) | (num_vis_points >= MIN_VIS_POINTS)
        else:
            return np.zeros((0,), dtype="?")

    @staticmethod
    def _compute_valid_bounding_boxes(labels: list[CropLabel]):
        results = np.zeros((len(labels),), dtype="?")
        for i, l in enumerate(labels):
            bbox = l.rect
            sizes = bbox[2:] - bbox[:2]
            if np.all(sizes > MIN_BBOX_SIZE):
                results[i] = True
        return results


def is_image_reasonable(crop: NDArray[np.uint8]):
    """If the frame is mostly uniform it probably doesn't have a person in it."""
    return np.any(np.std(crop, axis=(0, 1)) > 5.0)


def vis_crop_labels(ax, pose: Pose, bbox: NDArray[Any], landmarks: NDArray[Any], valid: bool):
    rw = {True: "w", False: "r"}[valid]
    circle = mpatches.Circle(pose.t, pose.size, ec=rw, fc="none")
    ax.add_artist(circle)

    ax.scatter(landmarks.T[0], landmarks.T[1], s=3.0, color="w")

    axis_scale = 30.0  # pixels
    xyz_proj = pose.t[None, :] + axis_scale * pose.rot.as_matrix().T[:, :2]
    for e, c in zip(xyz_proj, "rgb"):
        ax.plot([pose.t[0], e[0]], [pose.t[1], e[1]], color=c)

    rect = mpatches.Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1], ec=rw, fc="none")
    ax.add_artist(rect)


def vis_crop_extract(sequence_dir: str | Path, frame_id: int):
    extractor = ImageExtractor(Path(sequence_dir).parent)
    panseq = PanopticSequence(sequence_dir)
    bodies = panseq.get_body_pose_data(frame_id).bodies.values()
    crop_analysis = FaceCropAnalysis(bodies)

    ncams = 30
    fig, axes = plt.subplots(ncams, len(bodies), figsize=(len(bodies) * 5.0, ncams * 5.0))
    if len(axes.shape) == 1:
        axes = axes[:, None]

    for (_, cam_id), cam in panseq.cameras.items():
        if cam_id >= ncams:
            break
        labels, valid_viss = crop_analysis.compute(cam)
        for i, (label, valid_vis) in enumerate(zip(labels, valid_viss)):
            img = extractor.get(sequence_dir, frame_id, cam_id)
            crop, label = extract_crop(img, label)
            axes[cam_id, i].imshow(crop)
            vis_crop_labels(
                axes[cam_id, i],
                label.pose,
                label.rect,
                label.landmarks,
                valid_vis and is_image_reasonable(crop),
            )
    plt.savefig("/tmp/plot.pdf")


def iterate_crops(
    sequence_dir: str | Path,
    cam_id: int,
    max_num_frames: int | None = None,
    every: int = 60,
    use_shrinked_videos: bool = True,
):
    extractor = ImageExtractor(Path(sequence_dir).parent, use_shrinked_videos=use_shrinked_videos)
    panseq = CachedPanopticSequence(sequence_dir)
    labeled_frame_ids = frozenset(panseq.frames_nums)
    cam = panseq.cameras[HDCAM_ID, cam_id]
    nb_frames, _, _ = extractor.probe_video_info(sequence_dir, cam.id)
    max_num_frames = min(nb_frames, max_num_frames) if max_num_frames is not None else nb_frames
    for frame_num, frame_img in enumerate(
        extractor.stream_frames(sequence_dir, cam.id, max_num_frames=max_num_frames)
    ):
        if (not frame_num in labeled_frame_ids) or (not (frame_num % every == 0)):
            continue
        bodies = panseq.get_body_pose_data(frame_num)
        crop_analysis = FaceCropAnalysis(bodies.bodies.values())
        labels, valid_viss = crop_analysis.compute(cam)
        for label, valid_vis, body in zip(labels, valid_viss, bodies.bodies.values()):
            crop, label = extract_crop(frame_img, label)
            if not valid_vis or not is_image_reasonable(crop):
                continue
            yield crop, label, body.id, frame_num


def write_dataset_piece(
    out_fn: str | Path,
    sequence_dir: str | Path,
    cam_id: int,
    max_num_frames: int | None,
    use_shrinked_videos: bool,
):
    CachedPanopticSequence(
        sequence_dir
    )  # Check if it can be read. Prevent overwriting by hdf5 in case filenames are swapped.
    quats = []
    rects = []
    xy = []
    sizes = []
    landmarks = []
    individuals = []
    frame_nums = []
    images = []
    rot_correction = Rotation.from_rotvec([np.pi, 0.0, 0.0])
    for crop, label, individual, frame_num in iterate_crops(
        sequence_dir, cam_id, max_num_frames, use_shrinked_videos=use_shrinked_videos
    ):
        # n = len(ds_img)
        # if n <= i:
        #     ds_img.resize(2 * n, axis=0)
        # ds_img[i] = crop
        images.append(crop)
        quats.append((label.pose.rot * rot_correction).as_quat())
        rects.append(label.rect)
        xy.append(label.pose.t)
        sizes.append(label.pose.size)
        landmarks.append(label.landmarks[:68])
        individuals.append(individual)
        frame_nums.append(frame_num)

    quats, rects, xy, sizes, landmarks = map(
        lambda x: np.stack(x, axis=0), [quats, rects, xy, sizes, landmarks]
    )
    images = np.asarray(images, dtype=object)
    individuals = np.asarray(individuals, dtype="i1")
    frame_nums = np.asarray(frame_nums, "i4")
    xys = np.concatenate([xy, sizes[:, None]], axis=-1)
    N = len(images)

    order = np.argsort(
        frame_nums.astype(np.int64) + np.amax(frame_nums) * individuals.astype(np.int64)
    )
    quats, rects, xys, landmarks, images, individuals, frame_nums = map(
        lambda x: x[order], [quats, rects, xys, landmarks, images, individuals, frame_nums]
    )

    with h5py.File(str(out_fn), "w") as f:
        ds_img = create_pose_dataset(f, FieldCategory.image, count=N)
        for i, img in enumerate(images):
            ds_img[i] = img
        create_pose_dataset(f, FieldCategory.roi, data=rects, dtype="f2")
        create_pose_dataset(f, FieldCategory.quat, data=quats, dtype="f4")
        create_pose_dataset(f, FieldCategory.xys, data=xys, dtype="f4")
        # Face landmarks are wildly inaccurate. Therefore they are not saved.
        create_pose_dataset(f, FieldCategory.general, name="individual", data=individuals)
        f.create_dataset("frame", data=frame_nums)
        f.create_dataset(
            "sequence",
            shape=(N,),
            data=np.asarray([Path(sequence_dir).name.encode("ascii")], dtype="|S32").repeat(N),
        )
        f.create_dataset("cam", shape=(N,), data=np.asarray([cam_id], dtype="i1").repeat(N))


def write_dataset_pieces(
    out_dir: str | Path,
    sequence_dirs: list[str | Path],
    max_num_frames: int | None,
    use_shrinked_videos: bool,
):
    os.makedirs(out_dir, exist_ok=True)
    for sequence_dir in sequence_dirs:
        for cam in CachedPanopticSequence(sequence_dir).cameras.values():
            out_fn = Path(out_dir) / f"{Path(sequence_dir).name}_hdcam_{cam.id:02}.h5"
            if out_fn.exists():
                print(f"Skipped existing {out_fn}")
                continue
            write_dataset_piece(out_fn, sequence_dir, cam.id, max_num_frames, use_shrinked_videos)


# @dataclasses.dataclass
# class IndividualSampleInfo:
#     pose: Pose
#     sequence_id: str
#     frame_id: str
#     camera_id: int

# TODO: Can go into an extra script?
# class IndividualSampler:
#     def __init__(self):
#         self._cache_dir = "/tmp/cmu-processing-cache"

#     def my_cache(self):
#         return diskcache.Cache(self._cache_dir)

#     def collect_individual_rotations(self, paths: list[Path | str]):
#         executor = ProcessPoolExecutor(max_workers=os.cpu_count())
#         args = []
#         for path in map(Path, paths):
#             panseq = PanopticSequence(path)
#             # Collect one frame every 2 sec.
#             frame_num_iter = list(panseq.frames_nums)[::60]
#             for frame_id in frame_num_iter:
#                 args.append((panseq, frame_id))

#         by_individual = defaultdict(list)
#         for d in tqdm.tqdm(
#             executor.map(lambda arg: self._process_frame(*arg), args, chunksize=32),
#             # map(self._process_frame, args),
#             total=len(args),
#             desc="Frames",
#         ):
#             for k, v in d.items():
#                 by_individual[k] += v

#         print("total faces:", sum([len(v) for v in by_individual.values()]))

#         with self.my_cache() as c:
#             c["paths"] = paths
#             c["by_individual"] = by_individual

#     @staticmethod
#     def _process_frame(panseq: PanopticSequence, frame_id: int):
#         by_individual = defaultdict(list)
#         frame = panseq.get_body_pose_data(frame_id)
#         poses = [safe_guestimate_head_pose_w_dummy(b) for b in frame.bodies.values()]
#         for (hd_flag, cam_id), cam in panseq.cameras.items():
#             if hd_flag != HDCAM_ID:
#                 continue
#             proj_poses = [
#                 cam.project_pose(pose, raise_on_failure=False) for pose in poses
#             ]
#             # valid_mask = IndividualSampler._compute_non_overlapping_poses_mask(
#             #     proj_poses
#             # )
#             valid_mask = np.asarray([p.valid for p in proj_poses], dtype="?")
#             valid_mask = valid_mask & IndividualSampler._guestimate_not_self_occlusion(
#                 frame.bodies.values(), proj_poses
#             )
#             valid_mask = valid_mask & IndividualSampler._compute_valid_bounding_boxes(
#                 frame.bodies.values(), cam
#             )
#             for pose, individual in (
#                 (p, i)
#                 for (p, m, i) in zip(proj_poses, valid_mask, frame.bodies.keys())
#                 if m
#             ):
#                 by_individual[individual].append(
#                     IndividualSampleInfo(
#                         pose,
#                         sequence_id=panseq.directory,
#                         frame_id=frame_id,
#                         camera_id=cam_id,
#                     )
#                 )
#         return by_individual

#     def visu_all_frames(self):
#         with self.my_cache() as c:
#             by_individual: dict[str, list[IndividualSampleInfo]] = c["by_individual"]
#         fig, axes = plt.subplots(
#             len(by_individual), 3, figsize=(15, len(by_individual) * 5.0)
#         )
#         for ax, (k, v) in zip(axes, by_individual.items()):
#             rots = Rotation.concatenate([sample.pose.rot for sample in v])
#             hpb = as_hpb(rots)
#             for i in range(3):
#                 ax[i].hist(hpb[:, i], bins=90, label=f" {k}")
#             ax[0].legend()
#         for i, name in zip(range(3), "hpb"):
#             axes[-1, i].set(xlabel=name)
#         plt.show()


def shrink_videos(directories: list[str]):
    # Don't use. Looks like it impacts the model accuracy.
    for directory in directories:
        directory = Path(directory)
        os.makedirs(directory / SHRINKED_VIDEOS_DIR, exist_ok=True)
        for input in (directory / VIDEOS_DIR).glob("*.mp4"):
            output = directory / SHRINKED_VIDEOS_DIR / input.name
            if output.exists():
                print("Skipped ", input)
                continue
            # print(input, output)
            subprocess.check_call(
                [
                    "ffmpeg",
                    "-i",
                    input,
                    "-c:v",
                    "libx264",
                    "-b:v",
                    "4M",
                    "-pass",
                    "1",
                    "-an",
                    "-f",
                    "null",
                    "/dev/null",
                ]
            )
            subprocess.check_call(
                [
                    "ffmpeg",
                    "-i",
                    input,
                    "-c:v",
                    "libx264",
                    "-b:v",
                    "4M",
                    "-pass",
                    "2",
                    "-minrate",
                    "1M",
                    "-maxrate",
                    "8M",
                    output,
                ]
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("roots", nargs="*")
    subparsers = parser.add_subparsers()
    vis_parser = subparsers.add_parser("vis")
    vis_parser.add_argument("sequence_root")
    vis_parser.add_argument("cam", type=int)
    vis_parser.add_argument("frame_num", type=int)
    vis_parser.set_defaults(func=lambda args: vis_one(args.sequence_root, args.frame_num, args.cam))

    vis_parser2 = subparsers.add_parser("vis-crops")
    vis_parser2.add_argument("sequence_root")
    vis_parser2.add_argument("frame_num", type=int)
    vis_parser2.set_defaults(func=lambda args: vis_crop_extract(args.sequence_root, args.frame_num))

    create_piece_parser = subparsers.add_parser("create-piece")
    create_piece_parser.add_argument("sequence_root")
    create_piece_parser.add_argument("cam", type=int)
    create_piece_parser.add_argument("output")
    create_piece_parser.add_argument("-n", type=int, default=None)
    create_piece_parser.add_argument(
        "--sv", action="store_true", default=False, help="Use shrinked videos"
    )
    create_piece_parser.set_defaults(
        func=lambda args: write_dataset_piece(
            args.output, args.sequence_root, args.cam, args.n, args.sv
        )
    )

    create_all_parser = subparsers.add_parser("create-pieces")
    create_all_parser.add_argument("roots", nargs="*")
    create_all_parser.add_argument("output")
    create_all_parser.add_argument("-n", type=int, default=None)
    create_all_parser.add_argument(
        "--sv", action="store_true", default=False, help="Use shrinked videos"
    )
    create_all_parser.set_defaults(
        func=lambda args: write_dataset_pieces(args.output, args.roots, args.n, args.sv)
    )

    shrink_videos_parser = subparsers.add_parser("shrink-videos")
    shrink_videos_parser.add_argument("directories", nargs="*")
    shrink_videos_parser.set_defaults(func=lambda args: shrink_videos(args.directories))

    # stats_parser = subparsers.add_parser("stats")
    # stats_parser.add_argument("sequence_dirs", nargs="*")
    # stats_parser.set_defaults(
    #     func=lambda args: IndividualSampler().collect_individual_rotations(
    #         args.sequence_dirs
    #     )
    # )

    # stats_parser = subparsers.add_parser("stats-vis")
    # stats_parser.set_defaults(func=lambda args: IndividualSampler().visu_all_frames())

    args = parser.parse_args()
    args.func(args)
