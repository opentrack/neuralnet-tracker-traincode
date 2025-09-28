import abc
import os
import numpy as np
from os.path import join, dirname, basename, splitext
import argparse
import zipfile
import io
import h5py
import scipy.io
import cv2
import collections
from typing import List, cast
import re
import tqdm
import torch

from trackertraincode.datasets.preprocessing import (
    imdecode,
    compute_keypoints,
    depth_centered_keypoints,
    move_aflw_head_center_to_between_eyes,
    sanity_check_landmarks,
    get_3ddfa_shape_parameters,
)
from trackertraincode.datasets.dshdf5pose import create_pose_dataset, FieldCategory
from trackertraincode.neuralnets.rotrepr import QuatRepr
from trackertraincode.utils import aflw_rotation_conversion
from trackertraincode.facemodel.bfm import ScaledBfmModule, BFMModel
from trackertraincode.neuralnets.modelcomponents import PosedDeformableHead

C = FieldCategory


def discover_samples(zf):
    names = frozenset(['AFW', 'HELEN', 'IBUG', 'LFPW'])
    isInDataSubsets = lambda s: s.split(os.path.sep)[1] in names
    filenames = [
        f.filename
        for f in zf.filelist
        if (f.external_attr == 0x20 and splitext(f.filename)[1] == '.mat' and isInDataSubsets(f.filename))
    ]
    return sorted(filenames)


def remove_artificially_rotated_faces(filenames: List[str]):
    return list(filter(lambda fn: fn.endswith('_0.mat'), filenames))


def remove_original_faces(filenames: List[str]):
    return list(filter(lambda fn: not fn.endswith('_0.mat'), filenames))


def make_groups(filenames: List[str]):
    regex = re.compile(r'([\w| ]+)_(\d+).mat')
    d = collections.defaultdict(list)
    for fn in filenames:
        match = regex.match(os.path.basename(fn))
        assert match is not None, f'Fail to match {fn}'
        d[match.groups()[0]].append(fn)
    return d


def get_landmarks_filename(matfile: str):
    elements = matfile.split(os.path.sep)
    name = os.path.splitext(elements[-1])[0] + '_pts.mat'
    return os.path.sep.join(elements[:-2] + ['landmarks'] + elements[-2:-1] + [name])


class ReadSample:
    def __init__(self, full_face_bounding_box: bool, load_pt3d_68: bool, load_pt2d_68: bool, load_roi : bool, load_face_params : bool):
        assert not (full_face_bounding_box and load_roi)
        assert load_face_params or load_roi or load_pt3d_68, "Source for BBox"
        self._headmodel = PosedDeformableHead(ScaledBfmModule(BFMModel())) if full_face_bounding_box else None
        self._load_pt3d_68 = load_pt3d_68
        self._load_pt2d_68 = load_pt2d_68
        self._load_roi = load_roi
        self._load_face_params = load_face_params
        self._required_data = [
            'Pose_Para',
        ]
        if load_pt3d_68:
            # AFLW2k-3d has them. 300W-LP doesn't
            self._required_data.append('pt3d_68')
        if load_roi:
            self._required_data.append('roi')
        if load_face_params:
            self._required_data += [
                'Shape_Para',
                'Exp_Para',
            ]

    def __call__(self, zf, matfile):
        with io.BytesIO(zf.read(matfile)) as f:
            data = scipy.io.loadmat(f)
        assert all(
            (k in data) for k in self._required_data
        ), f"Data not found in file {matfile}. Contents is {data.keys()}"

        jpgbuffer = zf.read(splitext(matfile)[0] + '.jpg')
        img = imdecode(jpgbuffer, color=True)

        pitch, yaw, roll, tx, ty, tz, scale = data['Pose_Para'][0]
        rot = aflw_rotation_conversion(pitch, yaw, roll)

        h, w, _ = img.shape
        ty = h - ty
        human_head_radius_micron = 100.0e3
        proj_radius = 0.5 * scale / 224.0 * w * human_head_radius_micron
        coord = [tx, ty, proj_radius]

        coord = move_aflw_head_center_to_between_eyes(coord, rot)
        tx, ty, proj_radius = coord

        if self._load_face_params:
            f_shp, f_exp = get_3ddfa_shape_parameters(data)
            shapeparam = np.concatenate([f_shp, f_exp])
        else:
            shapeparam = None
            f_shp, f_exp = None, None

        if self._load_pt3d_68:
            pt3d = depth_centered_keypoints(data['pt3d_68'])
            pt3d[2] *= -1
        elif self._load_face_params:
            # Note: Landmarks in the landmarks folder of 300wlp omit the z-coordinate.
            #       So we have to reconstruction it using the model parameters.
            #       The reconstruction doesn't match the provided landmarks exactly.
            #       For consistency, the reconstruction is used.
            # Shape is (3,68)
            pt3d = compute_keypoints(f_shp, f_exp, proj_radius, rot, tx, ty)
            assert pt3d.shape == (3, 68), f"Bad shape: {pt3d.shape}"
            pt3d = depth_centered_keypoints(pt3d)
        else:
            pt3d = None

        if self._load_roi:
            x0, y0, x1, y1 = data['roi'][0]
            y0 = h - y0
            y1 = h - y1
        elif self._headmodel is None:
            # The matlab file contains a bounding box which is however way too big for the image size.
            x0, y0, _ = np.amin(pt3d, axis=1)
            x1, y1, _ = np.amax(pt3d, axis=1)
        else:
            assert shapeparam is not None
            vertices = self._headmodel(
                torch.from_numpy(coord), QuatRepr(value=torch.from_numpy(rot.as_quat())), torch.from_numpy(shapeparam)
            ).numpy()
            x0, y0, _ = np.amin(vertices, axis=0)
            x1, y1, _ = np.amax(vertices, axis=0)

        roi = np.array([x0, y0, x1, y1])

        if shapeparam is not None and pt3d is not None:
            # If landmarks are not computed based on parameters
            sanity_check_landmarks(coord, rot, pt3d, (f_shp, f_exp), 0.2, img)

        output = {
            'pose': rot.as_quat(),
            'coord': coord,
            'roi': roi,
            'image': np.frombuffer(jpgbuffer, dtype='B'),
        }

        if pt3d is not None:
            output['pt3d_68'] = np.ascontiguousarray(pt3d.T)
        
        if shapeparam is not None:
            output['shapeparam'] = shapeparam

        if self._load_pt2d_68:
            with io.BytesIO(zf.read(get_landmarks_filename(matfile))) as f:
                landmarkdata = scipy.io.loadmat(f)
            if 0:
                # 2d landmarks are not always applicable since they come from the original image, regardless of the artificial rotation.
                # ... that is, when taken from the mat file in the main folders.
                pt2d = data['pt2d']
            else:
                # When taken from the landmarkfolder they are good
                pt2d = landmarkdata['pts_2d']
            output['pt2d_68'] = np.ascontiguousarray(pt2d)

        return output


class HdfDatasetWriter:
    @abc.abstractmethod
    def get_file_groups(self, zf) -> list[list[str]] | list[str]:
        pass

    @abc.abstractmethod
    def make_sample_reader(self) -> ReadSample:
        pass

    def generate_hdf5_dataset(self, source_file, outfilename, count):
        read_sample = self.make_sample_reader()

        with zipfile.ZipFile(source_file) as zf, h5py.File(outfilename, 'w') as f:
            filename_groups = self.get_file_groups(zf)
            assert filename_groups
            grouped_faces = not isinstance(next(iter(filename_groups)), str)
            if count:
                filename_groups = filename_groups[:count]
            if grouped_faces:
                sequence_starts = np.cumsum([0] + [len(fs) for fs in filename_groups])
                N = sequence_starts[-1]
                f.create_dataset('sequence_starts', data=sequence_starts)
            else:
                N = len(filename_groups)

            if not grouped_faces:
                filename_groups = cast(list[list[str]], [filename_groups])

            sample = read_sample(zf, filename_groups[0][0])
            have_pt2d_68 = 'pt2d_68' in sample
            have_shapeparam = 'shapeparam' in sample

            ds_img = create_pose_dataset(f, C.image, count=N)
            ds_roi = create_pose_dataset(f, C.roi, count=N)
            ds_quats = create_pose_dataset(f, C.quat, count=N)
            ds_coords = create_pose_dataset(f, C.xys, count=N)
            ds_pt3d_68 = create_pose_dataset(f, C.points, name='pt3d_68', count=N, shape_wo_batch_dim=(68, 3))
            if have_pt2d_68:
                ds_pt2d_68 = create_pose_dataset(f, C.points, name='pt2d_68', count=N, shape_wo_batch_dim=(68, 2))
            if have_shapeparam:
                ds_shapeparams = create_pose_dataset(
                    f, C.general, name='shapeparams', count=N, shape_wo_batch_dim=(50,), dtype=np.float16
                )

            i = 0
            with tqdm.tqdm(total=N) as bar:
                for filenames in filename_groups:
                    for fn in filenames:
                        sample = read_sample(zf, fn)
                        ds_img[i] = sample['image']
                        ds_quats[i] = sample['pose']
                        ds_coords[i] = sample['coord']
                        ds_pt3d_68[i] = sample['pt3d_68']
                        if have_pt2d_68:
                            ds_pt2d_68[i] = sample['pt2d_68']  # type: ignore[reportPossiblyUnboundVariable]
                        ds_roi[i] = sample['roi']
                        if have_shapeparam:
                            ds_shapeparams[i] = sample['shapeparam']# type: ignore[reportPossiblyUnboundVariable]
                        i += 1
                        bar.update(1)


class HdfWriter300WLPWithArtificialRotations(HdfDatasetWriter):
    def __init__(self, only_large_poses, full_face_bounding_box):
        super().__init__()
        self.only_large_poses = only_large_poses
        self.full_face_bounding_box = full_face_bounding_box

    def get_file_groups(self, zf):
        filenames = discover_samples(zf)
        if self.only_large_poses:
            filenames = remove_original_faces(filenames)
        return list(make_groups(filenames).values())

    def make_sample_reader(self) -> ReadSample:
        return ReadSample(self.full_face_bounding_box, load_pt3d_68=False, load_pt2d_68=True, load_roi=False, load_face_params=True)


class HdfWriter300WLPWithoutRotations(HdfDatasetWriter):
    def __init__(self, full_face_bounding_box):
        self.full_face_bounding_box = full_face_bounding_box

    def get_file_groups(self, zf):
        filenames = discover_samples(zf)
        filenames = remove_artificially_rotated_faces(filenames)
        return list(make_groups(filenames).values())

    def make_sample_reader(self) -> ReadSample:
        return ReadSample(self.full_face_bounding_box, load_pt3d_68=False, load_pt2d_68=True, load_roi=False, load_face_params=True)


def generate_hdf5_dataset(source_file, outfilename, count, only_large_poses, full_face_bounding_box):
    HdfWriter300WLPWithArtificialRotations(only_large_poses, full_face_bounding_box).generate_hdf5_dataset(
        source_file, outfilename, count
    )


def generate_hdf5_dataset_wo_artificial_rotations(source_file, outfilename, count, full_face_bounding_box):
    HdfWriter300WLPWithoutRotations(full_face_bounding_box).generate_hdf5_dataset(source_file, outfilename, count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert dataset")
    parser.add_argument('source', help="source file", type=str)
    parser.add_argument('destination', help='destination file', type=str, nargs='?', default=None)
    parser.add_argument('-n', dest='count', type=int, default=None)
    parser.add_argument('--subset', choices=['large', 'original', 'both'], default='both')
    parser.add_argument('--reconstruct-head-bbox', default=False, action='store_true')
    args = parser.parse_args()
    dst = args.destination if args.destination else splitext(args.source)[0] + '.h5'
    if args.subset in ('both', 'large'):
        generate_hdf5_dataset(args.source, dst, args.count, args.subset == 'large', args.reconstruct_head_bbox)
    else:
        assert args.subset == 'original'
        generate_hdf5_dataset_wo_artificial_rotations(args.source, dst, args.count, args.reconstruct_head_bbox)
