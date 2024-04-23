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
from typing import List
import re
import tqdm

from trackertraincode.datasets.preprocessing import imdecode, compute_keypoints, depth_centered_keypoints, \
    move_aflw_head_center_to_between_eyes, sanity_check_landmarks, \
    get_3ddfa_shape_parameters, load_shape_components, head_bbox_from_keypoints
from trackertraincode.datasets.dshdf5pose import create_pose_dataset, FieldCategory
from trackertraincode.utils import aflw_rotation_conversion

C = FieldCategory

def discover_samples(zf):
    names = frozenset(['AFW', 'HELEN', 'IBUG', 'LFPW'])
    isInDataSubsets = lambda s: s.split(os.path.sep)[1] in names
    filenames = [
        f.filename for f in zf.filelist if 
        (f.external_attr==0x20 and splitext(f.filename)[1]=='.mat' and isInDataSubsets(f.filename)) ]
    return filenames


def remove_artificially_rotated_faces(filenames : List[str]):
    return list(filter(lambda fn: fn.endswith('_0.mat'), filenames))


def remove_original_faces(filenames : List[str]):
    return list(filter(lambda fn: not fn.endswith('_0.mat'), filenames))


def make_groups(filenames : List[str]):
    regex = re.compile('([\w| ]+)_(\d+).mat')
    d = collections.defaultdict(list)
    for fn in filenames:
        match = regex.match(os.path.basename(fn))
        assert match is not None, f'Fail to match {fn}'
        d[match.groups()[0]].append(fn)
    return d


def get_landmarks_filename(matfile : str):
    elements = matfile.split(os.path.sep)
    name = os.path.splitext(elements[-1])[0]+'_pts.mat'
    return os.path.sep.join(elements[:-2]+['landmarks']+elements[-2:-1]+[name])


def read_sample(zf, matfile):
    with io.BytesIO(zf.read(matfile)) as f:
        data = scipy.io.loadmat(f)

    jpgbuffer = zf.read(splitext(matfile)[0]+'.jpg')
    img = imdecode(jpgbuffer, cv2.IMREAD_COLOR)
    
    with io.BytesIO(zf.read(get_landmarks_filename(matfile))) as f:
        landmarkdata = scipy.io.loadmat(f)

    pitch, yaw, roll, tx, ty, tz, scale = data['Pose_Para'][0]
    rot = aflw_rotation_conversion(pitch, yaw,roll)

    h, w, _ = img.shape
    ty = h - ty
    human_head_radius_micron = 100.e3
    proj_radius = 0.5*scale / 224. * w * human_head_radius_micron
    coord = [ tx, ty, proj_radius ]

    coord = move_aflw_head_center_to_between_eyes(coord, rot)
    tx, ty, proj_radius = coord

    f_shp, f_exp = get_3ddfa_shape_parameters(data)

    # Note: Landmarks in the landmarks folder of 300wlp omit the z-coordinate.
    #       So we have to reconstruction it using the model parameters.
    #       The reconstruction doesn't match the provided landmarks exactly.
    #       For consistency, the reconstruction is used.
    # Shape is (3,68)
    pt3d = compute_keypoints(f_shp, f_exp, proj_radius, rot, tx, ty)
    assert (pt3d.shape == (3,68)), f"Bad shape: {pt3d.shape}"
    pt3d = depth_centered_keypoints(pt3d)

    if 0:
        # The matlab file contains a bounding box which is however way too big for the image size.
        x0, y0, _ = np.amin(pt3d, axis=1)
        x1, y1, _ = np.amax(pt3d, axis=1)
        roi = np.array([x0, y0, x1, y1])
    else:
        roi = head_bbox_from_keypoints(np.ascontiguousarray(pt3d.T))

    sanity_check_landmarks(coord, rot, pt3d, (f_shp, f_exp), 0.2, img)

    if 0:
        # Note: 2d landmarks are not always applicable since they come from the original image, regardless of the artificial rotation.
        # ... that is, when taken from the mat file in the main folders.
        pt2d = data['pt2d']
    else:
        # When taken from the landmarkfolder they are good
        pt2d = landmarkdata['pts_2d']

    return { 
        'pose' :  rot.as_quat(),
        'coord' : coord,
        'roi' : roi, 
        'image' : np.frombuffer(jpgbuffer, dtype='B'),
        'pt3d_68' : np.ascontiguousarray(pt3d.T),
        'pt2d_68' : np.ascontiguousarray(pt2d),
        'shapeparam' : np.concatenate([f_shp, f_exp])
    }


def generate_hdf5_dataset(source_file, outfilename, count, only_large_poses):
    with zipfile.ZipFile(source_file) as zf:
        filenames = discover_samples(zf)
        if only_large_poses:
            filenames = remove_original_faces(filenames)
        filename_groups = [*make_groups(filenames).items() ]
        np.random.RandomState(seed=123).shuffle(sorted(filename_groups))
        if count:
            filename_groups = filename_groups[:count]
        sequence_starts = np.cumsum([0]+[len(fs) for _,fs in filename_groups])
        N = sequence_starts[-1]
        with h5py.File(outfilename, 'w') as f:
            ds_img = create_pose_dataset(f, C.image, count=N)
            ds_roi = create_pose_dataset(f, C.roi, count=N)
            ds_quats = create_pose_dataset(f, C.quat, count=N)
            ds_coords = create_pose_dataset(f, C.xys, count=N)
            ds_pt3d_68 = create_pose_dataset(f, C.points, name='pt3d_68', count=N, shape_wo_batch_dim=(68,3))
            ds_pt2d_68 = create_pose_dataset(f, C.points, name='pt2d_68', count=N, shape_wo_batch_dim=(68,2))
            ds_shapeparams = create_pose_dataset(f, C.general, name='shapeparams', count=N, shape_wo_batch_dim=(50,), dtype=np.float16)
            f.create_dataset('sequence_starts', data = sequence_starts)
            i = 0
            with tqdm.tqdm(total=N) as bar:
                for _, filenames in filename_groups:
                    for fn in filenames:
                        sample = read_sample(zf, fn)
                        ds_img[i] = sample['image']
                        ds_quats[i] = sample['pose']
                        ds_coords[i] = sample['coord']
                        ds_pt3d_68[i] = sample['pt3d_68']
                        ds_pt2d_68[i] = sample['pt2d_68']
                        ds_roi[i] = sample['roi']
                        ds_shapeparams[i] = sample['shapeparam']
                        i += 1
                        bar.update(1)


def generate_hdf5_dataset_wo_artificial_rotations(source_file, outfilename, count=None):
    with zipfile.ZipFile(source_file) as zf:
        filenames = discover_samples(zf)
        filenames = remove_artificially_rotated_faces(filenames)
        np.random.RandomState(seed=123).shuffle(sorted(filenames))
        if count:
            filenames = filenames[:count]
        N = len(filenames)
        with h5py.File(outfilename, 'w') as f:
            ds_img = create_pose_dataset(f, C.image, count=N)
            ds_roi = create_pose_dataset(f, C.roi, count=N)
            ds_quats = create_pose_dataset(f, C.quat, count=N)
            ds_coords = create_pose_dataset(f, C.xys, count=N)
            ds_pt3d_68 = create_pose_dataset(f, C.points, name='pt3d_68', count=N, shape_wo_batch_dim=(68,3))
            ds_pt2d_68 = create_pose_dataset(f, C.points, name='pt2d_68', count=N, shape_wo_batch_dim=(68,2))
            ds_shapeparams = create_pose_dataset(f, C.general, name='shapeparams', count=N, shape_wo_batch_dim=(50,), dtype=np.float16)
            i = 0
            for fn in tqdm.tqdm(filenames):
                sample = read_sample(zf, fn)
                ds_img[i] = sample['image']
                ds_quats[i] = sample['pose']
                ds_coords[i] = sample['coord']
                ds_pt3d_68[i] = sample['pt3d_68']
                ds_pt2d_68[i] = sample['pt2d_68']
                ds_roi[i] = sample['roi']
                ds_shapeparams[i] = sample['shapeparam']
                i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert dataset")
    parser.add_argument('source', help="source file", type=str)
    parser.add_argument('destination', help='destination file', type=str, nargs='?', default=None)
    parser.add_argument('-n', dest = 'count', type=int, default=None)
    parser.add_argument('--pose-set', choices=['large','original','both'], default='both')
    args = parser.parse_args()
    dst = args.destination if args.destination else \
        splitext(args.source)[0]+'.h5'
    if args.pose_set in ('both','large'):
        generate_hdf5_dataset(args.source, dst, args.count, args.pose_set=='large')
    else:
        assert args.pose_set == 'original'
        generate_hdf5_dataset_wo_artificial_rotations(args.source, dst, args.count)