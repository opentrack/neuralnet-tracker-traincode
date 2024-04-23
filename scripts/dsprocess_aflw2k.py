import argparse
import io
import h5py
import numpy as np
from os.path import join, dirname, basename, splitext, sep
from scipy.spatial.transform import Rotation
import scipy.io
import cv2
import tqdm
import zipfile

from trackertraincode.datasets.preprocessing import imdecode, depth_centered_keypoints, \
    move_aflw_head_center_to_between_eyes, \
    get_3ddfa_shape_parameters, sanity_check_landmarks, load_shape_components
from trackertraincode.datasets.dshdf5pose import create_pose_dataset, FieldCategory
from trackertraincode.utils import aflw_rotation_conversion

C = FieldCategory


def is_sample_file(fn):
    return splitext(fn)[1]=='.mat' and not fn.endswith(sep) \
        and dirname(fn)=='AFLW2000'

def discover_samples(zf):
    filenames = [ 
        f.filename for f in zf.filelist if is_sample_file(f.filename)  ]
    return filenames


def read_data(zf, matfile):
    with io.BytesIO(zf.read(matfile)) as f:
        data = scipy.io.loadmat(f)

    assert all((k in data) for k in 'pt3d_68 Pose_Para'.split()), f"Data not found in file {matfile}. Contents is {data.keys()}"

    assert (data['pt3d_68'].shape == (3,68)), f"Bad shape: {data['pt3d_68'].shape}"

    pitch, yaw, roll, tx, ty, tz, scale = data['Pose_Para'][0]
    rot = aflw_rotation_conversion(pitch, yaw,roll)
    
    jpgbuffer = zf.read(splitext(matfile)[0]+'.jpg')
    img = imdecode(jpgbuffer, 'rgb')

    h, w, _ = img.shape
    ty = h - ty
    human_head_radius_micron = 100.e3
    proj_radius = 0.5*scale / 224. * w * human_head_radius_micron
    coord = [ tx, ty, proj_radius ]

    coord = move_aflw_head_center_to_between_eyes(coord, rot)

    pt3d = depth_centered_keypoints(data['pt3d_68'])
    pt3d[2] *= -1

    # There is data['roi'], but the coordinates seem to be outside of the face region. Very odd.
    x0, y0, _ = np.amin(pt3d, axis=1)
    x1, y1, _ = np.amax(pt3d, axis=1)
    roi = np.array([x0, y0, x1, y1])

    f_shp, f_exp = get_3ddfa_shape_parameters(data)

    sanity_check_landmarks(coord, rot, pt3d, (f_shp, f_exp), 0.2, img)

    return { 
        'pose' :  rot.as_quat(),
        'coord' : coord,
        'roi' : roi, 
        'image' : np.frombuffer(jpgbuffer, dtype='B'),
        'pt3d_68' : pt3d.T,
        'shapeparam' : np.concatenate([f_shp, f_exp])
    }


def generate_hdf5_dataset(source_file, outfilename, count=None):
    with zipfile.ZipFile(source_file) as zf:
        filenames = sorted(discover_samples(zf))
        np.random.RandomState(seed=123).shuffle(filenames)
        if count:
            filenames = filenames[:count]
        N = len(filenames)
        with h5py.File(outfilename, 'w') as f:
            ds_img = create_pose_dataset(f, C.image, count=N)
            ds_roi = create_pose_dataset(f, C.roi, count=N)
            ds_quats = create_pose_dataset(f, C.quat, count=N)
            ds_coords = create_pose_dataset(f, C.xys, count=N)
            ds_pt3d_68 = create_pose_dataset(f, C.points, name='pt3d_68', count=N, shape_wo_batch_dim=(68,3))
            ds_shapeparams = create_pose_dataset(f, C.general, name='shapeparams', count=N, shape_wo_batch_dim=(50,), dtype=np.float16)
            i = 0
            for matfile in tqdm.tqdm(filenames):
                sample = read_data(zf, matfile)
                ds_img[i] = sample['image']
                ds_quats[i] = sample['pose']
                ds_coords[i] = sample['coord']
                ds_pt3d_68[i] = sample['pt3d_68']
                ds_roi[i] = sample['roi']
                ds_shapeparams[i] = sample['shapeparam']
                i += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert dataset")
    parser.add_argument('source', help="source file", type=str)
    parser.add_argument('destination', help='destination file', type=str, nargs='?', default=None)
    parser.add_argument('-n', dest = 'count', type=int, default=None)
    args = parser.parse_args()
    dst = args.destination if args.destination else \
        splitext(args.source)[0]+'.h5'
    generate_hdf5_dataset(args.source, dst, args.count)