import argparse
import io
import h5py
import numpy as np
from os.path import join, dirname, basename, splitext, sep
from scipy.spatial.transform import Rotation
import scipy.io
import cv2
import progressbar
import zipfile

from datasets.preprocessing import imdecode, rotation_conversion_from_hell,\
    compute_keypoints_from_3ddfa_shape_params, depth_centered_keypoints, \
    move_aflw_head_center_to_between_eyes, extended_key_points_for_bounding_box, \
    get_3ddfa_shape_parameters


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
    rot = rotation_conversion_from_hell(pitch, yaw,roll)
    
    jpgbuffer = zf.read(splitext(matfile)[0]+'.jpg')
    img = imdecode(jpgbuffer, cv2.IMREAD_COLOR)

    h, w, _ = img.shape
    ty = h - ty
    human_head_radius_micron = 100.e3
    proj_radius = 0.5*scale / 224. * w * human_head_radius_micron
    coord = [ tx, ty, proj_radius ]

    coord = move_aflw_head_center_to_between_eyes(coord, rot)

    pt3d = depth_centered_keypoints(data['pt3d_68'])

    extpts = extended_key_points_for_bounding_box(pt3d.T).T
    x0, y0, _ = np.amin(extpts, axis=1)
    x1, y1, _ = np.amax(extpts, axis=1)
    roi = np.array([x0, y0, x1, y1])

    f_shp, f_exp = get_3ddfa_shape_parameters(data)

    return { 
        'pose' :  rot.as_quat(),
        'coord' : coord,
        'roi' : roi, 
        'image' : np.frombuffer(jpgbuffer, dtype='B'),
        'pt3d_68' : pt3d,
        'shapeparam' : np.concatenate([f_shp, f_exp])
    }


def generate_hdf5_dataset(source_file, outfilename, count=None):
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    with zipfile.ZipFile(source_file) as zf:
        filenames = sorted(discover_samples(zf))
        np.random.RandomState(seed=123).shuffle(filenames)
        if count:
            filenames = filenames[:count]
        N = len(filenames)
        cs = min(N, 1024)
        with h5py.File(outfilename, 'w') as f:
            ds_img = f.create_dataset('images', (N,), chunks=(cs,), maxshape=(N,), dtype=dt)
            ds_quats = f.create_dataset('quats', (N,4), chunks=(cs,4), maxshape=(N,4), dtype='f4')
            ds_coords = f.create_dataset('coords', (N,3), chunks=(cs,3), maxshape=(N,3), dtype='f4')
            ds_pt3d_68 = f.create_dataset('pt3d_68', (N,3,68), chunks=(cs,3,68), maxshape=(N,3,68), dtype='f4')
            ds_roi = f.create_dataset('rois', (N,4), chunks=(cs,4), maxshape=(N,4), dtype='f4')
            ds_shapeparams = f.create_dataset('shapeparams', (N,50), chunks=(cs,50), maxshape=(N,50), dtype='f4')
            i = 0
            with progressbar.ProgressBar() as bar:
                for matfile in bar(filenames):
                    sample = read_data(zf, matfile)
                    ds_img[i] = sample['image']
                    ds_quats[i] = sample['pose']
                    ds_coords[i] = sample['coord']
                    ds_pt3d_68[i] = sample['pt3d_68']
                    ds_roi[i] = sample['roi']
                    ds_shapeparams[i] = sample['shapeparam']
                    i += 1
            #print("")
            #print(f"Shape Std: {repr(ds_shapeparams[...].std(axis=0))}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert dataset")
    parser.add_argument('source', help="source file", type=str)
    parser.add_argument('destination', help='destination file', type=str, nargs='?', default=None)
    parser.add_argument('-n', dest = 'count', type=int, default=None)
    args = parser.parse_args()
    dst = args.destination if args.destination else \
        splitext(args.source)[0]+'.h5'
    generate_hdf5_dataset(args.source, dst, args.count)