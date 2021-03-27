import sys
import numpy as np
from os.path import join, dirname, basename, splitext
import progressbar
import glob
import numbers
import pandas as pd
import cv2
import io
import h5py
import argparse
import zipfile

import utils
from datasets.preprocessing import imencode, depth_centered_keypoints, \
    extended_key_points_for_bounding_box

def _videoid_from_path(x):
    return x.split('/')[-1].split('.')[0]


def _find_sequence_files(path):
    zipfilenames = glob.glob(join(path,'youtube_faces_with_keypoints_full_?.zip'))
    sequence_filenames = {}
    for fn in zipfilenames:
        with zipfile.ZipFile(fn) as zf:
            sequence_filenames[fn] = [ _videoid_from_path(f.filename) for f in zf.filelist if 
                f.filename.endswith('.npz') ]
    return sequence_filenames


def _keypoints_bounding_boxes(kpts):
    """
        Input dims: 3 x 68 ...
        Output: bounding box over first two space dimensions
    """
    x0, y0 = np.amin(kpts[:2,:], axis=1)
    x1, y1 = np.amax(kpts[:2,:], axis=1)
    return np.array([x0, y0, x1, y1])


class Video(object):
    def __init__(self, id, zf):
        prefix = splitext(basename(zf.filename))[0]
        path = prefix+'/'+id+'.npz'
        with io.BytesIO(zf.read(path)) as f:
            videoFile = np.load(f)
            self.colorImages = videoFile['colorImages']
            self.landmarks3D = videoFile['landmarks3D']
    
    def __len__(self):
        return self.colorImages.shape[-1]
    
    def __getitem__(self, index):
        return self.colorImages[...,index], self.landmarks3D[...,index].T


class YTFacesWithKeypointsRaw(object):
    def __init__(self, path):
        videoDF = pd.read_csv(join(path,'youtube_faces_with_keypoints_full.csv'))
        videoDF['videoDuration'] = videoDF['videoDuration'].astype(np.int32)
        videos_by_archive = _find_sequence_files(path)
        all_videos_from_archives = frozenset(sum(videos_by_archive.values(), []))
        video_ids_from_csv = frozenset(videoDF.loc[:,'videoID'])
        intersection = all_videos_from_archives.intersection(video_ids_from_csv)
        videoDF = videoDF.loc[videoDF.loc[:,'videoID'].isin(intersection),:].reset_index(drop=True)
        video_id_to_zipname = {}
        for k, videos in videos_by_archive.items():
            # Only use valid sequences that are in the csv
            videos = frozenset(videos).intersection(intersection)
            video_id_to_zipname.update({ v:k for v in videos })
        self.videoDF = videoDF
        self.video_id_to_zipname = video_id_to_zipname
        # FIXME: context manager?
        self.zipfiles = {
            k:zipfile.ZipFile(k, 'r') for k in videos_by_archive.keys() }

    def keys(self):
        return self.videoDF.loc[:,'videoID']
    
    def metadata(self, name):
        return self.videoDF.loc[:,name]
    
    def __getitem__(self, key):
        if isinstance(key, numbers.Integral):
            id = self.videoDF.loc[int(key),'videoID']
            return Video(id, self.zipfiles[self.video_id_to_zipname[id]])
        else:
            return Video(key, self.zipfiles[self.video_id_to_zipname[key]])
    
    def __iter__(self):
        for key in self.keys():
            yield self[key]
    
    def __len__(self):
        return len(self.video_id_to_zipname)


def generate_hdf5_dataset(source_dir, outfilename, count=None):
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    ytfaces = YTFacesWithKeypointsRaw(source_dir)
    keys = ytfaces.keys()
    keys = sorted(keys)
    keys = np.array(keys)
    np.random.RandomState(seed=123).shuffle(keys)
    if count is not None:
        keys = keys[:count]
    with h5py.File(outfilename, 'w') as f:
        with progressbar.ProgressBar() as bar:
            video_lengths = [ len(ytfaces[k]) for k in bar(keys) ]
        video_offsets = np.cumsum([0]+video_lengths)
        N = video_offsets[-1]
        cs = min(N, 1024)
        ds_img = f.create_dataset('images', (N,), chunks=(cs,), maxshape=(N,), dtype=dt)
        ds_pt3d_68 = f.create_dataset('pt3d_68', (N,3,68), chunks=(cs,3,68), maxshape=(N,3,68), dtype='f4')
        ds_file = f.create_dataset('files', (N,), chunks=(cs,), maxshape=(N,), dtype='S40')
        ds_roi = f.create_dataset('rois', (N,4), chunks=(cs,4), maxshape=(N,4), dtype='f4')
        i = 0
        with progressbar.ProgressBar() as bar:
            for k in bar(keys):
                video = ytfaces[k]
                for img, pt3d_68 in video:
                    assert (pt3d_68.shape == (3,68)), f"Bad shape: {pt3d_68.shape}"
                    ds_img[i] = imencode(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    ds_file[i] = ('ytf/{}/{}'.format(k,i)).encode('ascii')
                    ds_pt3d_68[i] = depth_centered_keypoints(pt3d_68)
                    ds_roi[i] = _keypoints_bounding_boxes(extended_key_points_for_bounding_box(pt3d_68.T).T)
                    i += 1
        f.create_dataset('sequence_starts', data = video_offsets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert dataset")
    parser.add_argument('source', help="source file", type=str)
    parser.add_argument('destination', help='destination file', type=str, nargs='?', default=None)
    parser.add_argument('-n', dest = 'count', type=int, default=None)
    args = parser.parse_args()
    dst = args.destination if args.destination else \
        args.source+'.h5'
    generate_hdf5_dataset(args.source, dst, args.count)