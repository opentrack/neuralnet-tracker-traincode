from os.path import join, dirname, basename, splitext
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import h5py

from datasets.preprocessing import imdecode, get_base_seed32, labels_to_lists
from torch.utils.data import Dataset, Subset
import torch

# To acertain exceptions are raised in worker processes, too
np.seterr(all='raise')

class Hdf5PoseDataset(Dataset):
    def __init__(self, filename, transform=None):
        super(Hdf5PoseDataset, self).__init__()
        self.transform = transform
        self.filename = filename
        self.fullsize = self._compute_size()
        self.h5file = None

    def _compute_size(self):
        with h5py.File(self.filename, 'r') as f:
            return f['images'].shape[0]

    def __len__(self):
        return self.fullsize

    def _grab_image(self, data, idx):
        data = data['images'][idx]
        if data.dtype == np.uint8 and data.ndim in (2,3):
            data = np.array(data)
            if data.ndim==2:
                data = np.stack([data,data,data], axis=-1)
            return data
        else:
            # JPEG encoded buffer must be decoded
            return imdecode(data, color='rgb')

    def generate_sample(self, data, idx):
        try:
            sample = {
                'image' : self._grab_image(data, idx),
                'roi' : np.array(data['rois'][idx]).astype(np.float32),
            }
            if 'coords' in data and 'quats' in data:
                coords = np.array(data['coords'][idx]).astype(np.float32)
                # coords = pos + [ scale ]
                rot = Rotation.from_quat(data['quats'][idx])
                sample.update({
                    'pose': rot, 
                    'coord' : coords,
                })
            if 'pt3d_68' in data:
                pts = np.array(data['pt3d_68'][idx,...]).astype(np.float32)
                sample.update({
                    'pt3d_68' : pts
                })
            if 'hasface' in data:
                sample['hasface'] = bool(data['hasface'][idx])
            if 'shapeparams' in data:
                sample['shapeparam'] = np.array(data['shapeparams'][idx]).astype(np.float32)
            if 'head_rois' in data:
                sample['roi_head'] = np.array(data['head_rois'][idx]).astype(np.float32)
            return sample
        except h5py.OSError as e:
            print (f"Error reading data at index {idx}!")
            raise

    def __getitem__(self, index):
        if index >= self.fullsize:
            raise IndexError
        if self.h5file is None:
            self.h5file = h5py.File(self.filename, 'r')
        sample = self.generate_sample(self.h5file, index)
        if self.transform:
            sample = self.transform(sample)
        return sample


def generate_uniform_cluster_probabilities(filename):
    with h5py.File(filename, 'r') as f:
        labels = np.array(f['cluster-labels'][...]).astype(np.int64)
    cluster_indices, counts = np.unique(labels, return_counts=True)
    probabilities = np.empty((len(counts),), dtype=np.float64)
    probabilities[...] = np.nan
    probabilities[cluster_indices] = np.reciprocal(counts.astype(np.float64))/len(counts)
    assert np.all(np.isfinite(probabilities))
    probabilities = probabilities[labels]
    probabilities = torch.DoubleTensor(probabilities)
    return probabilities