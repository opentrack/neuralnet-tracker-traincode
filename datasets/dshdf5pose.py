from os.path import join, dirname, basename, splitext
import sys
import numpy as np
from scipy.spatial.transform import Rotation
import h5py

from datasets.preprocessing import imdecode, get_base_seed32
from torch.utils.data import Dataset, IterableDataset, get_worker_info


class Hdf5PoseDataset(IterableDataset):
    def __init__(self, filename, shuffle=True, subset=None, transform=None):
        super(Hdf5PoseDataset, self).__init__()
        self.transform = transform
        self.filename = filename
        self.shuffle = shuffle
        self.fullsize = self._compute_size()
        # Generate indices
        self.subset = np.arange(self.fullsize)
        if subset is not None:
            self.subset = self.subset[subset]
        self.rnd = np.random.RandomState(seed=123)

    def _compute_size(self):
        with h5py.File(self.filename, 'r') as f:
            return f['images'].shape[0]

    def __len__(self):
        return len(self.subset)

    def _seed_worker(self, worker_info):
        if worker_info is not None:
            self.rnd.seed(get_base_seed32(worker_info))

    def _get_indices(self, worker_info):
        if self.shuffle:
            perm = self.rnd.permutation(self.subset)
        else:
            perm = self.subset
        if worker_info is not None:
            # split workload
            perm = perm[worker_info.id::worker_info.num_workers]
        return perm

    def generate_sample(self, data, idx):
        sample = {
            'image' : imdecode(data['images'][idx], color='rgb'),
            'roi' : np.array(data['rois'][idx]),
        }
        if 'coords' in data and 'quats' in data:
            coords = np.array(data['coords'][idx])
            rot = Rotation.from_quat(data['quats'][idx])
            sample.update({
                'pose': rot, 
                'coord' : coords,
            })
        if 'pt3d_68' in data:
            pts = np.array(data['pt3d_68'][idx,...])
            sample.update({
                'pt3d_68' : pts
            })
        if 'hasface' in data:
            sample['hasface'] = bool(data['hasface'][idx])
        return sample
    
    def _generate_sample(self, hf, idx):
        sample = self.generate_sample(hf, idx)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __iter__(self):
        worker_info = get_worker_info()
        if self.shuffle:
            self._seed_worker(worker_info)
        permutation = self._get_indices(worker_info)
        with h5py.File(self.filename, 'r') as f:
            for i in permutation:
                yield self._generate_sample(f, i)