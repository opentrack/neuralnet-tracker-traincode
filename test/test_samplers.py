import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SequentialSampler, BatchSampler
import torch

from trackertraincode.datasets.randomized import make_concat_dataset_item_sampler, ConcatDatasetSampler

NUM_WORKERS = 2

class _TestSet(Dataset):
    def __init__(self, n, start=0):
        super(_TestSet,self).__init__()
        self.n = n
        self.start = start
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        if i<0 or i >= self.n:
            raise IndexError()
        return torch.as_tensor(i+self.start, dtype=torch.int)


def load_dataset(ds, batchsize = 1, concat=True, repeats=None, sampler=None, batch_sampler=None):
    # if batchsize is None:
    #     batchsize = len(ds)//NUM_WORKERS # Seems to be the maximum
    train_loader = DataLoader(
        ds,
        batch_size=batchsize, 
        num_workers=NUM_WORKERS,
        sampler=sampler,
        batch_sampler=batch_sampler)
    real_repeats = repeats if repeats else 1
    ret = []
    for _ in range(real_repeats):
        y = [ y for y in train_loader ]
        if concat:
            y = torch.cat(y, dim=0)
        ret.append(y)
    return ret if repeats else ret[0]


def test_test_set1():
    N = 10
    ds = _TestSet(N)
    y = [ y for y in ds ]
    assert np.all(y == np.arange(N))


def test_interleaved_datasets2():
    N1 = 10
    N2 = 10
    M = 10
    BS = 10
    ds = ConcatDataset([
        _TestSet(N1),
        _TestSet(N2, start=N1)
    ])
    dssamplers = [
        SequentialSampler(ds.datasets[0]),
        SequentialSampler(ds.datasets[1])
    ]
    sampler = make_concat_dataset_item_sampler(
        ds,
        wrapped=dssamplers,
        weights=[0.5, 0.5],
        stop_after=M*BS)
    y = load_dataset(
        ds,
        sampler = sampler,
        batchsize=BS
        ).numpy()
    NY = M*BS
    assert len(y) == NY
    h, b = np.histogram(y, bins=np.linspace(-0.5, N1+N2-0.5, N1+N2+1))
    assert np.all(np.isclose(h, M//2, atol=1))


if __name__ == '__main__':
    test_test_set1()
    test_interleaved_datasets2()