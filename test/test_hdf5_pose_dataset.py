from os.path import join, dirname

from torch.utils.data import Dataset, DataLoader

from datasets.dshdf5pose import Hdf5PoseDataset


def nice_and_hashable(sample):
    return sample['coord']
    
    
def generate_samples(shuffle, subset, num_workers, batch_size):
    ds = Hdf5PoseDataset(
        join(dirname(__file__),'..','aflw2kmini.h5'),
        shuffle=shuffle,
        subset = subset,
        transform = nice_and_hashable # Because it cannot collate scipy rotations
    )
    train_loader = DataLoader(ds,
                            batch_size=batch_size,
                            shuffle=False, 
                            num_workers=num_workers)
    return [ tuple(item.numpy()) for batch in train_loader for item in batch ]


if __name__ == '__main__':
    items = generate_samples(False, None, 3, 2)
    assert len(items) == len(set(items)) # All unique
    print ("----- shuffled -------")
    shuffled = generate_samples(True, None, 3, 2)
    assert set(items) == set(shuffled) # Just reordered
    assert items != shuffled # But order should be different from unshuffled
    print ("----- subset -------")
    subset = generate_samples(False, slice(5), 3, 2)
    assert set(subset) == set(items[:5])
    print ("----- subset shuffled -------")
    shuffled = generate_samples(True, slice(5), 3, 2)
    assert set(subset) == set(shuffled)
    assert subset != shuffled