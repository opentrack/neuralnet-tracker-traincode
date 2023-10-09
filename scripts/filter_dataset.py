
import numpy as np
import h5py
import argparse

from trackertraincode.utils import copy_attributes

"""
    Remove sequences or frames.
"""


def _generate_frame_mask(sequence_picks, old_sequence_starts):
    mask = np.zeros((old_sequence_starts[-1],), dtype='?')
    new_sequence_start = np.empty(len(sequence_picks)+1, dtype=np.int64)
    n = 0
    last_end = 0
    for k, i in enumerate(sequence_picks):
        start, end = old_sequence_starts[i], old_sequence_starts[i+1]
        assert end>start 
        assert start >= last_end
        mask[start:end] = True
        new_sequence_start[k] = n
        n += end-start
        last_end = end
    new_sequence_start[-1] = n
    return mask, new_sequence_start


def _invert_indices(indices, total):
    return np.setdiff1d(np.arange(total), np.asarray(indices))


def _prepare_good_indices(total, good_indices, bad_indices):
    assert (good_indices is None) != (bad_indices is None)
    if bad_indices is not None:
        good_indices = _invert_indices(bad_indices, total)
    return np.sort(good_indices)


def filter_file_by_sequences(f, fout, good_sequences_indices = None, bad_sequence_indices = None):
    sequence_starts = np.array(f['sequence_starts'][...])
    good_sequences_indices = _prepare_good_indices(
        total = sequence_starts.shape[0]-1,
        good_indices=good_sequences_indices,
        bad_indices=bad_sequence_indices,
    )
    N = sequence_starts[-1]
    mask, new_sequence_start = _generate_frame_mask(good_sequences_indices, sequence_starts)
    for name, ds in f.items():
        if name == 'sequence_starts':
            fout.create_dataset(name, data = new_sequence_start)
        elif ds.shape[0] == N:
            idx, = np.nonzero(mask)
            new_ds = fout.create_dataset(name, data = ds[idx,...])
            copy_attributes(ds, new_ds)
        else:
            assert False


def filter_file_by_frames(f : h5py.File, fout, good_frame_indices = None, bad_frame_indices = None):
    assert (good_frame_indices is None) != (bad_frame_indices is None)
    assert not 'sequence_starts' in f, "Not supported"
    frame_count = next(iter(f.values())).shape[0]
    indices = _prepare_good_indices(frame_count, good_frame_indices, bad_frame_indices)
    for name, ds in f.items():
        if ds.shape[0] == frame_count:
            new_ds = fout.create_dataset(name, data = ds[indices,...])
            copy_attributes(ds, new_ds)
        else:
            assert False


def main():
    parser = argparse.ArgumentParser(description="Remove sequences")
    parser.add_argument('source', help="source file", type=str)
    parser.add_argument('destination', help='destination file', type=str)
    parser.add_argument('bad',help='indices of bad sequences', type=str)
    args = parser.parse_args()
    bad = [ int(s.strip()) for s in args.bad.split(',') ]
    print (f"Filtering {len(bad)} sequences")
    assert args.source != args.destination
    with h5py.File(args.source, 'r') as f:
        with h5py.File(args.destination, 'w') as fout:
            filter_file_by_sequences(f, fout, bad_sequence_indices=bad)


def test():
    mask, starts = _generate_frame_mask([1,3], [0, 2, 4, 5, 8])
    assert np.all(mask == np.array([False, False,  True,  True, False,  True,  True,  True]))
    assert np.all(starts == np.array([0, 2, 5]))



if __name__ == '__main__':
    test()
    main()