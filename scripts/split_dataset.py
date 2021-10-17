
from matplotlib import pyplot
import numpy as np
import h5py
import argparse
import os
import sys
from os.path import join, splitext

"""
Split dataset in test and train sets
"""

def generate_sequence_start_len(sequence_starts):
    for i, start in enumerate(sequence_starts[:-1]):
        yield start, sequence_starts[i+1]-start


def create_new_sequence_starts(sequences, num_samples):
    """
        Input: Sequence of (start,length) tuples
               Total number of samples in the dataset
        Output: Mask indicating the samples to be taken.
                New sequence starts.
    """
    mask = np.zeros(num_samples, dtype='?')
    new_sequence_start = np.empty(len(sequences)+1, dtype=np.long)
    end = 0
    for (i, (start, l)) in enumerate(sorted(sequences, key=lambda x: x[0])):
        mask[start:start+l] = True
        new_sequence_start[i] = end
        end += l
    new_sequence_start[-1] = end
    return mask, new_sequence_start


def make_split_indices(n, num_test, rng):
    if rng is not None:
        idx_test = rng.choice(n, size=num_test, replace=False)
        idx_train = np.setdiff1d(n, idx_test)
    else:
        idx_test = np.arange(num_test)
        idx_train = np.arange(num_test, n)
    return idx_test, idx_train


def generate_splits(f, ftest, ftrain, num_test, rng):
    if 'sequence_starts' in f:
        # Select entire sequences
        sequence_starts = np.array(f['sequence_starts'])
        num_total = len(sequence_starts)
        num_train = num_total - num_test
        print (f"Splitting sequence based: Total {num_total}, Test {num_test}, Train {num_train}")
        assert num_train>0
        assert num_test>0
        sequences = np.array([*generate_sequence_start_len(sequence_starts)])
        idx_test, idx_train = make_split_indices(len(sequences), num_test, rng)
        test_sequences = sequences[idx_test,:]
        train_sequences = sequences[idx_train,:]
        num_samples = sequence_starts[-1]
        print (f"Num samples {num_samples}")
        for fout, new_sequences in zip([ftest,ftrain], [test_sequences, train_sequences]):
            mask, new_sequence_starts = create_new_sequence_starts(new_sequences, num_samples)
            for name, ds in f.items():
                print (f"{name}: {ds.shape}, {ds.dtype}")
                if name == 'sequence_starts':
                    fout.create_dataset(name, data = new_sequence_starts)
                elif ds.shape[0] == num_samples:
                    fout.create_dataset(name, data = ds[mask,...])
                else:
                    print (name, ds)
                    assert False
    else:
        num_total = len(f['images'])
        num_train = num_total - num_test
        print (f"Splitting sample based: Total {num_total}, Test {num_test}, Train {num_train}")
        assert num_train>0
        assert num_test>0
        idx_test, idx_train = make_split_indices(num_total, num_test, rng)
        for fout, idx in zip([ftest,ftrain], [idx_test, idx_train]):
            for name, ds in f.items():
                print (f"{name}: {ds.shape}, {ds.dtype}")
                assert ds.shape[0] == num_total
                fout.create_dataset(name, data = ds[idx,...])


def main():
    parser = argparse.ArgumentParser(description="Generate test and train splits")
    parser.add_argument('source', help="source file", type=str)
    parser.add_argument('test_size', help="number of samples in the test split", type=int)
    parser.add_argument('--seed', help="rng seed", type=int, default=1234)
    parser.add_argument('--no-randomization', help="Take test set first, the rest is train. No random selection.", action='store_false', default=True, dest='randomization')
    args = parser.parse_args()
    args.test_filename = splitext(args.source)[0]+'_test.h5'
    args.train_filename = splitext(args.source)[0]+'_train.h5'
    rng = np.random.RandomState(args.seed) if args.randomization else None
    with h5py.File(args.source, 'r') as f, \
         h5py.File(args.test_filename, 'w') as ftest, \
         h5py.File(args.train_filename, 'w') as ftrain:
            generate_splits(f, ftest, ftrain, args.test_size, rng)


if __name__ == '__main__':
    main()