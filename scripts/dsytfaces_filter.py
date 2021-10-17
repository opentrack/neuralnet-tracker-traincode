
from matplotlib import pyplot
import numpy as np
import h5py
import argparse
import os
import sys
from os.path import join, splitext

"""
    Remove sequences with garbage labels.
    We can tell if garbage based on the fit loss from fitting the
    deformable head model. And I also use the size of the face boxes.
"""


def debug_plot(fitlosses, diameters, shapeparam):
    fig, ax = pyplot.subplots(1,2)
    x = fitlosses
    y = np.amax(np.abs(shapeparam), axis=1)
    ax[0].scatter(x,y)
    ax[0].set(xlabel = 'fitloss', ylabel='shapeparam')
    x = diameters
    y = fitlosses
    ax[1].scatter(x,y)
    ax[1].set(xlabel = 'face size', ylabel='fitloss')
    pyplot.show()


def sequence_filtering_data(sequence_nums, sequence_starts):
    mask = np.zeros((sequence_starts[-1],), dtype='?')
    new_sequence_start = np.empty(len(sequence_nums)+1, dtype=np.long)
    n = 0
    last_end = 0
    for k, i in enumerate(sequence_nums):
        start, end = sequence_starts[i], sequence_starts[i+1]
        assert end>start 
        assert start >= last_end
        mask[start:end] = True
        new_sequence_start[k] = n
        n += end-start
        last_end = end
    new_sequence_start[-1] = n
    return mask, new_sequence_start


def fill_filtered_file(f, fout):
    sequence_starts = np.array(f['sequence_starts'])
    fitlosses = np.array(f['fitloss'])
    N = sequence_starts[-1]
    rois = np.array(f['rois'])
    diameters = np.sqrt((rois[:,2]-rois[:,0])*(rois[:,3]-rois[:,1]))

    debug_plot(fitlosses, diameters, np.array(f['shapeparams']))

    # Build map from frames to their sequence number
    sequence_indices = np.empty((N,), dtype=np.long)
    for k, (i,j) in enumerate(zip(sequence_starts[:-1],sequence_starts[1:])):
        sequence_indices[i:j] = k

    idx_ok, = np.nonzero((diameters>120) & (fitlosses<0.05))
    seq_picks = np.sort(np.unique(sequence_indices[idx_ok]))

    mask, new_sequence_start = sequence_filtering_data(seq_picks, sequence_starts)
    for name, ds in f.items():
        if name == 'sequence_starts':
            fout.create_dataset(name, data = new_sequence_start)
        elif ds.shape[0] == N:
            fout.create_dataset(name, data = ds[mask,...])
        else:
            assert False


def main():
    parser = argparse.ArgumentParser(description="Remove sequences with bad labels")
    parser.add_argument('source', help="source file", type=str)
    parser.add_argument('destination', help='destination file', type=str, nargs='?', default=None)
    args = parser.parse_args()
    if args.destination is None:
        args.destination = splitext(args.source)[0]+'_filtered.h5'
    assert args.source != args.destination
    with h5py.File(args.source, 'r') as f:
        with h5py.File(args.destination, 'w') as fout:
            fill_filtered_file(f, fout)


def test():
    mask, starts = sequence_filtering_data([1,3], [0, 2, 4, 5, 8])
    assert np.all(mask == np.array([False, False,  True,  True, False,  True,  True,  True]))
    assert np.all(starts == np.array([0, 2, 5]))



if __name__ == '__main__':
    test()
    main()