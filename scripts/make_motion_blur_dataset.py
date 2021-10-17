import numpy as np
import h5py
import argparse
import progressbar
import os
import sys
from os.path import join, splitext

import datasets.preprocessing

"""
Averages frames from sequences to produce a new dataset with a motion blur effect.
"""

blur_lengths = [ 29, 65 ]

def make_it(f, fout, rnd : np.random.RandomState):
    ds_images = f['images']
    N = ds_images.shape[0]
    sequence_starts = np.array(f['sequence_starts'])
    new_sequence_starts = [ 0 ]
    blur_length_picks = []
    indices_to_transfer = []
    for start, end in zip(sequence_starts[:-1], sequence_starts[1:]):
        length = end-start
        if length < min(blur_lengths):
            continue
        admissible_blur_lengths = [ l for l in blur_lengths if l <= length ]
        blur_length, = rnd.choice(admissible_blur_lengths, size=1)
        blur_length_picks.append(blur_length)
        new_sequence_length = length // blur_length
        new_sequence_starts.append(new_sequence_starts[-1]+new_sequence_length)
        tr = list(x+blur_length//2 for x in range(start, end-blur_length+1, blur_length))
        indices_to_transfer += tr
        assert len(tr) == new_sequence_length
    
    newN = new_sequence_starts[-1]
    assert len(indices_to_transfer)==newN
    
    for name, ds in f.items():
        if name == 'sequence_starts':
            fout.create_dataset(name, data = new_sequence_starts)
        elif name == 'images':
            ds_new_images = fout.create_dataset(name, (newN,), chunks=(min(1024,newN),), maxshape=(newN,), dtype=ds_images.dtype)
        elif ds.shape[0] == N:
            fout.create_dataset(name, data = ds[indices_to_transfer,...])
        else:
            assert False

    with progressbar.ProgressBar(max_value=newN) as bar:
        for start, end, blur_length, new_start, new_end in zip(
            sequence_starts[:-1], sequence_starts[1:], blur_length_picks, new_sequence_starts[:-1], new_sequence_starts[1:]):
            images = [
                datasets.preprocessing.imdecode(buffer,color=True) \
                    for buffer in ds_images[start:end]]
            images = np.array(images)
            k = 0
            for i in range(0, end-start-blur_length+1, blur_length):
                avg = np.average(images[i:i+blur_length,...], axis=0)
                ds_new_images[new_start+k] = datasets.preprocessing.imencode(avg)
                k += 1
            assert new_start+k == new_end
            bar.update(new_start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate motion blurred dataset")
    parser.add_argument('source', help="source file", type=str)
    parser.add_argument('destination', help='destination file', type=str, nargs='?', default=None)
    args = parser.parse_args()
    if args.destination is None:
        args.destination = splitext(args.source)+'_filtered.h5'
    assert args.source != args.destination
    rnd = np.random.RandomState(seed=123)
    with h5py.File(args.source, 'r') as f:
        with h5py.File(args.destination, 'w') as fout:
            make_it(f, fout, rnd)