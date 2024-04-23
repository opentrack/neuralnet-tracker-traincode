from matplotlib import pyplot
import numpy as np
import tqdm
from contextlib import closing
import argparse

from face3drotationaugmentation.dataset300wlp import DatasetAFLW2k3D
from face3drotationaugmentation.generate import augment_eyes_only, make_sample_for_passthrough
from face3drotationaugmentation.datasetwriter import dataset_writer

deg2rad = np.pi/180.


def main(filename : str, outputfilename : str, max_num_frames : int, prob_closed_eyes : float):
    rng = np.random.RandomState(seed=1234567)

    with closing(DatasetAFLW2k3D(filename)) as ds300wlp, dataset_writer(outputfilename) as writer:
        num_frames = min(max_num_frames, len(ds300wlp))
        for _, sample in tqdm.tqdm(zip(range(num_frames), ds300wlp), total=num_frames):
            if sample['scale'] <= 0.: # TODO: actual decent validation?
                print (f"Error: invalid head size = {sample['scale']}. Putting original sample!")
                generated_sample = make_sample_for_passthrough(sample)
            else:
                generated_sample = augment_eyes_only(prob_closed_eyes, rng, sample)
            writer.write(sample['name'], generated_sample)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Only Eye Augmentation")
    parser.add_argument("aflw2k3d", type=str, help="zip file")
    parser.add_argument("outputfilename", type=str, help="hdf5 file")
    parser.add_argument("-n", help="subset of n samples", type=int, default=1<<32)
    parser.add_argument("--prob-closed-eyes", type=float, default=0., help="probability for closing eyes (between 0 and 1)")
    args = parser.parse_args()
    if not (args.outputfilename.lower().endswith('.h5') or args.outputfilename.lower().endswith('.hdf5')):
            raise ValueError("outputfilename must have hdf5 filename extension")
    main(args.aflw2k3d, args.outputfilename, args.n, prob_closed_eyes=args.prob_closed_eyes)