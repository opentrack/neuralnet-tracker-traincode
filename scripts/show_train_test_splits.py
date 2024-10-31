#!/usr/bin/env python3
import numpy as np
import torch
from matplotlib import pyplot
import tqdm
import itertools
import argparse

import trackertraincode.vis as vis

import trackertraincode.pipelines
import trackertraincode.datatransformation as dtr
from trackertraincode.datasets.batch import Batch, Metadata
from trackertraincode.pipelines import Tag

from train_poseestimator import parse_dataset_definition

NUM_WORKERS = 2

def visualize(loader, loader_outputs_list_of_batches=False):
    def iterate_predictions():
        the_iter = itertools.chain.from_iterable(loader) if loader_outputs_list_of_batches else loader
        for subset in the_iter:
            print(subset.meta.tag)
            subset = subset.to('cpu')
            subset['image'] = trackertraincode.pipelines.unwhiten_image(subset['image'])
            subset = dtr.unnormalize_batch(subset)
            subset = dtr.to_numpy(subset)
            yield from subset.iter_frames()

    def drawfunc(sample):
        return vis.draw_dataset_sample(sample, label=False)

    keepalive = vis.matplotlib_plot_iterable(iterate_predictions(), drawfunc)
    pyplot.show()


def show_train_test_splits():
    parser = argparse.ArgumentParser(description="Trains the model")
    parser.add_argument('--ds-weighting', help="Sample dataset with equal probability and use weights for scaling their losses", 
                        action="store_false", default=True, dest="ds_weight_are_sampling_frequencies")
    parser.add_argument('--ds', help='Which datasets to train on. See code.', type=str, default='300wlp')
    parser.add_argument('--raug', default=30, type=float, dest='rotation_aug_angle')
    parser.add_argument('--no-imgaug', default=True, action='store_false', dest='with_image_aug')
    parser.add_argument('--roi-override', default='extent_to_forehead', type=str, choices=['extent_to_forehead', 'original', 'landmarks'], dest='roi_override')
    args = parser.parse_args()

    dsids, dataset_weights = parse_dataset_definition(args.ds)

    train_loader, test_loader, _ = trackertraincode.pipelines.make_pose_estimation_loaders(
        inputsize = 129, 
        batchsize = 9,
        datasets = dsids,
        dataset_weights=dataset_weights,
        use_weights_as_sampling_frequency=args.ds_weight_are_sampling_frequencies,
        enable_image_aug=args.with_image_aug,
        rotation_aug_angle=args.rotation_aug_angle,
        roi_override=args.roi_override)
    visualize(train_loader, loader_outputs_list_of_batches=True)
    visualize(test_loader)



if __name__ == '__main__':  
    show_train_test_splits()