import numpy as np
import torch
from matplotlib import pyplot
import tqdm
import itertools

import trackertraincode.vis as vis

import trackertraincode.pipelines
import trackertraincode.datatransformation as dtr
from trackertraincode.datasets.batch import Batch, Metadata
from trackertraincode.pipelines import Tag

NUM_WORKERS = 2

def visualize(loader, loader_outputs_list_of_batches=False):
    def iterate_predictions():
        the_iter = itertools.chain.from_iterable(loader) if loader_outputs_list_of_batches else loader
        for subset in the_iter:
            print(subset.meta.tag)
            subset = dtr.to_device('cpu', subset)
            subset['image'] = trackertraincode.pipelines.unwhiten_image(subset['image'])
            subset = dtr.unnormalize_batch(subset, align_corners=False)
            subset = dtr.to_numpy(subset)
            yield from subset.iter_frames()

    def drawfunc(sample):
        return vis.draw_dataset_sample(sample, label=False)

    keepalive = vis.matplotlib_plot_iterable(iterate_predictions(), drawfunc)
    pyplot.show()


def show_train_test_splits():
    Id = trackertraincode.pipelines.Id
    dsids = [ Id.REPO_300WLP , Id.WFLW_LP ]
    train_loader, test_loader, _ = trackertraincode.pipelines.make_pose_estimation_loaders(
        inputsize = 129, 
        batchsize = 9,
        datasets = dsids,
        auglevel=2)
    visualize(train_loader, loader_outputs_list_of_batches=True)
    visualize(test_loader)



if __name__ == '__main__':  
    show_train_test_splits()