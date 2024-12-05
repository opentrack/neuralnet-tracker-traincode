from torch.utils.data import Dataset, DataLoader
import time
import torch
import numpy as np
import functools
from typing import List
from trackertraincode.datasets.batch import Batch, Metadata

import trackertraincode.train as train

def test_plotter():
    plotter = train.TrainHistoryPlotter()
    names = [ 'foo', 'bar', 'baz', 'lr' ]
    for e in range(4):
        for t in range(5):
            for name in names[:-2]:
                plotter.add_train_point(e, t, name, 10. + e + np.random.normal(0., 1.,(1,)))
        for name in names[1:]:
            plotter.add_test_point(e, name, 9. + e + np.random.normal())
        plotter.summarize_train_values()
        plotter.update_graph()
    plotter.close()