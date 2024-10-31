from torch.utils.data import Dataset, DataLoader
import time
import torch
import numpy as np
import functools
from typing import List
from trackertraincode.datasets.batch import Batch, Metadata

import trackertraincode.train as train
from trackertraincode.datatransformation import PostprocessingDataLoader


def update_fun(net, batch : Batch, optimizer : torch.optim.Optimizer, state : train.State, loss : train.CriterionGroup):
    optimizer.zero_grad()
    y = net(batch['image'])
    lossvals : List[train.LossVal] = loss.evaluate(y, batch, state.step)
    lossvals = [ v._replace(weight = v.val.new_full(v.val.shape, v.weight)) for v in lossvals ]
    l = sum((l.val.sum() for l in lossvals), 0.)
    l.backward()
    optimizer.step()
    lossvals = [ v._replace(val = v.val.detach().to('cpu', non_blocking=True)) for v in lossvals ]
    return [lossvals]


def test_run_the_training():
    class LossMock(object):
        def __call__(self, pred, batch):
            return torch.nn.functional.mse_loss(pred, batch['y'], reduction='none')  
    class MockDataset(Dataset):
        def __init__(self, n):
            self.n = n
        def __len__(self):
            return self.n
        def __getitem__(self, i):
            x = torch.rand((5,))
            y = torch.cos(x)
            return Batch(Metadata(0,batchsize=0), { 'image' : x, 'y' : y })

    net = torch.nn.Sequential(
        torch.nn.Linear(5,128),
        torch.nn.ReLU(),
        torch.nn.Linear(128,5))
    net.get_config = lambda : {}
    trainloader = DataLoader(MockDataset(20), batch_size=2, collate_fn=Batch.collate)
    testloader = PostprocessingDataLoader(MockDataset(8), batch_size=2, collate_fn=Batch.collate, unroll_list_of_batches=True)

    def cbsleep(state):
        print (f"State = {state}")
        time.sleep(0.01)

    c1 = train.Criterion('c1', LossMock(), 1.)
    c2 = train.Criterion('c2', LossMock(), None)
    c3 = train.Criterion('c3', LossMock(), 1.)

    train.run_the_training(
        20, 
        torch.optim.SGD(net.parameters(), 0.1),
        net, 
        trainloader, 
        testloader,
        functools.partial(update_fun,loss=train.CriterionGroup([ c1, c3 ])),
        train.DefaultTestFunc([c1, c2]),
        callbacks = [cbsleep, train.SaveBestCallback(net, 'c2', model_dir='/tmp',retain_max=3)],
        close_plot_on_exit=True,
        plot_save_filename='/tmp/testplot.pdf',artificial_epoch_length=1)


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




if __name__ == '__main__':
    test_plotter()
    test_run_the_training()