from torch.utils.data import Dataset, DataLoader
import time
from matplotlib import pyplot
import torch

import train

def test_run_the_training():

    class LossMock(object):
        def __call__(self, net, pred, batch):
            return torch.nn.functional.mse_loss(pred, torch.zeros(batch['image'].shape[0], 1))
    
    class OptimizerMock(object):
        def zero_grad(self):
            pass
        def step(self):
            pass
    
    class MockDataset(Dataset):
        def __init__(self, n):
            self.n = n
        def __len__(self):
            return self.n
        def __getitem__(self, i):
            return {'image' : torch.rand((5,))}

    net = torch.nn.Linear(5,1)
    trainloader = DataLoader(MockDataset(42), batch_size=2)
    testloader = DataLoader(MockDataset(8), batch_size=2)

    def cbsleep(epoch, net, criteriondata):
        time.sleep(0.1)

    train.run_the_training(10, OptimizerMock(), net, trainloader, testloader, 
        [ train.Criterion('c1', LossMock(), 1., test=True, train=True),
          train.Criterion('c2', LossMock(), None, test=True, train=False),
          train.Criterion('c3', LossMock(), 1., test=False, train=True) ],
        callbacks = [cbsleep, train.SaveBestCallback(net, 'c2', model_dir='/tmp')],
        close_plot_on_exit=True)

if __name__ == '__main__':
    test_run_the_training()