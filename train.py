from matplotlib import pyplot
from collections import namedtuple
import numpy as np
from os.path import join, isdir, dirname
import heapq
from typing import List
import tqdm
import sys
import multiprocessing
import queue
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import neuralnets.modelcomponents
import neuralnets.torchquaternion as torchquaternion

# For debugging ...
errordata = None

Criterion = namedtuple('Criterion', 'name f w test train logplot', defaults=(None,None,1.,False,False,True))
History = namedtuple('History', 'name test train logplot')

# From https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
def in_notebook():
    from IPython import get_ipython
    return get_ipython() is not None


class TrainHistoryPlotter(object):
    def __init__(self, num_rows):
        assert num_rows>=1
        self.queue = multiprocessing.Queue()
        self.plotting = multiprocessing.Process(None,self.run_plotting,args=(num_rows, self.queue,))
        self.plotting.start()

    @staticmethod
    def run_plotting(num_rows, queue_):
        figh = 3*num_rows
        fig, axes = pyplot.subplots(num_rows, 1, figsize=(10, figh))
        if num_rows==1:
            axes = [axes]
        fig.show()
        while pyplot.get_fignums():
            try:
                result=queue_.get(block=True, timeout=0.1)
            except queue.Empty:
                #pyplot.pause(0.1)
                fig.canvas.flush_events()
                time.sleep(0.1)
            else:
                if result is not None:
                    histories, learning_rates = result
                    for ax, history in zip(axes, histories):
                        ax.clear()
                        if history.train:
                            t, x, xerr = np.array(history.train).T
                            ax.errorbar(t, x, yerr=xerr, label=history.name, color='r')
                        if history.test:
                            ax.plot(*np.array(history.test).T, label='test '+history.name, marker='x', color='b')
                        if history.logplot:
                            ax.set(yscale='log')
                        ax.grid(axis='y', which='both')
                        ax.legend()
                    if learning_rates is not None:
                        ax = axes[-1]
                        ax.clear()
                        t, lr = np.array(learning_rates).T
                        ax.plot(t, lr, label = 'lr', marker='o', color='k')
                        ax.set(yscale='log')
                        ax.grid(axis='y', which='both')
                        ax.legend()
                    fig.canvas.flush_events()
                    fig.canvas.draw_idle()
                else:
                    # Terminate loop
                    pyplot.close(fig)

    def update_graph(self, histories, learning_rates):
        self.queue.put_nowait((histories, learning_rates))

    def close(self):
        self.queue.put(None)

def losses_on_batch(net, batch, criterions):
    images = batch['image']
    output = net(images)
    losses = []
    for i, crit in enumerate(criterions):
        loss = crit(output, batch)
        if not torch.isfinite(loss).all():
            global errordata
            errordata = (loss, output, str(crit.__class__))
            print (errordata)
            assert False
        losses.append(loss)
    return losses


def weighted_sum(items, weights):
    items = torch.cat([x[None] for x in items])
    return torch.sum(items*weights)


def run_the_training(
    n_epochs, 
    optimizer,
    net,
    train_loader,
    test_loader,
    criterions : List[Criterion],
    callbacks = [],
    scheduler = None,
    close_plot_on_exit = False):

    num_batches = max(1, len(train_loader))

    CritData = namedtuple('CritData', 'crit train_curve test_curve train_accum')

    display_data = []
    for i,crit in enumerate(criterions):
        display_data.append(CritData(
            crit = crit,
            train_curve = [],
            test_curve = [],
            train_accum = []
        ))
    test_metrics = [ x for x in display_data if x.crit.test ]
    train_losses = [ x for x in display_data if x.crit.train ]
    weights_list = [ x.crit.w for x in train_losses]
    learning_rates = [ ]
    criterion_weights = torch.Tensor(weights_list).to(dtype=torch.float32)

    plotter = TrainHistoryPlotter(len(criterions)+(0 if not scheduler else 1))
    if not in_notebook():
        pyplot.show(block=False)

    step = 0
    with tqdm.tqdm(total=n_epochs) as epochbar:
        for epoch in range(n_epochs):
            for x in train_losses:
                x.crit.f.epoch = epoch
                x.train_accum.clear()

            net.train()

            for batch_i, batch in enumerate(tqdm.tqdm(train_loader)):
                losses = losses_on_batch(net, batch, [x.crit.f for x in train_losses])
                loss = weighted_sum(losses, criterion_weights.to(losses[0].device))
                
                optimizer.zero_grad()
                
                loss.backward()

                optimizer.step()

                for x, l in zip(train_losses, losses):
                    x.train_accum.append((l.detach().cpu().numpy()))

                # if batch_i % 10 == 0:
                #     descr = 'Epoch: {}, Batch: {}/{}'.format(epoch + 1, batch_i+1, num_batches)
                #     plotter.update_status(descr)
                step += 1
                # if batch_i % 10 == 0:
                #     epochbar.update(batch_i + num_batches*epoch)

            for x in train_losses:
                x.train_curve.append((step, np.average(x.train_accum), np.std(x.train_accum)))

            net.eval()

            losses = losses_over_full_dataset(net, [x.crit.f for x in test_metrics], test_loader)
            for loss, x in zip(losses, test_metrics):
                x.test_curve.append((step, loss))

            if scheduler is not None:
                last_lr = next(iter(scheduler.get_last_lr()))
                learning_rates.append((step, last_lr))
                scheduler.step()

            plotter.update_graph([History(
                    name = x.crit.name, 
                    test = x.test_curve,
                    train = x.train_curve,
                    logplot = x.crit.logplot)
                for x in display_data], learning_rates)
            # TODO: notebook
            # if in_notebook():
            #     from IPython.display import display, clear_output
            #     clear_output(wait=True)
            #     display(pyplot.gcf())
            # else:
            #     fig = pyplot.gcf()
            #     fig.canvas.draw_idle()
            #     fig.canvas.flush_events()
            epochbar.update(1)
            
            for callback in callbacks:
                callback(epoch, net, [{
                    'crit' : x.crit,
                    'trainloss' : x.train_curve[-1][1] if x.crit.train else None,
                    'testloss' : x.test_curve[-1][1] if x.crit.test else None
                } for x in display_data])

    if close_plot_on_exit:
        plotter.close()

class SaveBestCallback(object):
    def __init__(self, net, loss_name, filename = None, model_dir=None):
        self.best_test_loss = float('inf')
        self.loss_name = loss_name
        if filename is None:
            self.filename = join(model_dir, 'best_'+loss_name+'_'+type(net).__name__+'.ckpt')
        else:
            self.filename = filename
        assert isdir(dirname(self.filename))

    def __call__(self, epoch, net, lossdata):
        l = { x['crit'].name:x['testloss'] for x in lossdata }[self.loss_name]
        if l < self.best_test_loss:
            self.best_test_loss = l
            print (f"Save model for best {self.loss_name}={l} to {self.filename} at epoch {epoch}")
            torch.save(net.state_dict(), self.filename)


def losses_over_full_dataset(net, criterions, data_loader):
    def convert(value):
        if isinstance(value, (np.ndarray,float)):
            return value
        elif isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        else:
            assert False

    values = np.zeros(len(criterions), dtype=np.float64)
    count = 0
    for sample in data_loader:
        images = sample['image']
        with torch.no_grad():
            output_pts = net(images)
            for i, crit in enumerate(criterions):
                value = convert(crit(output_pts, sample))
                values[i] += value
        count += 1
    result = (values/count)
    return result


##########################################
## Loss functions
##########################################


def masked_mean(x, w):
    # Probably don't want this
    x = x * w
    # return torch.sum(x) / (torch.sum(w) + 1.e-3)
    # Rather:
    return torch.mean(x)
    # Otherwise we weight contributions per sample unequally, depending
    # on how many samples of a particular type are in the batch!


def _element_wise_rotation_loss(pred, target):
    if len(target.shape)==2 and target.shape[-1]==4:
        return torchquaternion.distance(pred, target)
    assert False, "target tensor has invalid shape"


class QuatPoseLoss2(object):
    def __call__(self, pred, sample):
        target = sample['pose']
        quat = pred['pose']
        loss = _element_wise_rotation_loss(quat, target)
        if 'pose_enable' in sample:
            enable_mask = sample['pose_enable']
            return masked_mean(loss, enable_mask)
        else:
            return torch.mean(loss)

class CoordPoseLoss(object):
    c = nn.MSELoss(reduction='none')
    def __call__(self, pred, sample):
        target = sample['coord']
        coord = pred['coord']
        loss = torch.mean(self.c(coord, target), dim=1)
        if 'pose_enable' in sample:
            enable_mask = sample['pose_enable']
            return masked_mean(loss, enable_mask)
        else:
            return torch.mean(loss)


class CoordAttentionMapLoss(object):
    def __init__(self):
        self.grid = None

    def generate_featuremap_blobs(self, shape, half_size, positions, size_scale, device):
        if self.grid is None:
            c = torch.linspace(-half_size, half_size, shape)
            y, x = torch.meshgrid([c,c])
            y = y.float().to(device)
            x = x.float().to(device)
            self.grid = (y,x)
        y, x = self.grid
        y = y.flatten()[None,:] - positions[:,1,None]
        x = x.flatten()[None,:] - positions[:,0,None]
        dist_sqr = (x*x+y*y)
        sigma = 2.*half_size/(shape-1.)*size_scale
        val = torch.exp(-dist_sqr  / sigma**2)
        val = val.reshape((-1,shape,shape))
        val /= torch.sum(val, dim=(1,2), keepdims=True)
        return val

    def __call__(self, pred, sample):
        maps = pred['attention_logits']
        B = maps.size(0)
        S = maps.size(2)
        target = sample['coord'][:,:2]
        target_maps  = self.generate_featuremap_blobs(S, 1., target, 1., maps.device)
        if 0:
            idx = 3
            a = target_maps[idx].detach().cpu().numpy()
            b = maps[idx].detach().cpu().numpy()
            fig,ax = pyplot.subplots(1,2)
            ax[0].imshow(a)
            ax[1].imshow(b[0])
            pyplot.show()
        maps = maps.reshape((B, -1))
        maps = F.log_softmax(maps, dim=1)
        loss = torch.mean(F.kl_div(maps, target_maps.reshape(B,-1), reduction='none'), dim=1)
        enable_mask = sample['pose_enable']
        return masked_mean(loss, enable_mask)

class ShapeParameterLoss(object):
    def __call__(self, pred, sample):
        target = sample['shapeparam']
        loss = torch.mean(F.mse_loss(pred['shapeparam'], target, reduction='none'), dim=1)
        enable_mask = sample['pose_enable']
        return masked_mean(loss, enable_mask)


class QuaternionNormalizationRegularization(object):
    def __call__(self, pred, sample):
        unnormalized = pred['unnormalized_quat']
        assert len(unnormalized.shape)==2 and unnormalized.shape[-1]==4
        enable_mask = 0.5*(sample['pose_enable'] + sample['pt3d_68_enable'])
        norm_loss = torch.norm(unnormalized, p=2, dim=1)
        norm_loss = torch.square(1.-norm_loss)
        return masked_mean(norm_loss, enable_mask)

class ShapeRegularization(object):
    def __call__(self, pred, sample):
        params = pred['shapeparam']
        enable_mask = sample['pose_enable']
        loss = torch.mean(torch.square(params), dim=1)
        return masked_mean(loss, enable_mask)


class Points3dLoss(object):
    def __call__(self, pred, sample):
        pt3d_68 = sample['pt3d_68']
        assert pt3d_68.shape == pt3d_68.shape, f"Mismatch {pt3d_68.shape} vs {pred['pt3d_68'].shape}"
        assert pt3d_68.shape[1] == 3
        assert pt3d_68.shape[2] == 68
        l = F.mse_loss(pred['pt3d_68'], pt3d_68, reduction='none')
        l = torch.mean(l, dim=[1,2])
        enable_mask = sample['pt3d_68_enable']
        return masked_mean(l, enable_mask)

class BoxLoss(object):
    def __init__(self, dataname = 'roi'):
        self.dataname = dataname
    # Regression loss for bounding box prediction
    def __call__(self, pred, sample):
        # Only train this if the image shows a face
        target = sample[self.dataname]
        pred = pred[self.dataname]
        enable_mask = 0.5*(sample['pose_enable'] + sample['pt3d_68_enable'])
        l = torch.mean(F.smooth_l1_loss(pred, target, reduction='none', beta=0.1), dim=1)
        return masked_mean(l, enable_mask)


class HasFaceLoss(object):
    def __call__(self, pred, sample):
        target = sample['hasface']
        logits = pred['hasface_logits']
        enable_mask = sample['hasface_enable']
        l = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        assert l.shape == (target.size(0),)
        return masked_mean(l, enable_mask)


class LocalizerProbLoss(object):
    # Pytorch Manual:
    #   "This loss combines a Sigmoid layer and the BCELoss in one single class"
    bce = nn.BCEWithLogitsLoss(reduction='mean')
    def __call__(self, net, pred, sample):
        target = sample['hasface']
        pred = pred[:,0] # Grab the logit value for the "is face" score
        return self.bce(pred, target)


class LocalizerBoxLoss(object):
    # Regression loss for bounding box prediction
    def __call__(self, net, pred, sample):
        # Only train this if the image shows a face
        target = sample['roi']
        enable = sample['hasface']
        pred = pred[:,1:]
        err = F.smooth_l1_loss(pred, target, reduction='none', beta=0.1)
        return torch.mean(enable[:,None]*err)


##########################################
## Metrics for evaluation
##########################################
# And functions to run them over the datasets.

def metrics_over_full_dataset(net, metrics, data_loader):
    assert not net.training
    results = [ [] for _ in range(len(metrics)) ]
    for sample in tqdm.tqdm(data_loader):
        images = sample['image']
        with torch.no_grad():
            output_pts = net(images)
            for i, crit in enumerate(metrics):
                value = crit(output_pts, sample)
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy()
                value = np.atleast_1d(value)
                results[i].append(value)
    for i, r in enumerate(results):
        results[i] = np.concatenate(r, axis=0)
    return results


def k_worst_over_dataset(net, data_loader, metric, k):
    assert not net.training
    min_heap = []
    for batch in data_loader:
        images = batch['image']
        with torch.no_grad():
            preds = net(images)
            values = metric(preds, batch)
            if isinstance(values, torch.Tensor):
                values = values.cpu().numpy()
            for sample, pred, value in zip(utils.undo_collate(batch), utils.undo_collate(preds), values):
                heapq.heappush(min_heap, (value, sample, pred))
                if len(min_heap)>k:
                    heapq.heappop(min_heap)
    return min_heap


class LocalizerBoxMeanSquareErrors(object):
    def __init__(self, threshold):
        self.threshold = threshold
    def __call__(self, pred, sample):
        target = sample['roi']
        mask = (sample['hasface']>self.threshold)
        mask &= (pred['hasface']>self.threshold)
        err = F.mse_loss(pred['roi'], target[:,:], reduction='none')
        err[~mask,:] = np.nan
        err0 = torch.sum(err[:,:2], dim=1)
        err1 = torch.sum(err[:,2:], dim=1)
        return torch.cat([err0[:,None], err1[:,None]],dim=1)


class LocalizerIsFaceMatches(object):
    def __init__(self, threshold):
        self.threshold = threshold
    def __call__(self, pred, sample):
        target = sample['hasface']
        score = pred['hasface']
        match = torch.eq(target>self.threshold, score>self.threshold)
        return match


class PoseErr(object):
    c = nn.MSELoss(reduction='none')
    def __call__(self, pred, batch):
        assert isinstance(pred, dict)
        coord_target = batch['coord'].cpu().numpy()
        quat_target = batch['pose'].cpu().numpy()
        coord = pred['coord']
        quat  = pred['pose']
        coord = coord.cpu().numpy()
        quat  = quat.cpu().numpy()
        coord_errs = np.abs(coord-coord_target)
        rot_errs = (utils.convert_to_rot(quat_target).inv()*utils.convert_to_rot(quat)).magnitude()
        result = np.concatenate([rot_errs[:,None], coord_errs], axis=1)
        return result


class PoseEnableWeights(object):
    def __call__(self, pred, batch):
        return batch['pose_enable'].cpu().numpy()


class EulerAngleErrors(object):
    def __call__(self, pred, batch):
        quat_target = batch['pose'].cpu().numpy()
        quat  = pred['pose'].cpu().numpy()
        euler_target = np.array([ utils.inv_rotation_conversion_from_hell(q) for q in utils.convert_to_rot(quat_target) ])
        euler = np.array([ utils.inv_rotation_conversion_from_hell(q) for q in utils.convert_to_rot(quat) ])
        errors = utils.angle_errors(euler_target, euler)
        return errors
