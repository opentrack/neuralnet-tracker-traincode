from matplotlib import pyplot
from collections import defaultdict, namedtuple
import numpy as np
from os.path import join, splitext, isdir
import heapq

import ipywidgets
from IPython.display import display, clear_output

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

# For debugging ...
errordata = None


def run_the_training(
    n_epochs, 
    optimizer,
    net,
    train_loader,
    test_loader,
    criterions,
    criterion_weights,
    num_test_losses,
    model_dir,
    checkpoint_name,
    other_metrics,
    scheduler):
    
    assert isdir(model_dir)

    if checkpoint_name is None:
        checkpoint_name = type(net).__name__+'.ckpt'

    num_criterions = len(criterions)
    criterion_weights_gpu = torch.from_numpy(criterion_weights).type(torch.cuda.FloatTensor)
    
    num_metrics = num_criterions + len(other_metrics)

    fig, axes = pyplot.subplots(num_metrics, 1, figsize=(10, 3*(1+num_metrics)))
    if num_metrics <= 1:
        axes = [axes]

    DisplayData = namedtuple('DisplayData', 'metric run_on_test run_on_train train_curve test_curve ax')

    display_data = []
    for i in range(num_criterions):
        display_data.append(DisplayData(
            metric = criterions[i],
            run_on_test = i < num_test_losses,
            run_on_train = True,
            train_curve = [],
            test_curve = [],
            ax = axes[i]
        ))
    for i, m in enumerate(other_metrics):
        display_data.append(DisplayData(
            metric = m,
            run_on_test = True,
            run_on_train = False,
            train_curve = [],
            test_curve = [],
            ax = axes[i+num_criterions]
        ))

    progress = ipywidgets.IntProgress(
        value=0,
        min=0,
        max=len(train_loader) // train_loader.batch_size,
        step=1,
    )
    statustext = ipywidgets.Label("")
    status_widgets = ipywidgets.HBox([statustext,progress])
    display(status_widgets)

    def update_graph():
        for dd in display_data:
            ax = dd.ax
            ax.clear()
            if dd.run_on_train:
                t, x, xerr = np.array(dd.train_curve).T
                ax.errorbar(t, x, yerr=xerr, label=dd.metric.name, color='r')
            if dd.run_on_test:
                ax.plot(*np.array(dd.test_curve).T, label='test', marker='x', color='b')
            if not hasattr(dd.metric,'plot_logscale') or dd.metric.plot_logscale == True:
                ax.set(yscale='log')
            ax.grid(axis='y', which='both')
            ax.legend()
        fig.canvas.draw()
        clear_output()
        display(status_widgets)
        display(fig)

    best_test_loss = float('inf')
    step = 0    
    for epoch in range(n_epochs):
        epoch_losses = [[] for _ in range(num_criterions) ]
        
        net.train()
        
        for batch_i, data in enumerate(train_loader):

            images = data['image']
            images = images.type(torch.cuda.FloatTensor)
            output_poses = net(images)

            losses = []
            for i, crit in enumerate(criterions):
                loss = crit(net, output_poses, data)
                if not torch.isfinite(loss).all():
                    global errordata
                    errordata = (crit.name, loss, output_poses, net.deformweights, data)
                    assert False
                epoch_losses[i].append((loss.detach().cpu().numpy()))
                losses.append(loss[None])
            losses = torch.cat(losses)
            losses = losses * criterion_weights_gpu
            loss = torch.sum(losses)
            
            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()

            if batch_i % 10 == 9:
                descr = 'Epoch: {}, Batch: {}'.format(epoch + 1, batch_i+1)
                progress.value = batch_i
                statustext.value = descr
            step += 1
    
        for i, curve in enumerate(epoch_losses):
            assert display_data[i].metric is criterions[i]
            display_data[i].train_curve.append((step, np.average(curve), np.std(curve)))
    
        net.eval()

        losses = losses_over_full_dataset(net, [dd.metric for dd in display_data if dd.run_on_test], test_loader)
        loss_iter = iter(losses)
        total_test_loss = 0.
        for i, dd in enumerate(display_data):
            if not dd.run_on_test:
                continue
            loss = next(loss_iter)
            if dd.run_on_train:
                total_test_loss += criterion_weights[i]*loss
            dd.test_curve.append((step, loss))
            print (f"Test loss {dd.metric.name} = {loss}")
        
        if total_test_loss < best_test_loss:
            best_test_loss = total_test_loss
            torch.save(net.state_dict(), join(model_dir, 'best_'+checkpoint_name))

        torch.save(net.state_dict(), join(model_dir, checkpoint_name))

        if scheduler is not None:
            scheduler.step()

        update_graph()

    print('Finished Training')


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
        images = images.type(torch.cuda.FloatTensor)
        with torch.no_grad():
            output_pts = net(images)
            for i, crit in enumerate(criterions):
                value = convert(crit(net, output_pts, sample))
                values[i] += value
        count += 1
    result = (values/count)
    return result


##########################################
## Loss functions
##########################################

def quaternion_distance(a, b):
    # Different variants. Don't make much difference ... I think
    return 1.-torch.square(torch.sum(a * b, dim=1))
    #return 1. - torch.abs(torch.sum(a * b, dim=1))
    #return torch.min(torch.norm(a-b,p=2,dim=1), torch.norm(a+b,p=2,dim=1))


def masked_mean(x, w):
    # Probably don't want this
    x = x * w
    # return torch.sum(x) / (torch.sum(w) + 1.e-3)
    # Rather:
    return torch.mean(x)
    # Otherwise we weight contributions per sample unequally, depending
    # on how many samples of a particular type are in the batch!


class QuatPoseLoss2(object):
    name = 'pose'
    def __call__(self, net, pred, sample):
        target = sample['pose'].type(torch.cuda.FloatTensor)
        coord, quat = pred
        loss = quaternion_distance(quat, target)
        if 'pose_enable' in sample:
            enable_mask = sample['pose_enable'].type(torch.cuda.FloatTensor)
            return masked_mean(loss, enable_mask)
        else:
            return torch.mean(loss)

class CoordPoseLoss(object):
    c = nn.MSELoss(reduction='none')
    name = 'coord'
    def __call__(self, net, pred, sample):
        target = sample['coord'].type(torch.cuda.FloatTensor)
        coord, quat = pred
        loss = torch.mean(self.c(coord, target), dim=1)
        if 'pose_enable' in sample:
            enable_mask = sample['pose_enable'].type(torch.cuda.FloatTensor)
            return masked_mean(loss, enable_mask)
        else:
            return torch.mean(loss)


class QuaternionNormalizationRegularization(object):
    name = 'quatnorm'    
    def __call__(self, net, pred, sample):
        norm_loss = torch.norm(net.out.unnormalized, p=2, dim=1)
        norm_loss = torch.square(1.-norm_loss)
        return torch.mean(norm_loss)

    
class Points3dLoss(object):
    name = 'points3d'
    def __call__(self, net, pred, sample):
        pt3d_68 = sample['pt3d_68']
        pt3d_68 = pt3d_68.type(torch.cuda.FloatTensor)
        assert pt3d_68.shape == net.pt3d_68.shape, f"Mismatch {pt3d_68.shape} vs {net.pt3d_68.shape}"
        assert pt3d_68.shape[1] == 3 #in (2,3)
        assert pt3d_68.shape[2] == 68
        l = F.mse_loss(net.pt3d_68, pt3d_68, reduction='none')
        l = torch.mean(l, dim=[1,2])
        enable_mask = sample['pt3d_68_enable'].type(torch.cuda.FloatTensor)
        return masked_mean(l, enable_mask)

    
class DeformationMagnitudeRegularization(object):
    name = 'deform'
    def __call__(self, net, pred, sample):
        w = net.deformweights
        l = torch.square(w)
        return torch.mean(l)


class BoxLoss(object):
    # Regression loss for bounding box prediction
    name = 'box'
    def __call__(self, net, pred, sample):
        # Only train this if the image shows a face
        target = sample['roi'].type(torch.cuda.FloatTensor)
        pred = net.roi_pred
        return torch.mean(F.smooth_l1_loss(pred, target, reduction='none', beta=0.1))


class LocalizerProbLoss(object):
    # Pytorch Manual:
    #   "This loss combines a Sigmoid layer and the BCELoss in one single class"
    bce = nn.BCEWithLogitsLoss(reduction='mean')
    name = 'prob'
    def __call__(self, net, pred, sample):
        target = sample['hasface'].type(torch.cuda.FloatTensor)
        pred = pred[:,0] # Grab the logit value for the "is face" score
        return self.bce(pred, target)


class LocalizerBoxLoss(object):
    # Regression loss for bounding box prediction
    name = 'box'
    def __call__(self, net, pred, sample):
        # Only train this if the image shows a face
        target = sample['roi'].type(torch.cuda.FloatTensor)
        enable = sample['hasface'].type(torch.cuda.FloatTensor)
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
    for sample in data_loader:
        images = sample['image']
        images = images.type(torch.cuda.FloatTensor)
        with torch.no_grad():
            output_pts = net.inference(images)
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
        images = images.type(torch.cuda.FloatTensor)
        with torch.no_grad():
            preds = net.inference(images)
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
        target = sample['roi'].type(torch.cuda.FloatTensor)
        mask = (sample['hasface']>self.threshold).type(torch.cuda.BoolTensor)
        mask &= (pred['hasface']>self.threshold).type(torch.cuda.BoolTensor)
        err = F.mse_loss(pred['roi'], target[:,:], reduction='none')
        err[~mask,:] = np.nan
        err0 = torch.sum(err[:,:2], dim=1)
        err1 = torch.sum(err[:,2:], dim=1)
        return torch.cat([err0[:,None], err1[:,None]],dim=1)


class LocalizerIsFaceMatches(object):
    def __init__(self, threshold):
        self.threshold = threshold
    def __call__(self, pred, sample):
        target = sample['hasface'].type(torch.cuda.FloatTensor)
        score = pred['hasface']
        match = torch.eq(target>self.threshold, score>self.threshold)
        return match


class PoseErr(object):
    c = nn.MSELoss(reduction='none')
    name = 'pose'
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
        if 'pose_enable' in batch:
            enable_mask = batch['pose_enable'].cpu().numpy() > 0.5
            result[~enable_mask] = np.nan
        return result