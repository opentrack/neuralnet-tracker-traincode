from matplotlib import pyplot
from collections import namedtuple, defaultdict
import numpy as np
from os.path import join, isdir
from typing import Any, Dict, List, Sequence, Tuple, Union, Optional, Callable, NamedTuple
import tqdm
import multiprocessing
import queue
import time
import dataclasses
import copy
import os

from torch import Tensor
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, CyclicLR


from trackertraincode.datasets.batch import Batch
import trackertraincode.utils as utils


def weighted_mean(x : Tensor, w : Tensor, dim) -> Tensor:
    return torch.sum(x*w, dim).div(torch.sum(w,dim))

Criterion = namedtuple('Criterion', 'name f w', defaults=(None,None,1.))

@dataclasses.dataclass
class History:
    train : List[Any] = dataclasses.field(default_factory=list)
    test  : List[Any] = dataclasses.field(default_factory=list)
    current_train_buffer : List[Any] = dataclasses.field(default_factory=list)
    logplot : bool = True


class LossVal(NamedTuple):
    val : Tensor
    weight : float
    name : str


# From https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
def in_notebook():
    from IPython import get_ipython
    return get_ipython() is not None


class TrainHistoryPlotter(object):
    def __init__(self, save_filename = None):
        self.histories = defaultdict(History)
        self.queue = multiprocessing.Queue(maxsize=100)
        self.plotting = multiprocessing.Process(None,self.run_plotting,args=(self.queue, save_filename))
        self.plotting.start()

    @staticmethod
    def ensure_axes_are_ready(fig : pyplot.Figure, axes, last_rows, histories):
        num_rows = len(histories)
        if num_rows != last_rows:
            if num_rows > 5:
                r, c = (num_rows+1)//2, 2
            else:
                r, c = num_rows, 1
            fig.clear()
            fig.set_figheight(3*r)
            axes = fig.subplots(r, c)
            if c > 1:
                axes = axes.ravel()
            if num_rows==1:
                axes = [axes]
        else:
            for ax in axes:
                ax.clear()
        return fig, axes, num_rows

    @staticmethod
    def process_items(queue_, histories):
        have_new_items = False
        while True:
            try:
                result = queue_.get(block=True, timeout=0.1)
            except queue.Empty:
                return histories, True, have_new_items
            else:
                if result is None:
                    return histories, False, have_new_items
                else:
                    have_new_items = True
                    if histories is None:
                        histories = dict(result)
                    else:
                        for name, history in result:
                            histories[name].train += history.train
                            histories[name].test += history.test
                    del result

    @staticmethod
    def update_actual_graphs(histories, fig, axes, num_rows):
        fig, axes, num_rows = TrainHistoryPlotter.ensure_axes_are_ready(fig, axes, num_rows, histories)
        for ax, (name, history) in zip(axes, histories.items()):
            if name == 'lr':
                t, lr  = np.array(history.test).T
                ax.plot(t, lr, label = 'lr', marker='o', color='k')
                ax.set(yscale='log')
                ax.grid(axis='y', which='both')
                ax.legend()
            elif name == '|grad L|':
                t, x, xerr = np.array(history.train).T
                ax.errorbar(t, x, yerr=xerr, color='k', label=name)
                ax.set(yscale='log')
                ax.grid(axis='y', which='both')
                ax.legend()
            else:
                if history.train:
                    t, x, xerr = np.array(history.train).T
                    ax.errorbar(t, x, yerr=xerr, label=name, color='r')
                if history.test:
                    ax.plot(*np.array(history.test).T, label='test '+name, marker='x', color='b')
                # FIXME: Hack with `startswith('nll')`
                if history.logplot and not name.startswith('nll'):
                    ax.set(yscale='log')
                ax.grid(axis='y', which='both')
                ax.legend()
        pyplot.tight_layout()
        return fig, axes, num_rows

    @staticmethod
    def run_plotting(queue_, save_filename):
        histories = None
        fig, axes, num_rows = pyplot.figure(figsize=(10, 10)), None, 0
        fig.show()
        while pyplot.get_fignums():
            histories, keep_going, have_new_items = TrainHistoryPlotter.process_items(queue_, histories)
            if have_new_items:
                fig, axes, num_rows = TrainHistoryPlotter.update_actual_graphs(histories, fig, axes, num_rows)
                fig.canvas.draw_idle()
                if save_filename:
                    fig.savefig(save_filename)
            fig.canvas.flush_events()
            if keep_going:
                time.sleep(0.1)
            else:
                pyplot.close(fig)

    def add_train_point(self, epoch, step, name, value):
        self.histories[name].current_train_buffer.append((epoch, value))

    def add_test_point(self, epoch, name, value):
        self.histories[name].test.append((epoch,value))

    @staticmethod
    def summarize_single_train_history(k, h : History):
        if not h.current_train_buffer:
            return
        epochs, values = zip(*h.current_train_buffer)
        try:
            h.train.append((np.average(epochs), np.average(values), np.std(values)))
        except FloatingPointError:
            print (f"Floating point error at {k} in epochs {np.average(epochs)} with values:\n {str(values)}\n")
            h.train.append((np.average(epochs), np.nan, np.nan))
        h.current_train_buffer = []


    def summarize_train_values(self):
        for k, h in self.histories.items():
            TrainHistoryPlotter.summarize_single_train_history(k, h)

    def update_graph(self):
        self.queue.put(copy.deepcopy(list(self.histories.items())))
        for h in self.histories.values():
            h.test = []
            h.train = []

    def close(self):
        self.queue.put(None)


class ConsoleTrainOutput(object):
    def __init__(self):
        self.histories = defaultdict(History)

    def add_train_point(self, epoch, step, name, value):
        self.histories[name].current_train_buffer.append((epoch, value))

    def add_test_point(self, epoch, name, value):
        self.histories[name].test.append((epoch,value))

    def summarize_train_values(self):
        for k, h in self.histories.items():
            TrainHistoryPlotter.summarize_single_train_history(k, h)

    def update_graph(self):
        print ("Losses:")
        for name, h in self.histories.items():
            if h.train:
                epoch, mean, std = h.train[-1]
                train_str = f'{mean:.4f} +/- {std:.4f}'
            else:
                train_str = '----'
            if h.test:
                epoch, val = h.test[-1]
                test_str = f'{val:.4f}'
            else:
                test_str = '----'
            print (f"{name}: Train: {train_str}, Test: {test_str}")
            h.test = []
            h.train = []

    def close(self):
        pass


@dataclasses.dataclass
class State:
    lossvals : Dict[str,Any]
    epoch : int
    step : int
    visualizer : Union[TrainHistoryPlotter, ConsoleTrainOutput]
    grad_norm : Optional[float]
    num_samples_per_loss : Optional[defaultdict[int]] = None



class VirtualEpochBatchIter(object):
    def __init__(self, loader, num_samples):
        self.num_samples = num_samples
        self.iter = utils.cycle(loader)
    def __iter__(self):
        with tqdm.tqdm(total=self.num_samples) as bar:
            counter = 0
            while counter < self.num_samples:
                batch = next(self.iter)
                yield batch
                size = sum(b.meta.batchsize for b in batch) if isinstance(batch,list) else batch.meta.batchsize
                bar.update(size)
                counter += size


def NormalEpochBatchIter(loader):
    for batch in tqdm.tqdm(loader):
        yield batch


def run_the_training(
    n_epochs, 
    optimizer,
    net,
    train_loader,
    test_loader,
    update_func,
    test_func,
    callbacks = [],
    scheduler = None,
    artificial_epoch_length = None,
    close_plot_on_exit = False,
    plotting = True,
    plot_save_filename = None):

    plotter = \
        TrainHistoryPlotter(plot_save_filename) if plotting \
            else ConsoleTrainOutput()

    state = State(
        lossvals = {},
        epoch = 0,
        step = 0,
        visualizer = plotter,
        grad_norm=None,
        num_samples_per_loss = defaultdict(int)
    )

    with tqdm.tqdm(total=n_epochs) as epochbar:
        if artificial_epoch_length is None:
            train_iter = NormalEpochBatchIter(train_loader)
        else:
            train_iter = VirtualEpochBatchIter(train_loader, artificial_epoch_length)

        for epoch in range(n_epochs):
            state.epoch = epoch

            net.train()

            for batch in train_iter:
                lossvals = update_func(net, batch, optimizer, state)
                state.lossvals = lossvals
                for name, val in lossvals:
                    plotter.add_train_point(epoch, state.step, name, val.detach().to('cpu',non_blocking=True))
                state.step += 1
                if state.grad_norm is not None:
                    plotter.add_train_point(epoch, state.step, '|grad L|', state.grad_norm)

            if state.num_samples_per_loss is not None:
                print ("Samples: ", ', '.join([f"{k}: {v}" for k,v in  state.num_samples_per_loss.items()]))
                state.num_samples_per_loss = None # Disable printing

            plotter.summarize_train_values()

            net.eval()

            lossvals = losses_over_full_dataset(net, test_loader, test_func)
            state.lossvals = lossvals
            for name, l in lossvals:
                plotter.add_test_point(epoch, name, l)

            if scheduler is not None:
                last_lr = next(iter(scheduler.get_last_lr()))
                scheduler.step()
                plotter.add_test_point(epoch, 'lr', last_lr)

            plotter.update_graph()
            epochbar.update(1)
            
            for callback in callbacks:
                callback(state)

    if close_plot_on_exit:
        plotter.close()


class SaveCallback(object):
    def __init__(self, net, prefix='', model_dir=None):
        self.net = net
        self.prefix = prefix
        self.model_dir = model_dir

    @property
    def net_name(self):
        return self.net.name if hasattr(self.net,'name') else type(self.net).__name__

    @property
    def filename(self):
        return join(self.model_dir, f'{self.prefix}{self.net_name}.ckpt')

    def save(self):
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.net.state_dict(), self.filename)


class SaveBestCallback(SaveCallback):
    def __init__(self, net, loss_names : Union[str,List[str]], model_dir=None, save_name_prefix : Optional[str] = None, weights : Optional[List[float]] = None, retain_max = 1):
        if isinstance(loss_names, str):
            loss_names = [ loss_names ]
        if save_name_prefix is None:
            save_name_prefix = '-'.join(loss_names)
        if weights is None:
            weights = [ 1. for _ in loss_names ]
        assert len(weights) == len(loss_names)
        super().__init__(net, f'best_{save_name_prefix}_', model_dir)
        self.best_test_loss = float('inf')
        self.loss_names = loss_names
        self.weights = weights
        self.save_name_prefix = save_name_prefix
        self.retain_max = retain_max
        self.num = 0
        self.filenames = []
        assert retain_max >= 1

    def update_prefix(self):
        if self.retain_max <= 1:
            return
        self.prefix = f'best_{self.save_name_prefix}_{self.num:02d}_'
        self.num += 1

    def roll_files(self):
        if self.retain_max <= 1:
            return
        self.filenames.append(self.filename)
        if len(self.filenames) > self.retain_max:
            first = self.filenames.pop(0)
            symlinkname = join(self.model_dir, f'best_{self.save_name_prefix}_{self.net_name}.ckpt')
            os.unlink(first)
            try:
                os.unlink(symlinkname)
            except FileNotFoundError:
                pass
            os.symlink(self.filename, symlinkname)

    def _combined_loss(self, state : State):
        d = dict(state.lossvals)
        return sum((d[k]*w) for k,w in zip(self.loss_names, self.weights))

    def __call__(self, state : State):
        l = self._combined_loss(state)
        if l < self.best_test_loss:
            self.update_prefix()
            self.best_test_loss = l
            print (f"Save model for best {self.save_name_prefix}={l} to {self.filename} at epoch {state.epoch}")
            self.save()
            self.roll_files()


class SaveInIntervals(SaveCallback):
    def __init__(self, net, start_epoch, interval, model_dir=None):
        super().__init__(net, 'intervalxx_', model_dir)
        self.start_epoch = start_epoch
        self.interval = interval
    
    def __call__(self, state : State):
        since_start = state.epoch - self.start_epoch
        if since_start >= 0 and since_start % self.interval == 0:
            self.prefix = f'interval{state.epoch:02d}_'
            self.save()


def _check_loss(loss, pred, batch, name):
    if not torch.isfinite(loss).all():
        import pickle
        with open('/tmp/pred.pkl', 'wb') as f:
            pickle.dump(pred,f)
        with open('/tmp/batch.pkl', 'wb') as f:
            pickle.dump(batch, f)
        raise RuntimeError(f"Non-finite value created by loss {name}")


def checked_criterion_eval(lossfunc : Callable, pred : Dict, batch : Dict) -> List[LossVal]:
    loss = lossfunc(pred, batch)
    if isinstance(loss, Tensor):
        # Only enable for debugging:
        # _check_loss(loss, pred, batch, type(lossfunc).__name__)
        return [LossVal(loss,1.,'')]
    elif isinstance(loss, LossVal):
        return [loss]
    else:
        return loss


def compute_inf_norm_of_grad(net : nn.Module):
    device = next(iter(net.parameters())).device
    result = torch.zeros((), device=device, dtype=torch.float32, requires_grad=False)
    for p in net.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(float('inf'))
            result = torch.maximum(param_norm, result)
    return result


def _convert_multi_task_loss_list(multi_task_terms: Dict[str,List[Tuple[Tensor,float,int]]], device : str) -> Dict[str,Tuple[Tensor,Tensor,Tensor]]:
    # Convert list of list of tuples to list of tuples of tensors
    def _cvt_item(k, vals_weights_idx):
        vals, weights, idxs = zip(*vals_weights_idx)
        #print (f"CVT {k}: v {[v.shape for v in vals]}, w {[w.shape for w in weights]}")
        vals = [ (val*w).mean() for val, w in zip(vals, weights) ]
        vals = torch.stack(vals)
        #weights = torch.as_tensor(weights, dtype=torch.float32).to(device, non_blocking=True)
        #weights = torch.stack(weights)
        weights = torch.stack([w.mean() for w in weights])
        idxs = torch.as_tensor(idxs)
        return vals, weights, idxs
    return { k:_cvt_item(k,v) for k,v in multi_task_terms.items() }


def _accumulate_losses_over_batches(multi_task_terms: Sequence[Tuple[Tensor,Tensor,Tensor]], batchsizes : Tensor):
    all_lossvals = 0.
    for vals, weights, idxs in multi_task_terms:
        all_lossvals = all_lossvals + torch.sum(vals*batchsizes[idxs])
    all_lossvals = all_lossvals / torch.sum(batchsizes)
    return all_lossvals


def default_update_fun(net, batch : List[Batch], optimizer : torch.optim.Optimizer, state : State, loss):
    assert isinstance(batch, list)
    
    optimizer.zero_grad()

    inputs = torch.concat([b['image'] for b in batch], dim=0)
    
    assert torch.amin(inputs)>=-2., f"fuck {torch.amin(inputs)}"
    assert torch.amax(inputs)<= 2., f"fuck {torch.amin(inputs)}"

    preds = net(inputs)

    all_multi_task_terms = defaultdict(list)
    batchsizes = torch.tensor([ subset.meta.batchsize for subset in batch ], dtype=torch.float32).to(inputs.device, non_blocking=True)

    offset = 0
    for subset_idx, subset in enumerate(batch):
        frames_in_subset, = subset.meta.prefixshape
        subpreds = { k:v[offset:offset+frames_in_subset,...] for k,v in preds.items() }

        loss_func_of_subset = loss[subset.meta.tag] if isinstance(loss, dict) else loss
        multi_task_terms =  checked_criterion_eval(loss_func_of_subset, subpreds, subset)

        if 'dataset_weight' in subset:
            dataset_weights = subset['dataset_weight']
            assert dataset_weights.size(0) == subset.meta.batchsize
        else:
            dataset_weights = torch.ones((frames_in_subset,), device=inputs.device)

        for elem in multi_task_terms:
            weight = dataset_weights * elem.weight
            assert weight.shape == elem.val.shape, f"Bad loss {elem.name}"
            all_multi_task_terms[elem.name].append((elem.val, weight))
            if state.num_samples_per_loss is not None:
                state.num_samples_per_loss[elem.name] += frames_in_subset
        
        del multi_task_terms, loss_func_of_subset

        offset += frames_in_subset

    def _concat_over_subsets(items : List[Tuple[Tensor,Tensor]]):
        values, weights = zip(*items)
        return (
            torch.concat(values),
            torch.concat(weights))
    all_multi_task_terms = { k:_concat_over_subsets(v) for k,v in all_multi_task_terms.items() }

    loss_sum = torch.concat([ (values*weights) for values,weights in all_multi_task_terms.values() ]).sum() / batchsizes.sum()
    loss_sum.backward()
    
    if 1:
        state.grad_norm = compute_inf_norm_of_grad(net).to('cpu', non_blocking=True)
        # Gradients get very large more often than looks healthy ... Loss spikes a lot.
        # Gradient magnitudes below 0.1 seem to be normal. Initially gradients are larger,
        nn.utils.clip_grad_norm_(net.parameters(), 0.1, norm_type=float('inf'))

    optimizer.step()

    # This is only for logging
    for k, (vals, weights) in all_multi_task_terms.items():
        all_multi_task_terms[k] = weighted_mean(vals, weights, 0)
        
    return list(all_multi_task_terms.items())


class MultiTaskLoss(object):
    def __init__(self, criterions : Sequence[Criterion]):
        self.criterions = criterions

    def __iadd__(self, crit : Criterion):
        self.criterions += crit
        return self

    def __call__(self, pred, batch):
        def _eval_crit(crit : Criterion):
            return [ 
                LossVal(lv.val, lv.weight*crit.w, crit.name+lv.name) for lv in checked_criterion_eval(crit.f, pred, batch) ]
        return sum((_eval_crit(c) for c in self.criterions), start=[])


class DefaultTestFunc(object):
    def __init__(self, criterions):
        self.criterions = criterions
    def __iadd__(self, crit):
        self.criterions += crit
        return self
    def __call__(self, net, batch):
        images = batch['image']
        pred = net(images)
        return [ (crit.name,crit.f(pred, batch)) for crit in self.criterions ]


def losses_over_full_dataset(net : nn.Module, data_loader, test_func):
    def convert(value):
        if isinstance(value, (np.ndarray,float)):
            return value
        elif isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        else:
            assert False, f'cannot convert value {value} of type {type(value)}'

    values = {}
    counts = defaultdict(lambda : 0)
    for batch in data_loader:
        actual_test_func = test_func[batch.meta.tag] if isinstance(test_func, dict) else test_func
        with torch.no_grad():
            named_losses = actual_test_func(net, batch)
        for name, val in named_losses:
            val = convert(val.mean())
            try:
                accumulator = values[name]
            except KeyError:
                values[name] = val
            else:
                accumulator += val
            counts[name] += 1
    for k, v in values.items():
        v /= counts[k]
    return list(values.items())


##########################################
## Schedules
##########################################

def TriangularSchedule(optimizer, min_lr, lr, num_steps, *args, **kwargs):
    num_steps_up = min(max(1,num_steps*3//10), 33)
    num_steps_down = num_steps - num_steps_up
    return CyclicLR(optimizer, min_lr, lr, num_steps_up, num_steps_down, *args, mode='triangular', cycle_momentum=False, **kwargs)


def LinearUpThenSteps(optimizer, num_up, gamma, steps):
    steps = [0] + steps
    def lr_func(i):
        if i < num_up:
            return ((i+1)/num_up)
        else:
            step_index = [j for j,step in enumerate(steps) if i>step][-1]
            return gamma**step_index
    return LambdaLR(optimizer, lr_func)