import itertools
from matplotlib import pyplot
from collections import namedtuple, defaultdict
import numpy as np
from os.path import join, isdir
from typing import Any, Dict, List, Sequence, Tuple, Union, Optional, Callable, NamedTuple
import tqdm
import multiprocessing
import queue
import time
import pickle
import dataclasses
import copy
import os
import math

from torch import Tensor
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, CyclicLR


from trackertraincode.datasets.batch import Batch
import trackertraincode.neuralnets.io
import trackertraincode.utils as utils


def weighted_mean(x : Tensor, w : Tensor, dim) -> Tensor:
    return torch.sum(x*w, dim).div(torch.sum(w,dim))


class LossVal(NamedTuple):
    val : Tensor
    weight : float
    name : str


def concatenated_lossvals_by_name(vals : list[LossVal]):
    '''Sorts by name and concatenates.

    Assumes that names can occur multiple times. Then corresponding weights and
    values are concatenated. Useful for concatenating the loss terms from different
    sub-batches.

    Return:
        Dict[name,(values,weights)]
    '''
    value_lists = defaultdict(list)
    weight_lists = defaultdict(list)
    for v in vals:
        value_lists[v.name].append(v.val)
        weight_lists[v.name].append(v.weight)
    return {
        k:(torch.concat(value_lists[k]),torch.concat(weight_lists[k])) for k in value_lists
    }


class Criterion(NamedTuple):
    name : str
    f : Callable[[Batch,Batch],Tensor]
    w : Union[float,Callable[[int],float]]

    def evaluate(self, pred, batch, step) -> List[LossVal]:
        val = self.f(pred,batch)
        w = self._eval_weight(step)
        return [ LossVal(val, w, self.name) ]

    def _eval_weight(self, step):
        if isinstance(self.w, float):
            return self.w
        else:
            return self.w(step)


class CriterionGroup(NamedTuple):
    criterions : List[Union['CriterionGroup',Criterion]]
    name : str = ''
    w : Union[float,Callable[[int],float]] = 1.0

    def _eval_weight(self, step):
        if isinstance(self.w, float):
            return self.w
        else:
            return self.w(step)

    def evaluate(self, pred, batch, step) -> List[LossVal]:
        w = self._eval_weight(step)
        lossvals = sum((c.evaluate(pred, batch, step) for c in self.criterions), start=[])
        lossvals = [ LossVal(v.val,v.weight*w,self.name+v.name) for v in lossvals ]
        return lossvals


@dataclasses.dataclass
class History:
    train : List[Any] = dataclasses.field(default_factory=list)
    test  : List[Any] = dataclasses.field(default_factory=list)
    current_train_buffer : List[Any] = dataclasses.field(default_factory=list)
    logplot : bool = True



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
            if next(iter(values)).shape != ():
                values = np.concatenate(values)
            else:
                values = np.stack(values)
            h.train.append((np.average(epochs), np.average(values), np.std(values)))
        except FloatingPointError:
            with np.printoptions(precision=4, suppress=True, threshold=20000):
                print (f"Floating point error at {k} in epochs {np.average(epochs)} with values:\n {str(values)} of which there are {len(values)}\n")
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
                trainlossvals = update_func(net, batch, optimizer, state)
                for name, (val, _) in concatenated_lossvals_by_name(itertools.chain.from_iterable(trainlossvals)).items():
                    plotter.add_train_point(epoch, state.step, name, val)
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
        trackertraincode.neuralnets.io.save_model(self.net, self.filename)


class DebugData(NamedTuple):
    parameters : dict[str,Tensor]
    batches : list[Batch]
    preds : dict[str,Tensor]
    lossvals : list[list[LossVal]]

    def is_bad(self):
        '''Checks data for badness.
        
        Currently NANs and input value range.
        
        Return:
            True if so.
        '''
        #TODO: decouple for name of input tensor
        for k,v in self.parameters.items():
            if torch.any(torch.isnan(v)):
                print(f"{k} is NAN")
                return True
        for b in self.batches:
            for k, v in b.items():
                if torch.any(torch.isnan(v)):
                    print(f"{k} is NAN")
                    return True
            inputs = b['image']
            if  torch.amin(inputs)<-2. or torch.amax(inputs)>2.:
                print(f"Input image {inputs.shape} exceeds value limits with {torch.amin(inputs)} to {torch.amax(inputs)}")
                return True
        for k,v in self.preds.items():
            if torch.any(torch.isnan(v)):
                print(f"{k} is NAN")
                return True
        for lv_list in self.lossvals:
            for lv in lv_list:
                if torch.any(torch.isnan(lv.val)):
                    print(f"{lv.name} is NAN")
                    return True
        return False

class DebugCallback():
    '''For dumping a history of stuff when problems are detected.'''
    def __init__(self):
        self.history_length = 3
        self.debug_data : List[DebugData] = []
        self.filename = '/tmp/notgood.pkl'
    
    def observe(self, net_pre_update : nn.Module, batches : list[Batch], preds : dict[str,Tensor], lossvals : list[list[LossVal]]):
        '''Record and check.
        Args:
            batches: Actually sub-batches
            lossvals: One list of loss terms per sub-batch
        '''
        dd = DebugData(
            {k:v.detach().to('cpu', non_blocking=True,copy=True) for k,v in net_pre_update.state_dict().items()},
            [b.to('cpu', non_blocking=True,copy=True) for b in batches ],
            {k:v.detach().to('cpu', non_blocking=True,copy=True) for k,v in preds.items()},
            lossvals
        )
        if len(self.debug_data) >= self.history_length:
            self.debug_data.pop(0)
        self.debug_data.append(dd)
        torch.cuda.current_stream().synchronize()
        if dd.is_bad():
            with open(self.filename, 'wb') as f:
                pickle.dump(self.debug_data, f)
            raise RuntimeError("Bad state detected")


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


def compute_inf_norm_of_grad(net : nn.Module):
    device = next(iter(net.parameters())).device
    result = torch.zeros((), device=device, dtype=torch.float32, requires_grad=False)
    for p in net.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(float('inf'))
            result = torch.maximum(param_norm, result)
    return result


# g_debug = DebugCallback()


def default_update_fun(net, batch : List[Batch], optimizer : torch.optim.Optimizer, state : State, loss : dict[Any, Criterion | CriterionGroup] | Criterion | CriterionGroup):
    # global g_debug

    optimizer.zero_grad()

    inputs = torch.concat([b['image'] for b in batch], dim=0)

    preds = net(inputs)

    lossvals_by_name = defaultdict(list)
    all_lossvals : list[list[LossVal]] = []

    # Iterate over different datasets / loss configurations
    offset = 0
    for subset in batch:
        frames_in_subset, = subset.meta.prefixshape
        subpreds = { k:v[offset:offset+frames_in_subset,...] for k,v in preds.items() }

        # Get loss function and evaluate
        loss_func_of_subset : Union[Criterion,CriterionGroup] = loss[subset.meta.tag] if isinstance(loss, dict) else loss
        multi_task_terms : List[LossVal] =  loss_func_of_subset.evaluate(subpreds, subset, state.epoch)

        # Support loss weighting by datasets
        if 'dataset_weight' in subset:
            dataset_weight = subset['dataset_weight']
            assert dataset_weight.size(0) == subset.meta.batchsize
            multi_task_terms = [ v._replace(weight=v.weight*dataset_weight) for v in multi_task_terms ]
        else:
            # Else, make the weight member a tensor the same shape as the loss values
            multi_task_terms = [ v._replace(weight=v.val.new_full(size=v.val.shape,fill_value=v.weight)) for v in multi_task_terms ]

        all_lossvals.append(multi_task_terms)
        del multi_task_terms, loss_func_of_subset

        offset += frames_in_subset

    batchsize = sum(subset.meta.batchsize for subset in batch)
    # Concatenate the loss values over the sub-batches. 
    lossvals_by_name = concatenated_lossvals_by_name(itertools.chain.from_iterable(all_lossvals))
    # Compute weighted average, dividing by the batch size which is equivalent to substituting missing losses by 0.
    loss_sum = torch.concat([ (values*weights) for values,weights in lossvals_by_name.values() ]).sum() / batchsize

    # Transfer to CPU
    for loss_list in all_lossvals:
        for i, v in enumerate(loss_list):
            loss_list[i] = v._replace(val = v.val.detach().to('cpu', non_blocking=True))

    loss_sum.backward()

    # g_debug.observe(net, batch, preds, all_lossvals)

    if 1:
        state.grad_norm = compute_inf_norm_of_grad(net).to('cpu', non_blocking=True)
        # Gradients get very large more often than looks healthy ... Loss spikes a lot.
        # Gradient magnitudes below 0.1 seem to be normal. Initially gradients are larger,
        nn.utils.clip_grad_norm_(net.parameters(), 1.0, norm_type=float('inf'))

    optimizer.step()

    torch.cuda.current_stream().synchronize()
    return all_lossvals


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


def ExponentialUpThenSteps(optimizer, num_up, gamma, steps):
    steps = [0] + steps
    def lr_func(i):
        eps = 1.e-2
        scale = math.log(eps)
        if i < num_up:
            f = ((i+1)/num_up)
            #return torch.sigmoid((f - 0.5) * 15.)
            # a * exp(f / l) | f=1 == 1.
            # a * exp(f / l) | f=0 ~= eps
            # => a = eps
            # => ln(1./eps) = 1./l
            return eps * math.exp(-scale*f)
        else:
            step_index = [j for j,step in enumerate(steps) if i>step][-1]
            return gamma**step_index
    return LambdaLR(optimizer, lr_func)