import itertools
from matplotlib import pyplot
from collections import namedtuple, defaultdict
import numpy as np
from os.path import join, isdir
from typing import Any, Dict, List, Sequence, Tuple, Union, Optional, Callable, NamedTuple, Mapping, Protocol
import tqdm
import multiprocessing
import queue
import time
import pickle
import dataclasses
import copy
import os
import math

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch import Tensor
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, CyclicLR
from torch.optim.swa_utils import AveragedModel

from trackertraincode.neuralnets.io import save_model
from trackertraincode.datasets.batch import Batch


class LossVal(NamedTuple):
    val: Tensor
    weight: float
    name: str


def concatenated_lossvals_by_name(vals: list[LossVal]):
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
    return {k: (torch.concat(value_lists[k]), torch.concat(weight_lists[k])) for k in value_lists}


class Criterion(NamedTuple):
    name: str
    f: Callable[[Batch, Batch], Tensor]
    w: Union[float, Callable[[int], float]]

    def evaluate(self, pred, batch, step) -> List[LossVal]:
        val = self.f(pred, batch)
        w = self._eval_weight(step)
        return [LossVal(val, w, self.name)]

    def _eval_weight(self, step):
        if isinstance(self.w, float):
            return self.w
        else:
            return self.w(step)


class CriterionGroup(NamedTuple):
    criterions: List[Union['CriterionGroup', Criterion]]
    name: str = ''
    w: Union[float, Callable[[int], float]] = 1.0

    def _eval_weight(self, step):
        if isinstance(self.w, float):
            return self.w
        else:
            return self.w(step)

    def evaluate(self, pred, batch, step) -> List[LossVal]:
        w = self._eval_weight(step)
        lossvals = sum((c.evaluate(pred, batch, step) for c in self.criterions), start=[])
        lossvals = [LossVal(v.val, v.weight * w, self.name + v.name) for v in lossvals]
        return lossvals


@dataclasses.dataclass
class History:
    train: List[Any] = dataclasses.field(default_factory=list)
    test: List[Any] = dataclasses.field(default_factory=list)
    current_train_buffer: List[Any] = dataclasses.field(default_factory=list)
    logplot: bool = True


class TrainHistoryPlotter(object):
    def __init__(self, save_filename=None):
        self.histories = defaultdict(History)
        self.queue = multiprocessing.Queue(maxsize=100)
        self.plotting = multiprocessing.Process(None, self.run_plotting, args=(self.queue, save_filename))
        self.plotting.start()

    @staticmethod
    def ensure_axes_are_ready(fig: pyplot.Figure, axes, last_rows, histories):
        num_rows = len(histories)
        if num_rows != last_rows:
            if num_rows > 5:
                r, c = (num_rows + 1) // 2, 2
            else:
                r, c = num_rows, 1
            fig.clear()
            fig.set_figheight(3 * r)
            axes = fig.subplots(r, c)
            if c > 1:
                axes = axes.ravel()
            if num_rows == 1:
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
                t, lr = np.array(history.test).T
                ax.plot(t, lr, label='lr', marker='o', color='k')
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
                    ax.plot(*np.array(history.test).T, label='test ' + name, marker='x', color='b')
                # FIXME: Hack with `startswith('nll')`
                if history.logplot and (not name.startswith('nll')) and (not name == 'loss'):
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
        self.histories[name].test.append((epoch, value))

    @staticmethod
    def summarize_single_train_history(k, h: History):
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
                print(
                    f"Floating point error at {k} in epochs {np.average(epochs)} with values:\n {str(values)} of which there are {len(values)}\n"
                )
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
        self.plotting.join()


class ConsoleTrainOutput(object):
    def __init__(self):
        self.histories = defaultdict(History)

    def add_train_point(self, epoch, step, name, value):
        self.histories[name].current_train_buffer.append((epoch, value))

    def add_test_point(self, epoch, name, value):
        self.histories[name].test.append((epoch, value))

    def summarize_train_values(self):
        for k, h in self.histories.items():
            TrainHistoryPlotter.summarize_single_train_history(k, h)

    def update_graph(self):
        print("Losses:")
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
            print(f"{name}: Train: {train_str}, Test: {test_str}")
            h.test = []
            h.train = []

    def close(self):
        pass


class DebugData(NamedTuple):
    parameters: dict[str, Tensor]
    batches: list[Batch]
    preds: dict[str, Tensor]
    lossvals: list[list[LossVal]]

    def is_bad(self):
        '''Checks data for badness.

        Currently NANs and input value range.

        Return:
            True if so.
        '''
        # TODO: decouple for name of input tensor
        for k, v in self.parameters.items():
            if torch.any(torch.isnan(v)):
                print(f"{k} is NAN")
                return True
        for b in self.batches:
            for k, v in b.items():
                if torch.any(torch.isnan(v)):
                    print(f"{k} is NAN")
                    return True
            inputs = b['image']
            if torch.amin(inputs) < -2.0 or torch.amax(inputs) > 2.0:
                print(
                    f"Input image {inputs.shape} exceeds value limits with {torch.amin(inputs)} to {torch.amax(inputs)}"
                )
                return True
        for k, v in self.preds.items():
            if torch.any(torch.isnan(v)):
                print(f"{k} is NAN")
                return True
        for lv_list in self.lossvals:
            for lv in lv_list:
                if torch.any(torch.isnan(lv.val)):
                    print(f"{lv.name} is NAN")
                    return True
        return False


class DebugCallback:
    '''For dumping a history of stuff when problems are detected.'''

    def __init__(self):
        self.history_length = 3
        self.debug_data: List[DebugData] = []
        self.filename = '/tmp/notgood.pkl'

    def observe(
        self, net_pre_update: nn.Module, batches: list[Batch], preds: dict[str, Tensor], lossvals: list[list[LossVal]]
    ):
        '''Record and check.
        Args:
            batches: Actually sub-batches
            lossvals: One list of loss terms per sub-batch
        '''
        dd = DebugData(
            {k: v.detach().to('cpu', non_blocking=True, copy=True) for k, v in net_pre_update.state_dict().items()},
            [b.to('cpu', non_blocking=True, copy=True) for b in batches],
            {k: v.detach().to('cpu', non_blocking=True, copy=True) for k, v in preds.items()},
            lossvals,
        )
        if len(self.debug_data) >= self.history_length:
            self.debug_data.pop(0)
        self.debug_data.append(dd)
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()
        if dd.is_bad():
            with open(self.filename, 'wb') as f:
                pickle.dump(self.debug_data, f)
            raise RuntimeError("Bad state detected")


# g_debug = DebugCallback()


def default_compute_loss(
    preds : dict[str,Tensor],
    batch: List[Batch],
    current_epoch: int,
    loss: dict[Any, Criterion | CriterionGroup] | Criterion | CriterionGroup,
):
    """
    Return:
        Loss sum for backprop
        LossVals - one nested list per batch item. Tensors transfered to the cpu.
    """
    # global g_debug

    #inputs = torch.concat([b['image'] for b in batch], dim=0)
    #preds = net(inputs)

    lossvals_by_name = defaultdict(list)
    all_lossvals: list[list[LossVal]] = []

    # Iterate over different datasets / loss configurations
    offset = 0
    for subset in batch:
        (frames_in_subset,) = subset.meta.prefixshape
        subpreds = {k: v[offset : offset + frames_in_subset, ...] for k, v in preds.items()}

        # Get loss function and evaluate
        loss_func_of_subset: Union[Criterion, CriterionGroup] = (
            loss[subset.meta.tag] if isinstance(loss, dict) else loss
        )
        multi_task_terms: List[LossVal] = loss_func_of_subset.evaluate(subpreds, subset, current_epoch)

        # Support loss weighting by datasets
        if 'dataset_weight' in subset:
            dataset_weight = subset['dataset_weight']
            assert dataset_weight.size(0) == subset.meta.batchsize
            multi_task_terms = [v._replace(weight=v.weight * dataset_weight) for v in multi_task_terms]
        else:
            # Else, make the weight member a tensor the same shape as the loss values
            multi_task_terms = [
                v._replace(weight=v.val.new_full(size=v.val.shape, fill_value=v.weight)) for v in multi_task_terms
            ]

        all_lossvals.append(multi_task_terms)
        del multi_task_terms, loss_func_of_subset

        offset += frames_in_subset

    batchsize = sum(subset.meta.batchsize for subset in batch)
    # Concatenate the loss values over the sub-batches.
    lossvals_by_name = concatenated_lossvals_by_name(itertools.chain.from_iterable(all_lossvals))
    # Compute weighted average, dividing by the batch size which is equivalent to substituting missing losses by 0.
    loss_sum = torch.concat([(values * weights) for values, weights in lossvals_by_name.values()]).sum() / batchsize

    # Transfer to CPU
    for loss_list in all_lossvals:
        for i, v in enumerate(loss_list):
            loss_list[i] = v._replace(val=v.val.detach().to('cpu', non_blocking=True))
    if torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()
    return loss_sum, all_lossvals


class LightningModelWrapper(Protocol):
    @property
    def model() -> nn.Module: ...


class SwaCallback(Callback):
    def __init__(self, start_epoch):
        super().__init__()
        self._swa_model: AveragedModel | None = None
        self._start_epoch = start_epoch

    @property
    def swa_model(self):
        return self._swa_model.module

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._swa_model = AveragedModel(pl_module.model, device="cpu", use_buffers=True)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch > self._start_epoch:
            self._swa_model.update_parameters(pl_module.model)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        assert self._swa_model is not None
        swa_filename = join(trainer.default_root_dir, f"swa.ckpt")
        save_model(self._swa_model.module, swa_filename)


class MetricsGraphing(Callback):
    def __init__(self):
        super().__init__()
        self._visu: TrainHistoryPlotter | None = None
        self._metrics_accumulator = defaultdict(list)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        assert self._visu is None
        self._visu = TrainHistoryPlotter(save_filename=join(trainer.default_root_dir, "train.pdf"))

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int
    ):
        mt_losses: dict[str, torch.Tensor] = outputs["mt_losses"]
        for k, v in mt_losses.items():
            self._visu.add_train_point(trainer.current_epoch, batch_idx, k, v.numpy())
        self._visu.add_train_point(trainer.current_epoch, batch_idx, "loss", outputs["loss"].detach().cpu().numpy())

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.lr_scheduler_configs:  # scheduler is not None:
            scheduler = next(
                iter(trainer.lr_scheduler_configs)
            ).scheduler  # Pick the first scheduler (and there should only be one)
            last_lr = next(iter(scheduler.get_last_lr()))  # LR from the first parameter group
            self._visu.add_test_point(trainer.current_epoch, "lr", last_lr)

        self._visu.summarize_train_values()

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._metrics_accumulator = defaultdict(list)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: list[LossVal],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        for val in outputs:
            self._metrics_accumulator[val.name].append(val.val)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._visu is None:
            return
        for k, v in self._metrics_accumulator.items():
            self._visu.add_test_point(trainer.current_epoch - 1, k, torch.cat(v).mean().cpu().numpy())
        if trainer.current_epoch > 0:
            self._visu.update_graph()

    def close(self):
        self._visu.close()


class SimpleProgressBar(Callback):
    """Creates progress bars for total training time and progress of per epoch."""

    def __init__(self, batchsize: int):
        super().__init__()
        self._bar: tqdm.tqdm | None = None
        self._epoch_bar: tqdm.tqdm | None = None
        self._batchsize = batchsize

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._bar = tqdm.tqdm(total=trainer.max_epochs, desc='Training', position=0)
        self._epoch_bar = tqdm.tqdm(total=trainer.num_training_batches * self._batchsize, desc="Epoch", position=1)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._bar.close()
        self._epoch_bar.close()
        self._bar = None
        self._epoch_bar = None

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._epoch_bar.reset(self._epoch_bar.total)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._bar.update(1)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Mapping[str, Any],
        batch: list[Batch] | Batch,
        batch_idx: int,
    ) -> None:
        n = sum(b.meta.batchsize for b in batch) if isinstance(batch, list) else batch.meta.batchsize
        self._epoch_bar.update(n)


##########################################
## Schedules
##########################################


def TriangularSchedule(optimizer, min_lr, lr, num_steps, *args, **kwargs):
    num_steps_up = min(max(1, num_steps * 3 // 10), 33)
    num_steps_down = num_steps - num_steps_up
    return CyclicLR(
        optimizer, min_lr, lr, num_steps_up, num_steps_down, *args, mode='triangular', cycle_momentum=False, **kwargs
    )


def LinearUpThenSteps(optimizer, num_up, gamma, steps):
    steps = [0] + steps

    def lr_func(i):
        if i < num_up:
            return (i + 1) / num_up
        else:
            step_index = [j for j, step in enumerate(steps) if i > step][-1]
            return gamma**step_index

    return LambdaLR(optimizer, lr_func)


def ExponentialUpThenSteps(optimizer, num_up, gamma, steps):
    steps = [0] + steps

    def lr_func(i):
        eps = 1.0e-2
        scale = math.log(eps)
        if i < num_up:
            f = (i + 1) / num_up
            # return torch.sigmoid((f - 0.5) * 15.)
            # a * exp(f / l) | f=1 == 1.
            # a * exp(f / l) | f=0 ~= eps
            # => a = eps
            # => ln(1./eps) = 1./l
            return eps * math.exp(-scale * f)
        else:
            step_index = [j for j, step in enumerate(steps) if i > step][-1]
            return gamma**step_index

    return LambdaLR(optimizer, lr_func)
