from trackertraincode.neuralnets import io

from torch import nn
from torch import Tensor
import torch

import pytest


def test_load_save(tmp_path):
    class Model(nn.Module):
        def __init__(self, input_size: int, width: int):
            super().__init__()
            self._input_size = input_size
            self._width = width
            self.layers = nn.Sequential(
                nn.Linear(input_size, width), nn.ReLU(), nn.Linear(width, 1)
            )

        def __call__(self, x: Tensor):
            return self.layers(x)

        def get_config(self):
            return {"input_size": self._input_size, "width": self._width}

    m = Model(42, 32)
    m(torch.zeros((1, 42)))  # Model is executable?

    filename = tmp_path / "model.ckpt"
    io.save_model(m, filename)

    restored: Model = io.load_model(filename, [Model])
    for p, q in zip(restored.parameters(), m.parameters()):
        torch.testing.assert_close(p, q)
