from os.path import splitext
import argparse
import numpy as np
import os
import torch
import copy
from typing import Any, Protocol, Container

import torch.onnx
import torch.nn as nn


class SavableModel(Protocol):
    def state_dict() -> dict[str, Any]: ...
    def get_config() -> dict[str, Any]: ...
    def load_state_dict(d: dict[str, Any], strict: bool) -> None: ...


def complement_lightning_checkpoint(model: SavableModel, checkpoint: dict[str, Any]):
    assert "state_dict" in checkpoint
    checkpoint.update({"class_name": model.__class__.__name__, "config": model.get_config()})


def save_model(model: SavableModel, filename: str):
    contents = {"state_dict": model.state_dict()}
    complement_lightning_checkpoint(model, contents)
    torch.save(contents, filename)


class InvalidFileFormatError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


def load_model(filename: str, class_candidates: Container[type]):
    contents = torch.load(filename, weights_only=True)
    if not all(x in contents for x in ["state_dict", "class_name", "config"]):
        raise InvalidFileFormatError(f"Bad dict contents. Got {list(contents.keys())}")
    class_name = contents["class_name"]
    class_ = {c.__name__: c for c in class_candidates}[class_name]
    instance: SavableModel = class_(**contents["config"])
    instance.load_state_dict(contents["state_dict"], strict=True)
    return instance
