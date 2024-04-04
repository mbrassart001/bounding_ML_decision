import torch
import functools
import itertools

from torch import Tensor
from typing import Callable, Iterable

def reset_model(model: torch.nn.Module) -> None:
    for layer in model.modules():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()

def freeze_model(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False

def _apply_reduce_(func: Callable[[Tensor, Tensor], Tensor], tensors: Iterable[Tensor]) -> Tensor:
    if len(tensors) < 1:
        raise ValueError
    return functools.reduce(func, tensors)

def logical_big_or(tensors: Iterable[Tensor]) -> Tensor:
    return _apply_reduce_(torch.logical_or, tensors)

def bitwise_big_or(tensors: Iterable[Tensor]) -> Tensor:
    return _apply_reduce_(torch.bitwise_or, tensors)

def logical_big_and(tensors: Iterable[Tensor]) -> Tensor:
    return _apply_reduce_(torch.logical_and, tensors)

def bitwise_big_and(tensors: Iterable[Tensor]) -> Tensor:
    return _apply_reduce_(torch.bitwise_and, tensors)

def logical_cascade_xor(tensors: Iterable[Tensor]) -> Tensor:
    return _apply_reduce_(torch.logical_xor, tensors)

def bitwise_cascade_xor(tensors: Iterable[Tensor]) -> Tensor:
    return _apply_reduce_(torch.bitwise_xor, tensors)

def maximum(tensors: Iterable[Tensor]) -> Tensor:
    return _apply_reduce_(torch.maximum, tensors)

def minimum(tensors: Iterable[Tensor]) -> Tensor:
    return _apply_reduce_(torch.minimum, tensors)
