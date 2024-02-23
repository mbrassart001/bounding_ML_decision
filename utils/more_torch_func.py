import torch
import functools

def logical_big_or(*tensors):
    return functools.reduce(torch.logical_or, tensors)

def bitwise_big_or(*tensors):
    return functools.reduce(torch.bitwise_or, tensors)

def logical_big_and(*tensors):
    return functools.reduce(torch.logical_and, tensors)

def bitwise_big_and(*tensors):
    return functools.reduce(torch.bitwise_and, tensors)

def logical_cascade_xor(*tensors):
    return functools.reduce(torch.logical_xor, tensors)

def bitwise_cascade_xor(*tensors):
    return functools.reduce(torch.bitwise_xor, tensors)
