import os
import sys
sys.path.append(os.path.dirname(__file__))

import torch
import more_torch_functions as mtf

from typing import Optional, Callable, Mapping
from torch import Tensor
from collections import OrderedDict
from numbers import Number      

class Parallel(torch.nn.ModuleDict):
    def __init__(self, modules: Optional[Mapping[str, torch.nn.Module]]=None) -> None:
        super().__init__(modules=modules)

    def forward(self, input: Tensor) -> Tensor:
        d = OrderedDict()
        for name, module in self.items():
            d[name] = module(input)

        return d

class MaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, backward_method: Callable[[Tensor, Tensor, Tensor], Tensor]) -> Tensor:
        inputs = input.unbind()

        output = mtf.maximum(inputs)

        ctx.save_for_backward(input, output)
        ctx.backward_method = backward_method
        return output
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        input, output = ctx.saved_tensors
        return ctx.backward_method(input, output, grad_output), None
    
    # /!\ improve needed /!\
    @staticmethod
    def backward_blame_max(input: Tensor, output: Tensor, grad_output: Tensor) -> Tensor:
        return torch.where(input >= output, grad_output, 0)
    
    @staticmethod
    def backward_blame_all(input: Tensor, output: Tensor, grad_output: Tensor) -> Tensor:
        return grad_output.repeat((input.size(0),1,1))

class MaxLayer(torch.nn.Module):
    def __init__(self, backward_method: str="all") -> None:
        super().__init__()
        methods = ("all", "max")
        if backward_method not in methods:
            raise ValueError(f"Supported methods are {', '.join(methods)}")
        self.backward_func = getattr(MaxFunction, "backward_blame_" + backward_method)

    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        if not isinstance(inputs, OrderedDict):
            raise ValueError(f"Use mapping object as inputs instead of {type(inputs)}")
        
        input = torch.stack(list(inputs.values()))

        return MaxFunction.apply(input, self.backward_func)

class MaxHierarchicalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, true: Tensor, fp_mult: Number, cov_mult: Number) -> Tensor:
        inputs = input.unbind()

        output = mtf.maximum(inputs)

        ctx.save_for_backward(input, output)
        ctx.true_labels = true
        ctx.fp_mult = fp_mult
        ctx.cov_mult = cov_mult
        return output
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        input, output = ctx.saved_tensors
        true = ctx.true_labels

        # false positive multiplier
        mult = torch.where((input>=.5)&(true<.5), ctx.fp_mult, 1)
        mult[0] = 1

        # coverage multiplier
        cond = torch.full(grad_output.size(), False)
        for i, _ in enumerate(input[2:]):
            cond|=(input[i+1]<.5)
            mult[i+2] *= torch.where(cond&(true>.5), ctx.cov_mult, 1)

        grad = mult * grad_output
        
        return grad, None


class MaxHierarchicalLayer(torch.nn.Module):
    def __init__(self, fp_multiplier: Optional[Number]=10, cov_multiplier: Optional[Number]=2) -> None:
        super().__init__()
        self.true_labels = None
        self.fp_mult = fp_multiplier
        self.cov_mult = cov_multiplier

    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        if not isinstance(inputs, OrderedDict):
            raise ValueError(f"Use mapping object as inputs instead of {type(inputs)}")
        
        input = torch.stack(list(inputs.values()))

        return MaxHierarchicalFunction.apply(input, self.true_labels, self.fp_mult, self.cov_mult)
    
class MinFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor, backward_method: Callable[[Tensor, Tensor, Tensor], Tensor]) -> Tensor:
        inputs = input.unbind()

        output = mtf.minimum(inputs)

        ctx.save_for_backward(input, output)
        ctx.backward_method = backward_method
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        input, output = ctx.saved_tensors
        return ctx.backward_method(input, output, grad_output), None

    # /!\ improve needed /!\
    @staticmethod
    def backward_blame_min(input: Tensor, output: Tensor, grad_output: Tensor) -> Tensor:
        return torch.where(input <= output, grad_output, 0)

    @staticmethod
    def backward_blame_all(input: Tensor, output: Tensor, grad_output: Tensor) -> Tensor:
        return grad_output.repeat((input.size(0),1,1))

class MinLayer(torch.nn.Module):
    def __init__(self, backward_method: str="all") -> None:
        super().__init__()
        methods = ("all", "max")
        if backward_method not in methods:
            raise ValueError(f"Supported methods are {', '.join(methods)}")
        self.backward_func = getattr(MinFunction, "backward_blame_" + backward_method)

    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        if not isinstance(inputs, OrderedDict):
            raise ValueError(f"Use mapping object as inputs instead of {type(inputs)}")

        input = torch.stack(list(inputs.values()))

        return MinFunction.apply(input, self.backward_func)