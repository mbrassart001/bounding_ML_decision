import os
import sys
sys.path.append(os.path.dirname(__file__))

import torch
import more_torch_functions as mtf

from collections import OrderedDict        

class Parallel(torch.nn.ModuleDict):
    def __init__(self, modules=None):
        super().__init__(modules=modules)

    def forward(self, input):
        d = OrderedDict()
        for name, module in self.items():
            d[name] = module(input)

        return d

class MaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, backward_method):
        inputs = input.unbind()

        # maximum ? addition ? some sort of mean ? mix of the methods ?
        output = mtf.maximum(inputs)

        ctx.save_for_backward(input, output)
        ctx.backward_method = backward_method
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        return ctx.backward_method(input, output, grad_output), None
    
    # /!\ improve needed /!\
    @staticmethod
    def backward_blame_max(input, output, grad_output):
        return torch.where(input >= output, grad_output, 0)
    
    @staticmethod
    def backward_blame_all(input, output, grad_output):
        return grad_output.repeat((input.size(0),1,1))

class MaxLayer(torch.nn.Module):
    def __init__(self, backward_method="all"):
        super().__init__()
        methods = ("all", "max")
        if backward_method not in methods:
            raise ValueError(f"Supported methods are {', '.join(methods)}")
        self.backward_func = getattr(MaxFunction, "backward_blame_" + backward_method)

    def forward(self, inputs):
        if not isinstance(inputs, OrderedDict):
            raise ValueError("Use OrderedDict")
        
        input = torch.stack(list(inputs.values()))

        return MaxFunction.apply(input, self.backward_func)

class MaxHierarchicalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, true):
        inputs = input.unbind()

        output = mtf.maximum(inputs)

        ctx.save_for_backward(input, output)
        ctx.true_labels = true
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        true = ctx.true_labels

        # false positive multiplier
        mult = torch.where((input>=.5)&(true<.5), 10, 1)
        mult[0] = 1

        # coverage multiplier
        cond = torch.full(grad_output.size(), False)
        for i, _ in enumerate(input[2:]):
            cond|=(input[i+1]<.5)
            mult[i+2] *= torch.where(cond&(true>.5), 2, 1)

        grad = mult * grad_output
        
        return grad, None


class MaxHierarchicalLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.true_labels = None

    def forward(self, inputs):
        if not isinstance(inputs, OrderedDict):
            raise ValueError("Use OrderedDict")
        
        input = torch.stack(list(inputs.values()))

        return MaxHierarchicalFunction.apply(input, self.true_labels)

class DisplayLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        print("display", type(input), input)
        return input