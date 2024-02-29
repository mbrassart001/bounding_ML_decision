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
    def forward(ctx, input):
        inputs = input.unbind()

        # maximum ? addition ? some sort of mean ? mix of the methods ?
        output = mtf.maximum(inputs)

        ctx.save_for_backward(input, output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        return MaxFunction.backward_blame_all(input, output, grad_output)
    
    # /!\ improve needed /!\
    @staticmethod
    def backward_blame_max(input, output, grad_output):
        return torch.where(input >= output, grad_output, 0)
    
    @staticmethod
    def backward_blame_all(input, output, grad_output):
        return grad_output.repeat((input.size(0),1,1))

class MaxLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        if not isinstance(inputs, OrderedDict):
            raise ValueError("Use OrderedDict")
        
        input = torch.stack(list(inputs.values()))

        return MaxFunction.apply(input)
    
class DisplayLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        print("display", type(input), input)
        return input