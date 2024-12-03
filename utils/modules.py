import os
import sys
sys.path.append(os.path.dirname(__file__))

import torch
import more_torch_functions as mtf

from typing import Optional, Callable, Mapping
from torch import Tensor
from collections import OrderedDict
from numbers import Number      
from custom_activations import StepActivation

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
        input = torch.stack(list(inputs.values()))

        return MinFunction.apply(input, self.backward_func)
    
class DecisionLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        if not isinstance(inputs, Mapping):
            raise ValueError(f"Use mapping object as inputs instead of {type(inputs)}")
        
        high = inputs.get('high')
        low = inputs.get('low')
        big = inputs.get('big')

        output = torch.where(high > 0.5, high, torch.where(low < 0.5, low, big))
        return output

class ZeroOutput(torch.nn.Module):
    def forward(self, x):
        return torch.empty((x.size(0), 0))

class EncodingLayer(torch.nn.Module):
    default_encoding_size = 2

    def __init__(
            self,
            metadata: Mapping[str, tuple[int, int] | int],
            output_sizes: Mapping[str, int],
            default_encoding_size: int | None = None,
        ) -> None:
        super().__init__()

        self.hot_encode = False
        self.layer = torch.nn.ModuleList()
        self.slices_indices = []
        self.size_out = 0
        default_encoding_size = default_encoding_size or EncodingLayer.default_encoding_size
        
        for col, data in metadata.items():  
            if isinstance(data, tuple):
                start, end = data
                size_in = end - start + 1
                size_out = output_sizes.get(col, default_encoding_size)
                size_out = size_out if size_out > 0 else 0

                if size_out > 0:
                    self.layer.append(
                        torch.nn.Sequential(OrderedDict([
                            ('enc_linear', torch.nn.Linear(size_in, size_out)), 
                            ('enc_softmax', torch.nn.Softmax(dim=1)),
                        ]))
                    )
                else:
                    self.layer.append(ZeroOutput())

                self.slices_indices.append((start, end+1))
                self.size_out+=size_out
            else:
                size_out = output_sizes.get(col, 1)
                size_out = size_out if size_out > 0 else 0

                if size_out > 0:
                    self.layer.append(torch.nn.Identity())
                else:
                    self.layer.append(ZeroOutput())  

                self.slices_indices.append((data, data+1))
                self.size_out+=size_out

        # building a list containing for each modules : module, start index, end index
        self.module_iter = list(zip(self.layer, *zip(*self.slices_indices)))

    def forward(self, input: torch.Tensor):
        outputs = []
        for module, start, end in self.module_iter:
            input_slice = input[:, start:end]
            result = module(input_slice)
            if self.hot_encode and not isinstance(module, torch.nn.Identity):
                max_indices = torch.argmax(result, dim=1)
                one_hot_result = torch.zeros_like(result)
                one_hot_result.scatter_(1, max_indices.unsqueeze(1), 1)
                result = one_hot_result

            outputs.append(result)

        try:
            output = torch.cat(outputs, dim=1)
        except RuntimeError:
            print(input, outputs, self.module_iter, self.layer, self.slices_indices)
            raise Exception()
        return output
    
    @staticmethod
    def total_encoding_size(
        metadata: Mapping[str, tuple[int, int]|int],
        output_sizes: Mapping[str, int],
        default_encoding_size: int | None = None,
    ) -> int:
        default_encoding_size = default_encoding_size or EncodingLayer.default_encoding_size

        total_size = 0
        for c, x in metadata.items():
            if isinstance(x, tuple):
                added_size = output_sizes.get(c, default_encoding_size)
            else:
                added_size = output_sizes.get(c, 1)
            total_size += added_size
            output_sizes[c] = added_size
        return total_size