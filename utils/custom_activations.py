import torch
from torch import Tensor

class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:
        output = torch.where(input>=0, torch.tensor(1.0), torch.tensor(0.0))
        ctx.save_for_backward(input)
        return output
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        input, = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        return grad_input
    
class StepActivation(torch.nn.Module):
    def __init__(self, sigmoid_factor: int=1) -> None:
        super().__init__()
        self.sigmoid_factor = sigmoid_factor

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            return torch.sigmoid(self.sigmoid_factor*input)
        else:
            return StepFunction.apply(input)
        
class GumbelSoftmax(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return torch.nn.functional.gumbel_softmax(input, tau=1, hard=self.training)

def hard_softmax(input: Tensor, eps: float = 0, dim: int = -1):
    y_soft = input.softmax(dim)
    val_max = y_soft.max(dim, keepdim=True)[0]
    y_hard = torch.where(y_soft.detach() >= val_max * (1 - eps), 1.0, 0.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret

class HardSoftmax(torch.nn.Module):
    def __init__(self, eps: float=0.5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, input: Tensor) -> Tensor:
        return hard_softmax(input, self.eps)
    

class HistLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.uk = torch.nn.Parameter()# size ? init ?
        self.wk = torch.nn.Parameter()# size ? init ?

        # Conv1d(weights = 1, bias = -uk)
        # Absolute Value
        # Conv1d (weights = -1, bias = wk)
        # Exp
        # Threshold (relu at 1?)
        # ??? Global Average Pooling ???
    
    def forward(self, input):
        input = torch.nn.functional.conv1d(input, torch.ones_like(input), -self.uk)
        input = torch.abs(input)
        input = torch.nn.functional.conv1d(input, -torch.ones_like(input), self.wk)
        input = torch.exp(input)
        input = torch.relu(input - 1) + 1 # x if x > 1 else 1

        return input