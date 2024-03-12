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
    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            return torch.sigmoid(input)
        else:
            return StepFunction.apply(input)