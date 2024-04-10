import torch

from torch import Tensor

def loss_reduction(func):
    def wrapper(obj, *args, **kwargs):
        loss = func(obj, *args, **kwargs)

        if obj.reduction == "mean":
            loss = torch.mean(loss)
        elif obj.reduction == "sum":
            loss = torch.sum(loss)
        
        return loss
    return wrapper

class ExpLoss(torch.nn.modules.loss._Loss): # https://www.desmos.com/calculator/1gtkyr1dlm
    def __init__(
        self,
        lower_bound: int | float = 0,
        upper_bound: int | float = 1,
        b: int | float = 1.00, 
        base_pow: int | float = None,
        size_average = None,
        reduce = None,
        reduction: str='mean',
    ) -> None:
        super(ExpLoss, self).__init__(size_average, reduce, reduction)
        self.coef_div = upper_bound - lower_bound
        self.coef_mul = b
        self.coef_all = self.coef_mul/self.coef_div
        self.base_pow = base_pow

    @loss_reduction
    def forward(self, input: Tensor, label: Tensor) -> Tensor:
        if self.base_pow:
            loss = torch.abs(torch.pow(exponent=(input - label)*self.coef_all, self=self.base_pow) - 1)
        else:
            loss = torch.abs(torch.exp((input - label)*self.coef_all) - 1)

        return loss

class AsymMSELoss(torch.nn.modules.loss._Loss): # https://www.desmos.com/calculator/zmxcluqhkt
    def __init__(
        self,
        p: int | float = 2,
        size_average = None,
        reduce = None,
        reduction: str='mean',
    ) -> None:
        super(AsymMSELoss, self).__init__(size_average, reduce, reduction)
        self.p = p

    @loss_reduction
    def forward(self, input: Tensor, label: Tensor) -> Tensor:
        dif = label - input
        a = torch.square(dif)
        b = a*self.p
        loss = torch.where(dif < 0, b, a)

        return loss
    
class AsymBCELoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        p: int | float = 2,
        size_average = None,
        reduce = None,
        reduction: str='mean',
    ) -> None:
        super(AsymBCELoss, self).__init__(size_average, reduce, reduction)
        if p <= 0:
            raise ValueError(f"Argument p must be strictly greater than 0 ({p = } <= 0)")
        self.p1 = p if p >= 1 else 1
        self.p2 = 1/p if p < 1 else 1

    @loss_reduction
    def forward(self, input: Tensor, label: Tensor) -> Tensor:
        loss1 = torch.log(1-input+1e-8)
        loss1 = loss1.clamp(min=-100)
        loss2 = torch.log(input+1e-8)
        loss2 = loss2.clamp(min=-100)
        loss = -(self.p1*(1-label)*loss1+self.p2*label*loss2)
        
        return loss