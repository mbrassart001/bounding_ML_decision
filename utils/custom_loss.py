import torch

class ExpLoss(torch.nn.Module): # https://www.desmos.com/calculator/1gtkyr1dlm
    def __init__(self, lower_bound=0, upper_bound=1, b=1.00, base_pow=None, reduction='mean'):
        super(ExpLoss, self).__init__()
        self.coef_div = upper_bound - lower_bound
        self.coef_mul = b
        self.coef_all = self.coef_mul/self.coef_div
        self.base_pow = base_pow

        self.mean_reduction = reduction=='mean'
        self.sum_reduction = reduction=='sum'

    def forward(self, input, label):
        if self.base_pow:
            loss = torch.abs(torch.pow(exponent=(input - label)*self.coef_all, self=self.base_pow) - 1)
        else:
            loss = torch.abs(torch.exp((input - label)*self.coef_all) - 1)

        if self.mean_reduction:
            loss = torch.mean(loss)
        elif self.sum_reduction:
            loss = torch.sum(loss)

        return loss

class AsymMSELoss(torch.nn.Module): # https://www.desmos.com/calculator/zmxcluqhkt
    def __init__(self, p=2):
        super(AsymMSELoss, self).__init__()
        self.p = p

    def forward(self, input, label):
        dif = label - input
        a = torch.square(dif)
        b = a*self.p
        loss = torch.where(dif < 0, b, a)
        loss = torch.mean(loss)
        return loss
    
class AsymBCELoss(torch.nn.Module):
    def __init__(self, p=2):
        super(AsymBCELoss, self).__init__()
        assert(p>0)
        self.p1 = p if p >= 1 else 1
        self.p2 = 1/p if p < 1 else 1

    def forward(self, input, label):
        loss1 = torch.log(1-input+1e-8)
        loss1 = loss1.clamp(min=-100)
        loss2 = torch.log(input+1e-8)
        loss2 = loss2.clamp(min=-100)
        loss = -(self.p1*(1-label)*loss1+self.p2*label*loss2)
        
        loss = torch.mean(loss)
        return loss