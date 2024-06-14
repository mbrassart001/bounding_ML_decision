import torch
import torch.nn as nn

from collections import OrderedDict

class Model(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()

        hl1,hl2,hl3 = 50,50,25

        self.nn = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(input_size, hl1)),
            ('a1', nn.ReLU()),
            ('l2', nn.Linear(hl1, hl2)),
            ('a2', nn.ReLU()),
            ('l3', nn.Linear(hl2, hl3)),
            ('a3', nn.ReLU()),
            ('l4', nn.Linear(hl3, 1)),
            ('a4', nn.Sigmoid()),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nn(x)
        return x
