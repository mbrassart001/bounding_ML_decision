import os
import sys
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn

from collections import OrderedDict
from modules import Parallel, MaxLayer

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

class MultiApprox(nn.Module):
    def __init__(self, max_weighting="all") -> None:
        super().__init__()

        self.n_apx = 0
        self.net = nn.Sequential(OrderedDict([
            ('nets', Parallel(OrderedDict([
                ('approx', nn.Sequential(OrderedDict([
                    ('preprocess', nn.Sequential()),
                    ('apx_modules', Parallel(OrderedDict([]))),
                    ('logic', MaxLayer(backward_method=max_weighting)),
                ]))),
            ]))),
            ('logic', MaxLayer(backward_method=max_weighting)),
        ]))

    def forward(self, input):
        return self.net(input)
    
    def forward_apx_only(self, input: torch.Tensor) -> torch.Tensor:
        return self.net.nets.approx(input)
    
    def forward_nn_only(self, input: torch.Tensor) -> torch.Tensor:
        return self.net.nets.get_submodule("nn")(input)
    
    def forward_preprocess(self, input: torch.Tensor) -> torch.Tensor:
        return self.net.nets.approx.preprocess(input)

    def add_apx(self, module: nn.Module) -> None:
        self.n_apx += 1
        self.net.nets.approx.apx_modules.add_module(f'apx{self.n_apx}', module)
    
    def add_nn(self, module: nn.Module) -> None:
        self.net.nets.add_module(f'nn', module)

    def add_apx_preprocess(self, name, module: nn.Module) -> None:
        self.net.nets.approx.preprocess.add_module(name, module)
    
    def named_apx(self):
        return self.net.nets.approx.apx_modules.named_children()