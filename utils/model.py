import torch
import torch.nn as nn
import numpy as np

from collections import OrderedDict
from utils.modules import Parallel, MaxLayer, MinLayer, EncodingLayer
from utils.custom_activations import StepActivation, hard_softmax

from typing import Optional

def stack_layers(layers_sizes: list[int], activation: nn.Module, last_activation: Optional[nn.Module] = None):
    layers = []
    for i, (s1, s2) in enumerate(zip(layers_sizes, layers_sizes[1:]), start=1):
        layers.append((f'l{i}', nn.Linear(s1, s2)))
        layers.append((f'a{i}', activation() if not last_activation or i < len(layers_sizes) - 1 else last_activation()))   
    return layers 

class BaseModel(nn.Module):
    def __init__(
            self,
            input_size: int,
            *hidden_layers_sizes: tuple[int],
            activation: nn.Module = nn.ReLU,
            last_activation: nn.Module = StepActivation,
        ) -> None:
        super().__init__()

        layers_sizes = [input_size] + list(hidden_layers_sizes) + [1]
        layers = stack_layers(layers_sizes, activation, last_activation)

        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x
    
    def get_module(self, name):
        return self.net

class ApproxModel(BaseModel):
    def __init__(self, input_size: int, *hidden_layers_sizes: tuple[int], **kwargs) -> None:
        super().__init__(input_size, *hidden_layers_sizes, activation=StepActivation, last_activation=None)

        kwargs_enc = {k: kwargs.get(k) for k in ['metadata', 'output_sizes', 'default_encoding_size']}
        self.enc = EncodingLayer(**kwargs_enc)

    def forward(self, x):
        x = self.enc(x)
        x = hard_softmax(x)
        x = self.net(x)

        return x

class RobddModel(nn.Module):
    def __init__(self, robdd, encoding_layer) -> None:
        super().__init__()

        self.robdd = robdd
        self.inputs_with_number = [(input, int(input.name[1:])) for input in robdd.inputs]
        self.func = lambda x: int(self.robdd.restrict({input: x[number] for input, number in self.inputs_with_number}))
        self.enc = encoding_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc(x)
        x = hard_softmax(x)
        x = np.apply_along_axis(self.func, 1, x.detach().numpy())
        x = torch.tensor(x).unsqueeze(1)

        return x
    
    def _update_func(self):
        self.func = lambda x: int(self.robdd.restrict({input: x[number] for input, number in self.inputs_with_number}))

    def compose(self, mapping) -> None:
        self.robdd = self.robdd.compose(mapping)
        self._update_func()

    def smoothing(self, inputs) -> None:
        self.robdd = self.robdd.smoothing(inputs)
        self._update_func()

    def consensus(self, inputs) -> None:
        self.robdd = self.robdd.consensus(inputs)
        self._update_func()

class MultiApprox(nn.Module):
    def __init__(self, aggregation: str, backward_method: str = "all") -> None:
        super().__init__()

        if aggregation == "max":
            AggLayer = MaxLayer(backward_method=backward_method)
        elif aggregation == "min":
            AggLayer = MinLayer(backward_method=backward_method)
        else:
            raise ValueError()

        self.net = nn.Sequential(OrderedDict([
            ('apx_modules', Parallel(OrderedDict([]))),
            ('logic', AggLayer),
        ]))

        self.n_apx = 0

    def forward(self, x):
        x = self.net(x)
        return x
    
    def add_apx(self, module):
        self.n_apx+=1
        self.net.apx_modules.add_module(f'apx{self.n_apx}', module)

    def named_apx(self):
        return self.net.apx_modules.named_children()

class MultiRobddModel(nn.Module):
    def __init__(self, robdd_dict, logic_layer) -> None:
        super().__init__()

        self.net = nn.Sequential(OrderedDict([
            ('robdd_modules', Parallel(robdd_dict)),
            ('logic', logic_layer),
        ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x

    def compose(self, mapping) -> None:
        for _, robdd in self.named_robdd():
            robdd.compose(mapping)

    def smoothing(self, inputs) -> None:
        for _, robdd in self.named_robdd():
            robdd.smoothing(inputs)

    def consensus(self, inputs) -> None:
        for _, robdd in self.named_robdd():
            robdd.consensus(inputs)

    def named_robdd(self):
        return self.net.robdd_modules.named_children()

class GlobalModel(nn.Module):
    def __init__(self, up: nn.Module, down: nn.Module, big: nn.Module) -> None:
        super().__init__()

        self.net = nn.Sequential(OrderedDict([
            ('models', Parallel(OrderedDict([
                ('up',   up),
                ('down', down),
                ('big',  big),
            ]))),  
        ]))

    def forward(self, x) -> torch.Tensor:
        up = self.forward_up_only(x)
        down = self.forward_down_only(x)
        net = self.forward_big_only(x)

        return torch.where(up > 0.5, up, torch.where(down < 0.5, down, net))
    
    def forward_up_only(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net.models.up(x)
        return x
    
    def forward_down_only(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net.models.down(x)
        return x
    
    def forward_big_only(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net.models.big(x)
        return x
    
    def named_models(self):
        return self.net.models.named_children()
    
    def get_module(self, name):
        return self.net.models.get_submodule(name)