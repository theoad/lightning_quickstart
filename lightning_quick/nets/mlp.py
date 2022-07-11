from typing import Optional, Union
import torch
import torch.nn as nn
from torch.types import _size, _int
from numpy import prod


class MLP(nn.Sequential):
    def __init__(self,
                 in_size: Union[_int, _size],
                 out_size: Union[_int, _size],
                 hidden_size: _int,
                 num_layers: int,
                 activation: Optional[nn.Module] = nn.ReLU,
                 bias: Optional[bool] = True,
                 **unused_kwargs):

        self.in_size = torch.Size((in_size,)) if isinstance(in_size, int) else in_size
        self.out_size = torch.Size((out_size,)) if isinstance(out_size, int) else out_size
        self.in_flatten_size = prod(self.in_size).item()
        self.out_flatten_size = prod(self.out_size).item()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.bias = bias

        super().__init__(
            nn.Flatten(),
            nn.Linear(self.in_flatten_size, self.hidden_size, bias=self.bias), self.activation(),
            *([nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias), self.activation()] * (num_layers-2)),
            nn.Linear(self.hidden_size, self.out_flatten_size, bias=self.bias), self.activation(),
            nn.Unflatten(1, self.out_size)
        )
