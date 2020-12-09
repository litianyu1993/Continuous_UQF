import numpy as np
import torch
from torch import nn
class Simple_Example(nn.Module):
    def __init__(self):
        super(Simple_Example, self).__init__()
    def forward(self, x):
        return x/((1+x)**3)

def MC_integral(model, **option):
    option_default = {
        'num_examples': 100000,
        'range': [0, 1],
        'input_dim': 1
    }
    option = {**option_default, **option}
    x = (option['range'][0] - option['range'][1]) * torch.rand(option['num_examples'], option['input_dim']) + option['range'][1]
    out = model(x)
    return torch.mean(out, dim=0) * (option['range'][1] - option['range'][0])

def simple_test():
    model = Simple_Example()
    option = {
        'range': [5, 20]
    }
    print(MC_integral(model, **option))

if __name__ == '__main__':
    simple_test()