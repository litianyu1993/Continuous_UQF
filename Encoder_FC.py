from torch import nn
import torch
import numpy as np
from torch.nn import functional as F
import copy
class Encoder_FC(nn.Module):
    def __init__(self, input_dim = None, encoded_dim = None, hidden_units = None, device = 'cpu', seed=0, init_encoder = None):
        '''
        input_dim: The dimension of the input data; int
        encoded_dim: The dimension of the output encoded features; int
        hidden_units: Number of hidden_units for each hidden layer; list of int
        device: Which device to use, 'cpu' or 'gpu'
        seed: Random seed
        freeze: True if want to freeze the weights, False otherwise
        '''
        super(Encoder_FC, self).__init__()
        np.random.seed(seed)
        self.device = device
        if init_encoder is not None:
            self.encoder = init_encoder
        else:
            self.encoder = self.init_FC_encoder(hidden_units, input_dim, encoded_dim)

    def init_FC_encoder(self, hidden_units, input_dim, encoded_dim):
        encoder = []
        hidden_units_ = copy.deepcopy(hidden_units)
        hidden_units_.insert(0, input_dim)
        hidden_units_.append(encoded_dim)
        for i in range(len(hidden_units_) - 1):

            dim0 = hidden_units_[i]
            dim1 = hidden_units_[i + 1]
            encoder.append(nn.Linear(dim0, dim1).to(self.device))
        return encoder

    def forward(self, x):
        x = x.float()
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            x = F.relu(x)
        return x
class Decoder_FC(nn.Module):
    def __init__(self, Encoder):
        super(Decoder_FC, self).__init__()
        self.Encoder = Encoder
        self.num_neurons = [self.Encoder.encoder[0].weight.shape[0]]
        for layer in self.Encoder.encoder:
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False
            self.num_neurons.append(layer.weight.shape[1])



if __name__ == '__main__':
    encoder_params = {'input_dim': 3,
                      'encoded_dim': 5,
                      'hidden_units': [10],
                      'device': 'cpu'}
    encoder = Encoder_FC(**encoder_params)
    x = np.random.normal(0, 1, [1000, 3])
    print(encoder(torch.from_numpy(x)).shape)