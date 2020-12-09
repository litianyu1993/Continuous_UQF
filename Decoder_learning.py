import numpy as np
import torch
from torch.nn import functional as F
from torch import optim
import copy
from torch import nn
import gym
import pickle
from torch.optim.lr_scheduler import StepLR
from Learning_CWFA_AO import CWFA_AO, train, vali, Training_process

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data = None, data_path = None):
        'Initialization'
        if data is not None:
            if len(data) != 2:
                raise Exception("data need to be a list of 2, first is input, second is output")
            else:
                self.x = data[0]
                self.y = data[1]
        elif data_path is not None:
            if len(data_path) != 2:
                raise Exception("datapath need to be a list of 2, first is input, second is output")
            else:
                self.x = np.genfromtxt(data_path[0], delimiter=',')
                self.y = np.genfromtxt(data_path[1], delimiter=',')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

    def __getitem__(self, index):
        'Generates one sample of data'

        x = self.x[index]
        y = self.y[index]
        return x, y
class linear(nn.Module):
    def __init__(self, input_dim, output_dim, device = 'cpu'):
        super(linear, self).__init__()
        self.device = device
        self.fc = nn.Linear(input_dim, output_dim).to(self.device)
    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.float()
        return self.fc(x)

class Decoder_FC(nn.Module):
    def __init__(self, encoder_FC, device = 'cpu'):
        super(Decoder_FC, self).__init__()
        self.device = device
        self.encoder = encoder_FC
        self.Decoder = []
        for i in range(len(self.encoder)):
            self.Decoder.append(nn.Linear(self.encoder[-i-1].weight.shape[0], self.encoder[-i-1].weight.shape[1]).to(self.device))

        self.input_dim = self.encoder[-1].weight.shape[1]
        self._get_params()
    def _rescale(self, w):
        size = 1
        for i in w.shape:
            size *= i
        return w/(torch.sqrt(size))
    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.float()
        #x = F.sigmoid(self.Decoder(x))
        for i in range(len(self.Decoder)):
            #print(x.shape, self.Decoder[i].weight.shape)
            x = self.Decoder[i](x)
            if not i == len(self.Decoder) - 1:
                x = F.sigmoid(x)
        return x
    def _get_params(self):
        self.params = nn.ParameterList([])
        for i in range(len(self.Decoder)):
            self.params.append(self.Decoder[i].weight)
            self.params.append(self.Decoder[i].bias)

def encoding(x, encoder):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    x = x.float()
    for i in range(len(encoder)):
        x = encoder[i](x)
        x = F.sigmoid(x)
    return x

def train(model, device, train_loader, optimizer):
    model.train()
    error = []
    #optimizer.zero_grad()
    for batch_idx, (x, target) in enumerate(train_loader):
        x, target = x, target.to(device)
        optimizer.zero_grad()
        output = model(x).to(device)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        error.append(loss.item())
    print(output[0], target[0])

    return sum(error) / len(error)


def vali(model, device, test_loader):
    test_loss = 0
    with torch.no_grad():
        for x, target in test_loader:
            x, target = x.to(device), target.to(device)
            output = model(x).to(device)
            test_loss += F.mse_loss(output, target).item()  # sum up batch loss

    test_loss /= len(test_loader)

    # print('Test set: Average loss: {:.4f}'.format(
    #     test_loss))
    return test_loss

def Training_process(model, training_generator, validation_generator, scheduler, optimizer, epochs, device = 'cpu', verbose = False):
    train_loss_tt = []
    vali_loss_tt = []

    # # Training
    for epoch in range(1, epochs + 1):
        train_loss_tt.append(train(model, device, training_generator, optimizer))
        vali_loss_tt.append(vali(model, device, validation_generator))
        if verbose:
            print('\nTrain Epoch: ' + str(epoch) + ' Training Loss: ' + str(train_loss_tt[-1])+' Validation Loss: '+str(vali_loss_tt[-1]))
        scheduler.step()
    return model, train_loss_tt, vali_loss_tt


if __name__ == '__main__':
    L = 2
    load_kde = True
    env_name = 'Pendulum-v0'
    lr = 0.01
    epochs = 2000
    generator_params = {'batch_size': 1024,
                        'shuffle': True,
                        'num_workers': 0}
    kde_params = {'env': gym.make(env_name),
                  'num_trajs': 1000,
                  'max_episode_length': 10}

    cwfa_params = {'rank': 30,
                   'dim_a': 3,
                   'dim_o': 4,
                   'encode_a_dim': 10,
                   'encode_o_dim': 10,
                   'out_dim': 1,
                   'action_hidden': [10],
                   'obs_hidden':[10],
                   'device': 'cpu'}
    scheduler_params = {
        'step_size': 500,
        'gamma': 1
    }
    N = 1000
    cwfa = CWFA_AO(**cwfa_params)
    decoder = Decoder_FC(cwfa.action_encoder)
    y = torch.tensor(np.random.normal(0, 1, [N, 3])).float()
    x = encoding(y, cwfa.action_encoder).detach()
    #x = y + 1
    dataset = Dataset(data = [x, y])
    train_generator = torch.utils.data.DataLoader(dataset, **generator_params)

    N = 1000
    y = torch.tensor(np.random.normal(0, 1, [N, 3])).float()
    x = encoding(y, cwfa.action_encoder).detach()
    dataset = Dataset(data=[x, y])
    vali_generator = torch.utils.data.DataLoader(dataset, **generator_params)





    optimizer = optim.Adam(decoder.parameters(), lr=lr, amsgrad=True)
    scheduler = StepLR(optimizer, **scheduler_params)
    decoder, train_loss_tt, vali_loss_tt = Training_process(decoder, train_generator, vali_generator, scheduler, optimizer,
                                                         epochs, device='cpu',
                                                         verbose=True)

    N = 1000
    y = torch.tensor(np.random.normal(0, 1, [N, 3])).float()
    x = encoding(y, cwfa.action_encoder).detach()
    print(x.shape, y.shape)
    from sklearn.linear_model import LinearRegression

    reg = LinearRegression().fit(x.numpy(), y.numpy())
    pred = reg.predict(x).reshape(y.numpy().shape)
    print(np.mean((pred - y.numpy())**2))

