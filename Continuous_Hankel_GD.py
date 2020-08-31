from torch import nn
import torch
import numpy as np
from tensorly import random, tenalg
from torch.nn import functional as F
import tensorly as tl
from dataset import Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from matplotlib import pyplot as plt
tl.set_backend('pytorch')
import itertools
# def MPS_contraction(mps, x, return_vec = True):
#     '''
#     mps: a list of MPS cores (tensorly format), the i-th core is of shape (r_{i-1}, d_i, r_i),
#        for the first core r_0 =1, for the last core r_m = 1.
#     x: input tensor to be contracted (tensorly format), should be of size (n, d_1, d_2, ..., d_m).
#     return: a tensor of shape (n, c_1, c_2, ..., c_m)
#     '''
#     assert len(mps) == x.ndim - 1, "dimension mismatch between MPO and input tensor x"
#     for i in range(len(mps)):
#         assert mps[i].shape[1] == x.shape[i+1], str(i)+"th core does not match the input dimension, note the thrid dimension of the mpo core should be the same of the corresponding input tensor dimension"
#
#     output_tmp = tenalg.contract(x, [1], mps[0], [1])
#     for i in range(1, x.ndim - 1):
#         output_tmp = tenalg.contract(output_tmp, [1, output_tmp.ndim - 1], mps[i], [1, 0])
#     return output_tmp.squeeze()

def MPS_contraction_single_sample(mps, x):
    '''
    :param mps:
    :param x: x is of shape [1, traj_length, input_dim]
    :return:
    '''
    #print(x.shape)
    # for i in range(x.shape[0]):
    #     print(i, x[i].shape, mps[i].shape)
    contracted = [tenalg.contract(x[i], 0, mps[i], 1).squeeze() for i in range(x.shape[0])]
    tmp = contracted[0]
    for i in range(1, x.shape[0]):
        tmp = tmp @ contracted[i]
    return tmp.squeeze()

def MPS_contraction_samples(mps, x):
    # for data in x:
    #     print(data.shape)
    # for i in range(len(mps)):
    #     print(mps[i].shape)
    contracted = [MPS_contraction_single_sample(mps, data) for data in x]
    return contracted

class Hankel(nn.Module):
    def __init__(self, rank = 5, input_dim = 4, encoded_dim = 10, encoder_hidden = 10,
                 output_dim = 1, max_length = 6, seed=0, device='cpu', if_rescale_weights = True, encoder = None):
        super(Hankel, self).__init__()
        np.random.seed(seed)
        self.H = []
        for i in range(max_length):

            dim_0 = rank
            dim_2 = rank
            dim_1 = encoded_dim

            if i == 0:
                dim_0 = 1
            if i == max_length - 1:
                dim_2 = output_dim

            H_tmp = tl.tensor(np.random.rand(dim_0, dim_1, dim_2))
            if if_rescale_weights:
                bound = 1. / np.sqrt(dim_0 * dim_1 * dim_2)
                #bound =  1.
                H_tmp = H_tmp * 2 * bound - bound

            H_tmp = tl.tensor(H_tmp, device=device, requires_grad=True)
            self.H.append(H_tmp)

        # Feature mapping of action observation vector
        if encoder is None:
            self.encoder1 = nn.Linear(input_dim, encoder_hidden).to(device)
            self.encoder2 = nn.Linear(encoder_hidden, encoded_dim).to(device)
        else:
            self.encoder1 = encoder[0]
            self.encoder2 = encoder[1]
            self.encoder1.weight.requires_grad = False
            self.encoder1.bias.requires_grad = False
            self.encoder2.weight.requires_grad = False
            self.encoder2.bias.requires_grad = False
        # for i in range(len(self.H)):
        #     print('init', self.H[i].shape)
        # Some class properties that might come handy
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.device = device
        self.encoded_dim = encoded_dim


    def forward(self, x):
        '''
        :param x: of shape [num_trajectories, length_traj, input_dim]
        :return: forwarded results
        '''
        # Encode the features
        input_shape = x.shape
        x = x.reshape(x.shape[0] * x.shape[1], -1)
        x = tl.tensor(x)
        x = x.float()
        x = self.encoder1(x)
        x = F.relu(x)
        x = self.encoder2(x)
        encoded_x = F.relu(x)
        # Now x is of shape [n * length_traj, input_dim]
        encoded_x = encoded_x.reshape(input_shape[0], input_shape[1], -1)
        # Now x is of shape [n, length_traj, input_dim]
        # ones = tl.tensor(np.ones((encoded_x.shape[0], encoded_x.shape[1], 1)))
        # encoded_x = torch.cat((encoded_x, ones), dim = 2)
        # for i in range(len(self.H)):
        #     print(i, self.H[i].shape)
        contracted_x = MPS_contraction_samples(self.H, encoded_x)
        return torch.stack(contracted_x)


# The train and test function of the Tensorized Network
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    error = []
    #optimizer.zero_grad()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # print(data.shape)
        # print(target.shape)
        # print(data)
        # print(target)
        optimizer.zero_grad()
        output = model(data).to(device)
        # print(output[:5])
        loss = F.mse_loss(output, target)
        loss.backward(retain_graph=True)
        optimizer.step()
        # if batch_idx % log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #                100. * batch_idx / len(train_loader), loss.item()))

        error.append(loss.item())
    # print(output[:5], target[:5])
    # print(model.H[0])
    for i in range(len(model.H)):
        print(model.H[i].grad)
    print(model.encoder1.weight.grad)
    print(model.encoder2.weight.grad)
    print(loss)
    print(model.encoder1.bias.grad)
    print(model.encoder2.bias.grad)

    print('\nTrain Epoch: ' + str(epoch) + ' Training Loss: ' + str(sum(error) / len(error)))
    return sum(error) / len(error)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).to(device)
            test_loss += F.mse_loss(output, target).item()  # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)

    print('Test set: Average loss: {:.4f}'.format(
        test_loss))
    return test_loss

def Training_process(hankel, training_generator, validation_generator, lr, step_size, gamma, epochs, device = 'cpu'):
    params = [hankel.encoder1.bias, hankel.encoder1.weight,
                            hankel.encoder2.bias, hankel.encoder2.weight]
    for i in range(len(hankel.H)):
        params.append(hankel.H[i])

    # params.append(hankel.encoder1.weight)
    # params.append(hankel.encoder2.weight)


    optimizer = optim.SGD(params, lr=lr)
    #for i in range(len(hankel.H)):
    #    print('train hankel', hankel.H[i].shape)
    # scheduler for automatic decaying the learning rate
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_loss_tt = []
    vali_loss_tt = []

    # # Training
    for epoch in range(1, epochs + 1):
        train_loss_tt.append(train(hankel, device, training_generator, optimizer, epoch))
        vali_loss_tt.append(test(hankel, device, validation_generator))
        scheduler.step()
    return hankel, train_loss_tt, vali_loss_tt

if __name__ == '__main__':
    device = 'cpu'
    hankel = Hankel(rank = 5, input_dim = 4, encoded_dim = 10, encoder_hidden = 10,
                 output_dim = 2, max_length = 6, seed=0, device=device, if_rescale_weights= False)
    np.random.seed(0)


    # Parameters
    generator_params = {'batch_size': 512,
              'shuffle': True,
              'num_workers': 0}
    lr = 0.01
    step_size = 100
    gamma = 1
    epochs = 500

    x = np.random.rand(2000, 6, 4)
    y = hankel(x)
    training = Dataset(data=[x, y])

    # Generators
    training_generator = torch.utils.data.DataLoader(training, **generator_params)

    x = np.random.rand(1000, 6, 4)
    y = hankel(x)
    validation = Dataset(data=[x, y])
    validation_generator = torch.utils.data.DataLoader(validation, **generator_params)

    hankel = Hankel(rank=3, input_dim=4, encoded_dim=3, encoder_hidden=10,
                    output_dim=2, max_length=6, seed=0, device=device, if_rescale_weights=True)

    hankel, train_error, vali_error = Training_process(hankel = hankel, training_generator = training_generator
                                                       , validation_generator = validation_generator,
                                                       lr = lr, step_size = step_size, gamma = gamma, epochs = epochs)
    print(hankel(x[:5]))
    print(y[:5])
    plt.plot(train_error)
    plt.plot(vali_error)
    plt.show()


