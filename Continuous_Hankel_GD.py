from torch import nn
import torch
import numpy as np
from tensorly import random, tenalg
from torch.nn import functional as F
import tensorly as tl
from dataset import Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import copy
#torch.autograd.set_detect_anomaly(True)
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

# def MPS_contraction_single_sample(mps, x):
#     '''
#     :param mps:
#     :param x: x is of shape [1, traj_length, input_dim]
#     :return:
#     '''
#     #print(x.shape)
#     # for i in range(x.shape[0]):
#     #     print(i, x[i].shape, mps[i].shape)
#     contracted = [tenalg.contract(x[i], 0, mps[i], 1).squeeze() for i in range(x.shape[0])]
#     tmp = contracted[0]
#     for i in range(1, x.shape[0]):
#         tmp = tmp @ contracted[i]
#         #tmp = F.relu(tmp)
#     return tmp.squeeze()

# def MPS_contraction_samples(mps, x):
#     contracted = [MPS_contraction_single_sample(mps, data) for data in x]
#     return contracted

def MPO_contraction_single_sample(mpo, x1, x2):
    contracted_final = []
    for i in range(x1.shape[0]):

        tmp = tenalg.contract(mpo[i], 1, x1[i], 0)
        #print(x2.shape, tmp.shape, mpo[i].shape)
        tmp = tenalg.contract(x2[i], 0, tmp, 1)
        contracted_final.append(tmp)

    #contracted_final = [tenalg.contract(x2[i], 0, contracted[i], 1) for i in range(x2.shape[0])]
    tmp = contracted_final[0]
    for i in range(1, x1.shape[0]):
        tmp = tmp @ contracted_final[i]
    return tmp.squeeze()
def MPO_contraction_samples(mpo, x1, x2):
    #contracted = [MPO_contraction_single_sample(mpo, data1, data2) for data1, data2 in zip(x1.clone(), x2.clone())]
    assert  len(x1) == len(x2), print('action observation needs to have the same number')
    assert x1.shape[1] == x2.shape[1], print('action observation needs to have the same time steps')

    contracted = []
    for t in range(x1.shape[1]):
        tmp1 = x1[:,t, :]
        tmp2 = x2[:,t, :]
        temp_core = torch.einsum('pjkl, ij, ik->pil', mpo[t], tmp1, tmp2)
        contracted.append(temp_core)

    tmp_contract = contracted[0]
    for t in range(1, len(contracted)):
        tmp_contract = torch.einsum('pil, lik->pik', tmp_contract, contracted[t])
    return tmp_contract.squeeze()

def get_tensor_size(w):
    size = 1
    for i in range(len(w.shape)):
        size *= w.shape[i]
    return size

def init_weight(w):
    w = np.random.normal(0, 0.1, w.shape)/np.sqrt(get_tensor_size(w))
    return w

class Hankel(nn.Module):
    def __init__(self, action_dim, obs_dim, rank, encoded_dim_action, encoded_dim_obs, hidden_units_action, hidden_units_obs
                 , output_dim, max_length, seed = 0, device = 'cpu', rescale = False, freeze_encoder = False,
                 encoder_action = None, encoder_obs = None):
        super(Hankel, self).__init__()
        np.random.seed(seed)
        self.H = []
        self.encoded_action_dim =  encoded_dim_action
        self.encoded_obs_dim = encoded_dim_obs
        self.device = device
        #Construct encoder for action

        if encoder_action is not None:
            self.encoder_action  = encoder_action
        else:
            self.encoder_action = self.init_FC_encoder(hidden_units_action, action_dim, encoded_dim_action, freeze_encoder)
        if encoder_obs is not None:
            self.encoder_obs = encoder_obs
        else:
            self.encoder_obs = self.init_FC_encoder(hidden_units_obs, obs_dim, encoded_dim_obs, freeze_encoder)

        for k in range(max_length):

            dim_0 = rank
            dim_1 = self.encoded_action_dim
            dim_2 = self.encoded_obs_dim
            dim_3 = rank


            if k == 0:
                dim_0 = 1
            if k == max_length - 1:
                dim_3 = output_dim

            #H_tmp = tl.tensor(np.random.rand(dim_0, dim_1, dim_2, dim_3))*2 - 1
            H_tmp = tl.tensor(np.random.normal(0, 1, [dim_0, dim_1, dim_2, dim_3]))
            identity = np.zeros([dim_0, dim_1*dim_2, dim_3])
            for j in range(min(dim_0, dim_3)):
                identity[j, :, j] += 1.
            if rescale:
                #H_tmp = init_weight(H_tmp)
                H_tmp = init_weight(H_tmp) + identity.reshape([dim_0, dim_1, dim_2, dim_3])

            H_tmp = torch.nn.parameter.Parameter(torch.tensor(H_tmp.float(), requires_grad=True)).to(device)
            self.H.append(H_tmp)

    def init_FC_encoder(self, hidden_units, input_dim, encoded_dim, freeze_encoder):
        encoder = []
        hidden_units_ = copy.deepcopy(hidden_units)
        hidden_units_.insert(0, input_dim)
        hidden_units_.append(encoded_dim)
        for i in range(len(hidden_units_) - 1):

            dim0 = hidden_units_[i]
            dim1 = hidden_units_[i + 1]
            encoder.append(nn.Linear(dim0, dim1).to(self.device))
            #identity = np.zeros([dim0, dim1])
            # for j in range(min(dim0, dim1)):
            #     encoder[-1].weight.data[j, j] = encoder[-1].weight.data[j, j] + 1.
            #encoder[-1].weight = encoder[-1].weight + torch.from_numpy(identity).float()
            if freeze_encoder:
                encoder[-1].weight.requires_grad = False
                encoder[-1].bias.requires_grad = False
        return encoder

    def encode_FC(self, encoder_FC, x):
        x = x.float()
        for i in range(len(encoder_FC)):
            x = encoder_FC[i](x)
            #x = F.sigmoid(x)
        return x

    def forward(self, action, obs):
        if not torch.is_tensor(action):
            action = torch.from_numpy(action).float()
        if not torch.is_tensor(obs):
            obs = torch.from_numpy(obs).float()
        action = action.float()
        obs = obs.float()

        input_action_shape = action.shape
        input_obs_shape = obs.shape
        action_r = action.reshape(action.shape[0] * action.shape[1], -1)
        obs_r = obs.reshape(obs.shape[0] * obs.shape[1], -1)
        encoded_action = self.encode_FC(self.encoder_action, action_r)
        encoded_obs = self.encode_FC(self.encoder_obs, obs_r)

        encoded_action = encoded_action.reshape(input_action_shape[0], input_action_shape[1], -1)
        encoded_obs = encoded_obs.reshape(input_obs_shape[0], input_obs_shape[1], -1)
        #print(encoded_action.shape, encoded_obs.shape, action.shape, obs.shape)
        contracted = MPO_contraction_samples(self.H, encoded_action, encoded_obs)
        #print(contracted.shape)
        return contracted
    def get_params(self):
        params = nn.ParameterList([])
        for i in range(len(self.encoder_action)):
            params.append(self.encoder_action[i].weight)
            params.append(self.encoder_action[i].bias)

        for i in range(len(self.encoder_obs)):
            params.append(self.encoder_obs[i].weight)
            params.append(self.encoder_obs[i].bias)

        for i in range(len(self.H)):
            params.append(self.H[i])

        return params


# class Hankel(nn.Module):
#     def __init__(self, rank = 5, input_dim = 4, encoded_dim = 10, encoder_hidden = 10,
#                  output_dim = 1, max_length = 6, seed=0, device='cpu', if_rescale_weights = True, encoder = None):
#         super(Hankel, self).__init__()
#         np.random.seed(seed)
#         self.H = []
#         for i in range(max_length):
#
#             dim_0 = rank
#             dim_2 = rank
#             dim_1 = encoded_dim
#
#             if i == 0:
#                 dim_0 = 1
#             if i == max_length - 1:
#                 dim_2 = output_dim
#
#             H_tmp = tl.tensor(np.random.rand(dim_0, dim_1, dim_2))
#             if if_rescale_weights:
#                 bound = 1. / np.sqrt(dim_0 * dim_1 * dim_2)
#                 #bound =  1.
#                 H_tmp = H_tmp * 2 * bound - bound
#
#             H_tmp = tl.tensor(H_tmp, device=device, requires_grad=True)
#             self.H.append(H_tmp)
#
#         # Feature mapping of action observation vector
#         if encoder is None:
#             self.encoder1 = nn.Linear(input_dim, encoder_hidden).to(device)
#             self.encoder2 = nn.Linear(encoder_hidden, encoded_dim).to(device)
#         else:
#             self.encoder1 = encoder[0]
#             self.encoder2 = encoder[1]
#             self.encoder1.weight.requires_grad = False
#             self.encoder1.bias.requires_grad = False
#             self.encoder2.weight.requires_grad = False
#             self.encoder2.bias.requires_grad = False
#         # for i in range(len(self.H)):
#         #     print('init', self.H[i].shape)
#         # Some class properties that might come handy
#         self.output_dim = output_dim
#         self.input_dim = input_dim
#         self.device = device
#         self.encoded_dim = encoded_dim
#
#
#     def forward(self, x):
#         '''
#         :param x: of shape [num_trajectories, length_traj, input_dim]
#         :return: forwarded results
#         '''
#         # Encode the features
#         input_shape = x.shape
#         x = x.reshape(x.shape[0] * x.shape[1], -1)
#         x = tl.tensor(x)
#         x = x.float()
#         x = self.encoder1(x)
#         x = F.relu(x)
#         x = self.encoder2(x)
#         encoded_x = F.relu(x)
#         # Now x is of shape [n * length_traj, input_dim]
#         encoded_x = encoded_x.reshape(input_shape[0], input_shape[1], -1)
#         # Now x is of shape [n, length_traj, input_dim]
#         # ones = tl.tensor(np.ones((encoded_x.shape[0], encoded_x.shape[1], 1)))
#         # encoded_x = torch.cat((encoded_x, ones), dim = 2)
#         # for i in range(len(self.H)):
#         #     print(i, self.H[i].shape)
#         contracted_x = MPS_contraction_samples(self.H, encoded_x)
#         #contracted_x = [MPS_contraction_single_sample(self.H, e_x) for e_x in encoded_x]
#         return torch.stack(contracted_x)


# The train and test function of the Tensorized Network
def train(model, device, train_loader, optimizer):
    model.train()
    error = []
    #optimizer.zero_grad()
    for batch_idx, (action, obs, target) in enumerate(train_loader):
        action, obs, target = action.to(device), obs.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(action, obs).to(device)
        loss = F.mse_loss(output, target)
        loss.backward(retain_graph=True)
        #print(model.H[0].grad)
        optimizer.step()
        error.append(loss.item())
    print(output[0], target[0])

    return sum(error) / len(error)


def test(hankel, device, test_loader):
    hankel.eval()
    test_loss = 0
    with torch.no_grad():
        for action, obs, target in test_loader:
            action, obs, target = action.to(device), obs.to(device), target.to(device)
            output = hankel(action, obs).to(device)
            test_loss += F.mse_loss(output, target).item()  # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)

    # print('Test set: Average loss: {:.4f}'.format(
    #     test_loss))
    return test_loss
#
def Training_process(hankel, training_generator, validation_generator, scheduler, optimizer, epochs, device = 'cpu', verbose = False):
    # params = [hankel.encoder1.bias, hankel.encoder1.weight,
    #                         hankel.encoder2.bias, hankel.encoder2.weight]
    # for i in range(len(hankel.H)):
    #     params.append(hankel.H[i])
    #
    # # params.append(hankel.encoder1.weight)
    # # params.append(hankel.encoder2.weight)
    #
    #
    # optimizer = optim.Adam(params, lr=lr, amsgrad  = True)
    # #for i in range(len(hankel.H)):
    # #    print('train hankel', hankel.H[i].shape)
    # # scheduler for automatic decaying the learning rate
    # scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_loss_tt = []
    vali_loss_tt = []

    # # Training
    for epoch in range(1, epochs + 1):
        train_loss_tt.append(train(hankel, device, training_generator, optimizer))
        vali_loss_tt.append(test(hankel, device, validation_generator))
        if verbose:
            print('\nTrain Epoch: ' + str(epoch) + ' Training Loss: ' + str(train_loss_tt[-1])+' Validation Loss: '+str(vali_loss_tt[-1]))
        scheduler.step()
    return hankel, train_loss_tt, vali_loss_tt



if __name__ == '__main__':
    # Parameters
    generator_params = {'batch_size': 512,
                        'shuffle': True,
                        'num_workers': 0}
    lr = 0.01
    amsgrad = True
    # lr = 0.01
    step_size = 100
    gamma = 1
    epochs = 500

    device = 'cpu'
    action_dim = 2
    obs_dim = 4
    encoded_dim_action = 10
    encoded_dim_obs = 10
    hidden_units_action = [10, 10]
    hidden_units_obs = [3,3]
    output_dim = 2
    rank = 5
    max_length = 4
    num_train = 2000
    num_vali = 1000


    hankel = Hankel(action_dim, obs_dim, rank, encoded_dim_action, encoded_dim_obs, hidden_units_action, hidden_units_obs
                 , output_dim, max_length, seed = 0, device = 'cpu', rescale = False, freeze_encoder = False)

    actions = np.random.rand(num_train, max_length, action_dim)
    obs = np.random.rand(num_train, max_length, obs_dim)
    y = hankel(actions, obs)
    training = Dataset(data=[actions, obs, y])

    # Generators

    training_generator = torch.utils.data.DataLoader(training, **generator_params)

    actions = np.random.rand(num_vali, max_length, action_dim)
    obs = np.random.rand(num_vali, max_length, obs_dim)
    y = hankel(actions, obs)
    validation = Dataset(data=[actions, obs, y])
    validation_generator = torch.utils.data.DataLoader(validation, **generator_params)

    hankel = Hankel(action_dim, obs_dim, rank, encoded_dim_action, encoded_dim_obs, hidden_units_action, hidden_units_obs
                 , output_dim, max_length, seed = 0, device = 'cpu', rescale = True, freeze_encoder = False)

    params = []
    for i in range(len(hankel.encoder_action)):
        params.append(hankel.encoder_action[i].weight)
        params.append(hankel.encoder_action[i].bias)

    for i in range(len(hankel.encoder_obs)):
        params.append(hankel.encoder_obs[i].weight)
        params.append(hankel.encoder_obs[i].bias)

    for i in range(len(hankel.H)):
        params.append(hankel.H[i])

    optimizer = optim.Adamax(params, lr = lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    Training_process(hankel, training_generator, validation_generator, scheduler, optimizer, device='cpu',
                     verbose=True, epochs = epochs)



