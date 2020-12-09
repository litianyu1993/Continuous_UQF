import numpy as np
import torch
from torch.nn import functional as F
from torch import optim
import copy
from torch import nn
import gym
import pickle
from torch.optim.lr_scheduler import StepLR
from Getting_Hankels import get_dataset
from Getting_Hankels import get_data_generator, get_all_kdes

class CWFA_AO(nn.Module):

    def __init__(self, **option):
        option_default = {
            'rank': 5,
            'dim_a': 3,
            'dim_o': 3,
            'encode_a_dim': 5,
            'encode_o_dim': 5,
            'out_dim': 1,
            'action_hidden': None,
            'obs_hidden': None,
            'alpha': None,
            'A': None,
            'Omega': None,
            'action_encoder': None,
            'obs_encoder': None,
            'freeze_encoder': False,
            'device': 'cpu'
        }
        option = {**option_default, **option}
        rank, dim_a, dim_o, encode_a_dim, encode_o_dim, out_dim, \
        action_hidden, obs_hidden, alpha, A, Omega, action_encoder, obs_encoder, freeze_encoder, device = \
        option['rank'], option['dim_a'], option['dim_o'], option['encode_a_dim'], option['encode_o_dim'], option['out_dim'], \
        option['action_hidden'], option['obs_hidden'], option['alpha'], option['A'], option['Omega'], option['action_encoder'], \
        option['obs_encoder'], option['freeze_encoder'], option['device']

        super(CWFA_AO, self).__init__()
        self.device = device
        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = torch.nn.parameter.Parameter(torch.from_numpy(np.random.normal(0, 0.1, [1, rank])).float(),requires_grad=True).to(device)
        if A is not None:
            self.A = A
        else:
            self.A = torch.nn.parameter.Parameter(torch.from_numpy(np.random.normal(0, 0.1, [rank, encode_a_dim, encode_o_dim, rank])).float(),requires_grad=True).to(device)

        if Omega is not None:
            self.Omega = Omega
        else:
            self.Omega =torch.nn.parameter.Parameter(torch.from_numpy(np.random.normal(0, 0.1, [rank, out_dim])).float(),requires_grad=True).to(device)
        if action_encoder is not None:
            self.action_encoder = action_encoder
        else:
            self.action_encoder = self._init_FC_encoder(action_hidden, dim_a, encode_a_dim, freeze_encoder)
        if obs_encoder is not None:
            self.obs_encoder =  obs_encoder
        else:
            self.obs_encoder = self._init_FC_encoder(obs_hidden, dim_o, encode_o_dim, freeze_encoder)
        self.leakyReLu = torch.nn.LeakyReLU(inplace=False)
        self._get_params()
        self.dim_a = dim_a
        self.dim_o = dim_o

    def _init_FC_encoder(self, hidden_units, input_dim, encoded_dim, freeze_encoder):
        encoder = []
        hidden_units_ = copy.deepcopy(hidden_units)
        hidden_units_.insert(0, input_dim)
        hidden_units_.append(encoded_dim)
        #self.bn = []
        for i in range(len(hidden_units_) - 1):

            dim0 = hidden_units_[i]
            dim1 = hidden_units_[i + 1]
            encoder.append(nn.Linear(dim0, dim1).to(self.device))
            if freeze_encoder:
                encoder[-1].weight.requires_grad = False
                encoder[-1].bias.requires_grad = False
        return encoder

    def encode_FC(self, encoder_FC, x):
        x = x.float()
        for i in range(len(encoder_FC)):
            x = encoder_FC[i](x)
            if i < len(encoder_FC) - 1:
                x = self.leakyReLu(x)
            else:
                x = F.sigmoid(x)
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
        encoded_action = self.encode_FC(self.action_encoder, action_r)
        encoded_obs = self.encode_FC(self.obs_encoder, obs_r)

        act_seq = encoded_action.reshape(input_action_shape[0], input_action_shape[1], -1)
        obs_seq = encoded_obs.reshape(input_obs_shape[0], input_obs_shape[1], -1)

        mps = []
        for t in range(act_seq.shape[1]):
            if t == 0:
                mps.append(torch.einsum('ip,pjkl ->ijkl', self.alpha.reshape(1, -1), self.A))
            elif t == act_seq.shape[1] - 1:
                mps.append(torch.einsum('pjkl, lm -> pjkm', self.A, self.Omega))
            else:
                mps.append(self.A)
        contracted = []
        for t in range(act_seq.shape[1]):
            tmp1 = act_seq[:, t, :]
            tmp2 = obs_seq[:, t, :]
            #print(mps[t].dtype, tmp1.dtype, tmp2.dtype)
            temp_core = torch.einsum('pjkl, ij, ik->pil', mps[t], tmp1, tmp2)
            contracted.append(temp_core)
        tmp_contract = contracted[0]
        for t in range(1, len(contracted)):
            tmp_contract = torch.einsum('pil, lik->pik', tmp_contract, contracted[t])
        return tmp_contract.squeeze()

    def _get_params(self):
        self.params = nn.ParameterList([])
        self.params.append(self.alpha)
        self.params.append(self.A)
        self.params.append(self.Omega)
        for i in range(len(self.action_encoder)):
            self.params.append(self.action_encoder[i].weight)
            self.params.append(self.action_encoder[i].bias)

        for i in range(len(self.obs_encoder)):
            self.params.append(self.obs_encoder[i].weight)
            self.params.append(self.obs_encoder[i].bias)
        return

def train(model, device, train_loader, optimizer):
    #model.train()
    error = []
    #optimizer.zero_grad()
    for batch_idx, (action, obs, target) in enumerate(train_loader):
        action, obs, target = action.to(device), obs.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(action, obs).to(device)
        loss = F.mse_loss(output, target)
        loss.backward()
        #print(torch.norm(model.A.grad))
        optimizer.step()
        error.append(loss.item())
    print(output[0], target[0])

    return sum(error) / len(error)


def vali(hankel, device, test_loader):
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

def tp (train, scheduler, options):
    train_loss_tt = []
    validate_loss_tt = []
    for epoch in range(1, options.epochs + 1):
        train()
        if options['verbose']:
            print()
        scheduler.step()

    return train_loss_tt, validate_loss_tt

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

def compute_MSE(pred, y):
    if torch.is_tensor(pred):
        pred = pred.detach().numpy()
    if torch.is_tensor(y):
        y = y.detach().numpy()
    pred = pred.reshape(pred.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    return np.mean((pred - y)**2)

if __name__ == '__main__':
    L = 2
    load_kde = True
    env_name = 'Pendulum-v0'
    lr = 0.001
    epochs = 10000
    generator_params = {'batch_size': 256,
                        'shuffle': True,
                        'num_workers': 0}
    kde_params = {'env': gym.make(env_name),
                  'num_trajs': 1000,
                  'max_episode_length': 10}
    sampling_params_train = {'env': gym.make(env_name),
                             'num_trajs': 1000,
                             'max_episode_length': 100}
    sampling_params_vali = {'env': gym.make(env_name),
                            'num_trajs': 100,
                            'max_episode_length': 100}
    if not load_kde:
        kde_l, kde_2l, kde_2l1 = get_all_kdes(kde_params, L)
        pickle.dump(kde_l, open('kde_l' + env_name, 'wb'))
        pickle.dump(kde_2l, open('kde_2l' + env_name, 'wb'))
        pickle.dump(kde_2l1, open('kde_2l1' + env_name, 'wb'))
    else:
        kde_l = pickle.load(open('kde_l' + env_name, 'rb'))
        kde_2l = pickle.load(open('kde_2l' + env_name, 'rb'))
        kde_2l1 = pickle.load(open('kde_2l1' + env_name, 'rb'))

    training_dataset= get_dataset(**sampling_params_train, kde=kde_2l1, window_size=2 * L + 1)
    training_generator= get_data_generator(dataset=training_dataset, **generator_params)
    vali_dataset= get_dataset(**sampling_params_vali, kde=kde_2l1, window_size=2 * L + 1)
    vali_generator= get_data_generator(dataset=vali_dataset, **generator_params)


    cwfa_params = {'rank': 100,
                   'dim_a': training_dataset.action.shape[-1],
                   'dim_o': training_dataset.obs.shape[-1],
                   'encode_a_dim': 5,
                   'encode_o_dim': 5,
                   'out_dim': training_dataset.y.ndim,
                   'action_hidden': [5],
                   'obs_hidden':[5],
                   'device': 'cpu'}
    scheduler_params = {
        'step_size': 500,
        'gamma': 1
    }
    cwfa = CWFA_AO(**cwfa_params)
    optimizer = optim.Adam(cwfa.parameters(), lr=lr, amsgrad=True)
    scheduler = StepLR(optimizer, **scheduler_params)
    cwfa, train_loss_tt, vali_loss_tt = Training_process(cwfa, training_generator, vali_generator, scheduler, optimizer, epochs, device='cpu',
                     verbose=True)
    from matplotlib import pyplot as plt
    plt.plot(np.log(train_loss_tt), label = 'Train Error')
    plt.plot(np.log(vali_loss_tt), label = 'Vali Error')
    plt.legend()
    plt.show()

    test_dataset = get_dataset(**sampling_params_vali, kde=kde_2l1, window_size=2 * L + 1)
    pred = cwfa(test_dataset.action, test_dataset.obs)
    print('Test MSE: ', compute_MSE(pred, test_dataset.y))
    plt.scatter(np.arange(len(pred)), pred.detach().numpy(), label = 'Prediction')
    plt.scatter(np.arange(len(test_dataset.y)), test_dataset.y.detach().numpy(), label = 'Target')
    plt.legend()
    plt.show()

    plt.scatter(np.arange(len(pred))[:20], pred.detach().numpy()[:20], label='Prediction')
    plt.scatter(np.arange(len(test_dataset.y))[:20], test_dataset.y.detach().numpy()[:20], label='Target')
    plt.legend()
    plt.show()


    pred = cwfa(training_dataset.action, training_dataset.obs)
    plt.scatter(np.arange(len(pred))[:20], pred.detach().numpy()[:20], label='Prediction')
    plt.scatter(np.arange(len(training_dataset.y))[:20], training_dataset.y.detach().numpy()[:20], label='Target')
    plt.legend()
    plt.show()



    np.savetxt('Train_error.csv', train_loss_tt, delimiter=',')
    np.savetxt('Vali_error.csv', vali_loss_tt, delimiter=',')