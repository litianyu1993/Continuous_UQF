import numpy as np
import tensorly as tl
import gym
from torch import nn
import torch
from gradien_descent import train, validate, train_validate, fit
from torch import optim
from preprocess import get_kde, get_data_loaders
from Encoder import Encoder

class CWFA_AO(nn.Module):

    def __init__(self, action_encoder, obs_encoder, **option):
        super(CWFA_AO, self).__init__()
        option_default = {
            'random_init': True,
            'rank': 5,
            'device': 'cpu',
            'init_std': 0.1,
            'out_dim':1
        }
        option = {**option_default, **option}
        self.device = option['device']

        if not option['random_init']:
            alpha, A, Omega = option['alpha'], option['A'], option['Omega']
            if isinstance(alpha, np.ndarray):
                alpha = torch.from_numpy(alpha)
            if isinstance(A, np.ndarray):
                A = torch.from_numpy(A)
            if isinstance(Omega, np.ndarray):
                Omega = torch.from_numpy(Omega)

            self.alpha = alpha
            self.A = A
            self.Omega = Omega

            self.rank = self.A.shape[0]
            self.action_in_dim = action_encoder.out_dim
            self.obs_in_dim = obs_encoder.out_dim
            self.output_dim = Omega.shape[1] if len(Omega.shape) > 1 else 1
            if len(Omega.shape) == 1:
                self.Omega = torch.unsqueeze(self.Omega, 1)
        else:
            self.output_dim = option['out_dim']
            self.rank = option['rank']
            self.action_in_dim = action_encoder.out_dim
            self.obs_in_dim = obs_encoder.out_dim
            self.alpha = torch.nn.parameter.Parameter(torch.tensor(torch.normal(0, option['init_std'], [self.rank]),
                                                                  requires_grad=True)).to(self.device)
            self.A = torch.nn.parameter.Parameter(torch.tensor(torch.normal(0, option['init_std'], [self.rank, self.action_in_dim, self.obs_in_dim, self.rank]),
                                                                   requires_grad=True)).to(self.device)
            self.Omega = torch.nn.parameter.Parameter(torch.tensor(torch.normal(0, option['init_std'], [self.rank, self.output_dim]),
                                                                   requires_grad=True)).to(self.device)

        self.alpha.reshape([self.rank, ])
        self.action_encoder = action_encoder
        self.obs_encoder = obs_encoder

    def forward(self, x):
        actions = x[0]
        obss = x[1]
        if not torch.is_tensor(actions):
            actions = torch.from_numpy(actions).float()
        if not torch.is_tensor(obss):
            obss = torch.from_numpy(obss).float()
        actions = actions.float()
        obss = obss.float()

        assert obss.shape[1] == actions.shape[1] , print(
            'length mismatch between inputs')

        input_action_shape = actions.shape
        input_obs_shape = obss.shape
        action_r = actions.reshape(actions.shape[0] * actions.shape[1], -1)
        obs_r = obss.reshape(obss.shape[0] * obss.shape[1], -1)
        encoded_action = self.action_encoder(action_r)
        encoded_obs = self.obs_encoder(obs_r)

        act_seq = encoded_action.reshape(input_action_shape[0], input_action_shape[1], -1)
        obs_seq = encoded_obs.reshape(input_obs_shape[0], input_obs_shape[1], -1)

        #tmp = self.alpha.repeat(act_seq.shape[0], 1)
        tmp = torch.einsum('i, ijkl, nj, nk -> nl', self.alpha, self.A, act_seq[:, 0, :], obs_seq[:, 0, :])
        for i in range(1, actions.shape[1]):
            tmp = torch.einsum('ni, ijkl, nj, nk -> nl', tmp, self.A, act_seq[:, i, :], obs_seq[:, i, :])
        #print(tmp.shape, self.Omega.shape)
        tmp = torch.einsum('ni, im-> nm', tmp, self.Omega)
        return tmp.squeeze()

    def planning(self, x, next_A):
        actions = x[0]
        obss = x[1]
        if not torch.is_tensor(actions):
            actions = torch.from_numpy(actions).float()
        if not torch.is_tensor(obss):
            obss = torch.from_numpy(obss).float()
        actions = actions.float()
        obss = obss.float()

        assert obss.shape[1] == actions.shape[1], print(
            'length mismatch between inputs')

        input_action_shape = actions.shape
        input_obs_shape = obss.shape
        action_r = actions.reshape(actions.shape[0] * actions.shape[1], -1)
        obs_r = obss.reshape(obss.shape[0] * obss.shape[1], -1)
        encoded_action = self.action_encoder(action_r)
        encoded_obs = self.obs_encoder(obs_r)

        act_seq = encoded_action.reshape(input_action_shape[0], input_action_shape[1], -1)
        obs_seq = encoded_obs.reshape(input_obs_shape[0], input_obs_shape[1], -1)

        # tmp = self.alpha.repeat(act_seq.shape[0], 1)
        tmp = torch.einsum('i, ijkl, nj, nk -> nl', self.alpha, self.A, act_seq[:, 0, :], obs_seq[:, 0, :])
        for i in range(1, actions.shape[1]):
            tmp = torch.einsum('ni, ijkl, nj, nk -> nl', tmp, self.A, act_seq[:, i, :], obs_seq[:, i, :])
        print(tmp.shape, next_A.shape, self.Omega.shape)
        tmp = torch.einsum('ni, ijk, km', tmp, next_A, self.Omega)
        return tmp.squeeze()

    def _get_params(self):
        self.params = nn.ParameterList([])
        self.params.append(self.alpha)
        self.params.append(self.Omega)
        self.params.append(self.A)

        if not self.freeze_encoders:
            for i in range(len(self.action_encoder.encoder)):
                self.params.append(self.action_encoder.encoder[i].weight)
                self.params.append(self.action_encoder.encoder[i].bias)

            for i in range(len(self.obs_encoder.encoder)):
                self.params.append(self.obs_encoder.encoder[i].weight)
                self.params.append(self.obs_encoder.encoder[i].bias)
        return

    def fit(self, train_lambda, validate_lambda, scheduler, **option):

        default_option = {
            'verbose': True,
            'epochs': 1000
        }
        option = {**default_option, **option}
        train_validate(self, train_lambda, validate_lambda, scheduler, option)

    def build_true_Hankel_tensor(self,l):
        H = self.alpha
        for i in range(l):
            H = np.tensordot(H,self.A,[H.ndim-1,0])
        H = np.tensordot(H,self.Omega,[H.ndim-1,0])
        return H

def learn_CWFA(**option_list):
    kde_option = option_list['kde_option']
    train_gen_option = option_list['train_gen_option']
    validate_gen_option = option_list['validate_gen_option']
    action_encoder_option = option_list['action_encoder_option']
    obs_encoder_option = option_list['obs_encoder_option']
    cwfa_option = option_list['cwfa_option']
    fit_option = option_list['fit_option']

    gen_options = {
        'train_gen_option': train_gen_option,
        'validate_gen_option': validate_gen_option
    }

    if kde_option['use_kde']:
        kde = get_kde(**kde_option)
        gen_options['train_gen_option']['kde'] = kde
        gen_options['validate_gen_option']['kde'] = kde
    else:
        gen_options['train_gen_option']['kde'] = None
        gen_options['validate_gen_option']['kde'] = None


    train_dataset, train_loader, validate_dataset, validate_loader = get_data_loaders(batch_size=256, **gen_options)

    action_encoder_option['input_dim'] = train_dataset.action.shape[2]
    obs_encoder_option['input_dim'] = train_dataset.obs.shape[2]
    action_encoder = Encoder(**action_encoder_option)
    obs_encoder = Encoder(**obs_encoder_option)

    cwfa = CWFA_AO(action_encoder, obs_encoder, **cwfa_option)
    optimizer = optim.Adam(cwfa.parameters(), lr=fit_option['lr'], amsgrad=True)
    train_lambda = lambda model: train(model, cwfa_option['device'], train_loader, optimizer)
    validate_lambda = lambda model: validate(model, cwfa_option['device'], validate_loader)
    fit_option['optimizer'] = optimizer
    cwfa = fit(cwfa, train_lambda, validate_lambda, **fit_option)
    return cwfa


if __name__ == '__main__':
    window_size = 5
    option_lists = {
        'kde_option': {
            'env': gym.make('Pendulum-v0'),
            'num_trajs': 100,
            'max_episode_length': 10,
            'window_size': window_size,
            'load_kde': True,
            'use_kde': True
        },
        'train_gen_option': {
            'env': gym.make('Pendulum-v0'),
            'num_trajs': 1000,
            'max_episode_length': 100,
            'window_size': window_size},
        'validate_gen_option': {
            'env': gym.make('Pendulum-v0'),
            'num_trajs': 1000,
            'max_episode_length': 10,
            'window_size': window_size},

        'cwfa_option':{
            'random_init': True,
            'rank': 20,
            'device': 'cpu',
            'init_std': 0.1,
            'out_dim':1
        },
        'action_encoder_option': {
            'hidden_units': [10],
            'out_dim': 10,
            'final_activation': torch.nn.Tanh(),
            'inner_activation': torch.nn.LeakyReLU(inplace=False)
        },
        'obs_encoder_option': {
            'hidden_units': [10],
            'out_dim': 10,
            'final_activation': torch.nn.Tanh(),
            'inner_activation': torch.nn.LeakyReLU(inplace=False)
        },
        'fit_option': {
            'epochs': 1000,
            'verbose': True,
            'lr': 0.001,
            'step_size': 500,
            'gamma': 0.1
        }
    }
    learn_CWFA(**option_lists)