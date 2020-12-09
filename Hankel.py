from torch import nn
import torch
import pickle
from preprocess import get_dataset, get_data_generator, construct_KDE
from Dataset import Dataset_Action_Obs_Y as Dataset
import gym
from gradien_descent import train, validate, train_validate
from torch import optim
from torch.optim.lr_scheduler import StepLR
from Encoder import Encoder
class Hankel(nn.Module):
    def __init__(self, action_encoder, obs_encoder, **option):
        super(Hankel, self).__init__()
        option_default = {
            'rank': 5,
            'out_dim': 1,
            'max_length': 5,
            'device': 'cpu',
            'freeze_encoder': False,
            'mps': None,
            'init_std': 0.1
        }
        option = {**option_default, **option}
        self.action_encoder = action_encoder
        self.obs_encoder = obs_encoder
        self.length = option['max_length']
        self.rank = option['rank']
        self.encoded_a_dim = action_encoder.out_dim
        self.encoded_o_dim = obs_encoder.out_dim
        self.device = option['device']
        self.out_dim = option['out_dim']
        init_std = option['init_std']

        if option['mps'] is None:
            self.mps = []
            for k in range(self.length):

                dim_0 = self.rank
                dim_1 = self.encoded_a_dim
                dim_2 = self.encoded_o_dim
                dim_3 = self.rank

                if k == 0:
                    dim_0 = 1
                if k == self.length - 1:
                    dim_3 = self.out_dim

                H_tmp = torch.nn.parameter.Parameter(torch.tensor(torch.normal(0, init_std, [dim_0, dim_1, dim_2, dim_3]),
                                                                  requires_grad=True)).to(self.device)
                self.mps.append(H_tmp)
        else:
            self.mps = option['mps']
        self.leakyReLu = torch.nn.LeakyReLU(inplace=False)
        self._get_params()

    def forward(self, x):
        actions = x[0]
        obss = x[1]
        if not torch.is_tensor(actions):
            actions = torch.from_numpy(actions).float()
        if not torch.is_tensor(obss):
            obss = torch.from_numpy(obss).float()
        actions = actions.float()
        obss = obss.float()

        assert self.length == actions.shape[1] and self.length == obss.shape[1], print('length mismatch between Hankel and input')

        input_action_shape = actions.shape
        input_obs_shape = obss.shape
        action_r = actions.reshape(actions.shape[0] * actions.shape[1], -1)
        obs_r = obss.reshape(obss.shape[0] * obss.shape[1], -1)
        encoded_action = self.action_encoder(action_r)
        encoded_obs = self.obs_encoder(obs_r)

        act_seq = encoded_action.reshape(input_action_shape[0], input_action_shape[1], -1)
        obs_seq = encoded_obs.reshape(input_obs_shape[0], input_obs_shape[1], -1)

        tmp = torch.einsum('ijkl, nj, nk -> nil', self.mps[0], act_seq[:, 0, :], obs_seq[:, 0, :]).squeeze()
        for i in range(1, self.length):
            tmp = torch.einsum('ni, ijkl, nj, nk -> nl', tmp, self.mps[i], act_seq[:, i, :], obs_seq[:, i, :])

        return tmp.squeeze()

    def _get_params(self):
        self.params = nn.ParameterList([])
        for i in range(len(self.mps)):
            self.params.append(self.mps[i])

        for i in range(len(self.action_encoder.encoder)):
            self.params.append(self.action_encoder.encoder[i].weight)
            self.params.append(self.action_encoder.encoder[i].bias)

        for i in range(len(self.obs_encoder.encoder)):
            self.params.append(self.obs_encoder.encoder[i].weight)
            self.params.append(self.obs_encoder.encoder[i].bias)
        return

    def convert_to_np(self, **option):
        '''
        Convert pytorch version of the MPS to Numpy version
        '''
        option_default = {
            'mps': self.mps
        }
        option = {**option_default, **option}
        mps = option['mps']

        mps_numpy = []
        for i in range(len(mps)):
            mps_numpy_core = torch.einsum('ijkl->iljk',  mps[i])



def Hankel_test(load_kde = False, kde_address = 'kde_test', window_size = 3):
    kde_option = {
        'env': gym.make('Pendulum-v0'),
        'num_trajs': 100,
        'max_episode_length': 10,
        'window_size': window_size
    }
    if load_kde:
        f = open(kde_address, "rb")
        kde = pickle.load(f)
        f.close()
    else:
        kde = construct_KDE(**kde_option)
        f = open(kde_address, "wb")
        pickle.dump(kde, f)
        f.close()

    train_gen_option = {
        'env': gym.make('Pendulum-v0'),
        'num_trajs': 1000,
        'max_episode_length': 100,
        'window_size': window_size}
    validate_gen_option = {
        'env': gym.make('Pendulum-v0'),
        'num_trajs': 1000,
        'max_episode_length': 10,
        'window_size': window_size}

    hankel_option = {
        'rank': 20,
        'out_dim': 1,
        'max_length': window_size,
        'device': 'cpu',
        'freeze_encoder': False,
        'mps': None,
        'init_std': 0.1
    }
    train_dataset =  get_dataset(kde, **train_gen_option)
    train_loader = get_data_generator(train_dataset, batch_size=256, shuffle=True, num_workers=0)

    validate_dataset = get_dataset(kde, **validate_gen_option)
    validate_loader = get_data_generator(validate_dataset, batch_size=256, shuffle=True, num_workers=0)

    #print(train_dataset.action.shape)
    action_encoder_option = {
        'input_dim': train_dataset.action.shape[2],
        'hidden_units': [10],
        'out_dim': 10,
        'final_activation': torch.nn.Tanh(),
        'inner_activation': torch.nn.LeakyReLU(inplace=False)
    }

    obs_encoder_option = {
        'input_dim': train_dataset.obs.shape[2],
        'hidden_units': [10],
        'out_dim': 10,
        'final_activation': torch.nn.Tanh(),
        'inner_activation': torch.nn.LeakyReLU(inplace=False)
    }

    scheduler_params = {
        'step_size': 500,
        'gamma': 0.1
    }
    train_option = {
        'epochs': 1000,
        'verbose': True,
        'lr': 0.001
    }
    action_encoder = Encoder(**action_encoder_option)
    obs_encoder = Encoder(**obs_encoder_option)

    hankel = Hankel(action_encoder, obs_encoder, **hankel_option)

    optimizer = optim.Adam(hankel.parameters(), lr=train_option['lr'], amsgrad=True)
    scheduler = StepLR(optimizer, **scheduler_params)

    train_lambda = lambda model: train(model, hankel_option['device'], train_loader, optimizer)
    validate_lambda = lambda model: validate(model, hankel_option['device'], validate_loader)

    train_option = {
        'verbose': True,
        'epochs': 1000
    }
    print(train_option)
    train_validate(hankel, train_lambda, validate_lambda, scheduler, train_option)

def Hankel_test_simple():
    action_encoder_option = {
        'input_dim': 2,
        'hidden_units': [],
        'out_dim': 2
    }

    obs_encoder_option = {
        'input_dim': 2,
        'hidden_units': [],
        'out_dim': 2
    }

    scheduler_params = {
        'step_size': 500,
        'gamma': 0.1
    }
    train_option = {
        'epochs': 1000,
        'verbose': True,
        'lr': 0.01
    }
    hankel_option = {
        'rank': 5,
        'out_dim': 1,
        'max_length': 5,
        'device': 'cpu',
        'freeze_encoder': False,
        'mps': None,
        'init_std': 1
    }
    action_encoder_test = Encoder(**action_encoder_option)
    obs_encoder_test = Encoder(**obs_encoder_option)

    hankel_test = Hankel(action_encoder_test, obs_encoder_test, **hankel_option)
    actions = torch.rand(1000, 5, 2)
    obs = torch.rand(1000, 5, 2)
    y = hankel_test([actions, obs]).detach()
    training = Dataset(data=[actions, obs, y])
    actions = torch.rand(1000, 5, 2)
    obs = torch.rand(1000, 5, 2)
    y = hankel_test([actions, obs]).detach()
    validating = Dataset(data=[actions, obs, y])
    generator_params = {'batch_size': 512,
                        'shuffle': True,
                        'num_workers': 0}
    train_loader = torch.utils.data.DataLoader(training, **generator_params)
    validate_loader = torch.utils.data.DataLoader(validating, **generator_params)


    action_encoder = Encoder(**action_encoder_option)
    obs_encoder = Encoder(**obs_encoder_option)

    hankel_option = {
        'rank': 5,
        'out_dim': 1,
        'max_length': 5,
        'device': 'cpu',
        'freeze_encoder': False,
        'mps': None,
        'init_std': 0.1
    }

    hankel = Hankel(action_encoder, obs_encoder, **hankel_option)
    optimizer = optim.Adam(hankel.parameters(), lr=train_option['lr'], amsgrad=True)
    scheduler = StepLR(optimizer, **scheduler_params)

    train_lambda = lambda model: train(model, hankel_option['device'], train_loader, optimizer)
    validate_lambda = lambda model: validate(model, hankel_option['device'], validate_loader)

    train_option = {
        'verbose': True,
        'epochs': 1000
    }
    print(train_option)
    train_validate(hankel, train_lambda, validate_lambda, scheduler, train_option)



if __name__ == '__main__':
    print('starting test')
    Hankel_test(load_kde=False, kde_address='kde_test', window_size=5)
    #Hankel_test_simple()



