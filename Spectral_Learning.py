from Getting_Hankels import construct_all_hankels
import gym
from TT_learning import TT_spectral_learning
import tensorly as tl
from CWFA_AO import CWFA_AO
from Encoder_FC import Encoder_FC
import pickle
from Getting_Hankels import get_dataset
import numpy as np
import torch

def convert_mpo_to_mps(mpo):
    mps = []
    for i in range(len(mpo)):
        core = mpo[i]
        core = tl.base.partial_unfold(core, mode = 2)
        core = tl.moveaxis(core, 1, 2)
        mps.append(core)
    return mps
def convert_mps_back_to_mpo(mpo, mps):
    new_mpo = []
    for i in range(len(mps)):
        core = mps[i]
        core = tl.moveaxis(core, 1, 2)
        core = tl.base.partial_fold(core, mode = 2, shape = mpo[i].shape)
        new_mpo.append(core)
    return mpo
def convert_mps_core_back_to_mpo_core(mpo_core, mps_core):
    new_mpo_core = tl.moveaxis(mps_core, 1, 2)
    new_mpo_core = tl.base.partial_fold(new_mpo_core, mode=2, shape = mpo_core.shape)
    return new_mpo_core
def extract_Hankel_and_encoder_from_NN(model):
    Han = [tl.tensor(model.H[i].detach().numpy()) for i in range(len(model.H))]
    return Han

def conver_all_hankel_to_mps(hankels):
    new_han = []
    for hankel in hankels:
        new_han.append(convert_mpo_to_mps(hankel))
    return new_han

def convert_all_hankel_back_to_mpo(hankels_mps, hankels_mpo):
    new_han = []
    for mps, mpo in zip(hankels_mps, hankels_mpo):
        new_han.append(convert_mps_back_to_mpo(mpo, mps))
    return new_han
def test(dataset, cwfa, action_encoder, obs_encoder):
    encoded_action = action_encoder(dataset.action.reshape(-1, dataset.action.shape[-1])).\
        reshape(dataset.action.shape[0], dataset.action.shape[1], -1).cpu().detach().numpy()
    encoded_obs = obs_encoder(dataset.obs.reshape(-1, dataset.obs.shape[-1])).\
        reshape(dataset.obs.shape[0], dataset.obs.shape[1], -1).cpu().detach().numpy()

    tl.set_backend('numpy')
    pred = []
    for i in range(len(encoded_obs)):
        #print(encoded_action[i].shape, encoded_obs[i].shape)
        pred.append(cwfa.predict(encoded_action[i], encoded_obs[i]))
    return np.asarray(pred).squeeze()
if __name__ == '__main__':
    L = 2
    load_kde = True
    env_name = 'Pendulum-v0'
    lr = 0.001
    epochs = 1000

    generator_params = {'batch_size': 512,
                        'shuffle': True,
                        'num_workers': 0}
    kde_params = {'env': gym.make(env_name),
                  'num_trajs': 500,
                  'max_episode_length': 10}

    Hankel_params = {'rank': 5,
                     'encoded_dim_action': 3,
                     'encoded_dim_obs': 3,
                     'hidden_units_action': [10],
                     'hidden_units_obs': [10],
                     'seed': 0,
                     'device': 'cpu',
                     'rescale': True}
    sampling_params_train = {'env': gym.make(env_name),
                             'num_trajs': 10000,
                             'max_episode_length': 10}
    sampling_params_vali = {'env': gym.make(env_name),
                            'num_trajs': 100,
                            'max_episode_length': 10}
    scheduler_params = {
        'step_size': 50,
        'gamma': 1
    }

    hankel_l, hankel_2l, hankel_2l1 = construct_all_hankels(L, load_kde, env_name, lr, epochs, generator_params,
                                                            kde_params,
                                                            sampling_params_train, sampling_params_vali, Hankel_params,
                                                            scheduler_params)
    tl.set_backend('numpy')
    Hankel_l = extract_Hankel_and_encoder_from_NN(hankel_l)
    Hankel_2l = extract_Hankel_and_encoder_from_NN(hankel_2l)
    Hankel_2l1 = extract_Hankel_and_encoder_from_NN(hankel_2l1)

    Hankels_mpo = [Hankel_l, Hankel_2l, Hankel_2l1]
    Hankels_mps = conver_all_hankel_to_mps(Hankels_mpo)

    print(Hankels_mps[-1])

    action_encoder = Encoder_FC(init_encoder=hankel_2l1.encoder_action)
    obs_encoder = Encoder_FC(init_encoder=hankel_2l1.encoder_obs)


    alpha, A, Omega = TT_spectral_learning(Hankels_mps[1], Hankels_mps[2], Hankels_mps[0])
    cwfa_ao = CWFA_AO(alpha, convert_mps_core_back_to_mpo_core(Hankel_2l1[1], A), Omega)

    print(alpha)
    print(A)
    print(Omega)

    tl.set_backend('pytorch')
    env_name = 'Pendulum-v0'
    kde_l = pickle.load(open('kde_l' + env_name, 'rb'))
    kde_2l = pickle.load(open('kde_2l' + env_name, 'rb'))
    kde_2l1 = pickle.load(open('kde_2l1' + env_name, 'rb'))

    sampling_params_test = {'env': gym.make(env_name),
                             'num_trajs': 2000,
                             'max_episode_length': 10}
    test_l = get_dataset(**sampling_params_test, kde=kde_l, window_size=L)
    test_2l= get_dataset(**sampling_params_test, kde=kde_2l, window_size=2*L)
    test_2l1 = get_dataset(**sampling_params_test, kde=kde_2l1, window_size=2 * L + 1)

    pred = test(test_l, cwfa_ao, action_encoder, obs_encoder)
    target = test_l.y.cpu().detach().numpy().squeeze()
    print('MSE for length L: ', np.mean((pred - target)**2))

    pred = test(test_2l, cwfa_ao, action_encoder, obs_encoder)
    target = test_2l.y.cpu().detach().numpy().squeeze()
    pred = test(test_2l, cwfa_ao, action_encoder, obs_encoder)
    print('MSE for length 2L: ', np.mean((pred -target) ** 2))

    pred = test(test_2l1, cwfa_ao, action_encoder, obs_encoder)
    target = test_2l1.y.cpu().detach().numpy().squeeze()
    pred = test(test_2l1, cwfa_ao, action_encoder, obs_encoder)
    print('MSE for length 2L+1: ', np.mean((pred - target) ** 2))




