from CWFA_AO import learn_CWFA
import gym
import torch
from acting import convert_wfa_to_uqf, get_decoder
import numpy as np
from sample_trajectories import get_trajectories
import pickle
if __name__ == '__main__':
    window_size = 5
    load_cwfa = False
    env = gym.make('Pendulum-v0')
    option_lists = {
        'kde_option': {
            'env': env,
            'num_trajs': 100,
            'max_episode_length': 10,
            'window_size': window_size,
            'load_kde': True
        },
        'train_gen_option': {
            'env': env,
            'num_trajs': 1000,
            'max_episode_length': 100,
            'window_size': window_size},
        'validate_gen_option': {
            'env': env,
            'num_trajs': 1000,
            'max_episode_length': 10,
            'window_size': window_size},

        'cwfa_option': {
            'random_init': True,
            'rank': 20,
            'device': 'cpu',
            'init_std': 0.1,
            'out_dim': 1
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

    cwfa_address = env.unwrapped.spec.id + str(window_size) + str(option_lists['cwfa_option']['rank'])
    if load_cwfa:
        f = open(cwfa_address, 'rb')
        cwfa = pickle.load(f)
        f.close()
    else:
        f = open(cwfa_address, 'wb')
        cwfa = learn_CWFA(**option_lists)
        pickle.dump(cwfa, f)
        f.close()

    uqf, next_A = convert_wfa_to_uqf(cwfa_ao=cwfa)
    decoder_option = {
        'sample_size_train': 10000,
        'sample_size_vali': 1000,
        'lr': 0.001,
        'epochs': 1000,
        'gamma': 1,
        'step_size': 500,
        'batch_size': 256
    }
    action_decoder = get_decoder(uqf.action_encoder, **decoder_option)

    sampling_option = {
        'env': env,
        'num_trajs': 10,
        'max_episode_length': 100,
        'uqf': None,
        'next_A': None,
        'decoder': None
    }
    observation_all, reward_all, action_all = get_trajectories(**sampling_option)
    #print(reward_all)
    reward_all = reward_all.reshape(reward_all.shape[0], -1)
    sum_rewards = np.sum(reward_all, axis = 1)
    mean_sum_rewards = np.mean(sum_rewards)
    print('Average sum rewards: ' + str(mean_sum_rewards))
    #print(np.mean(reward_all, axis = 1))
