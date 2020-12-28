from CWFA_AO import learn_CWFA
import gym
import torch
from acting import convert_wfa_to_uqf, get_decoder
import numpy as np
from sample_trajectories import get_trajectories
from Toy_CWFA import CustomEnv
import pickle
if __name__ == '__main__':
    window_size = 5
    load_model = False
    #env = gym.make('MountainCarContinuous-v0')
    env = gym.make('Pendulum-v0')
    #env = CustomEnv()
    option_lists = {
        'kde_option': {
            'env': env,
            'num_trajs': 100,
            'max_episode_length': 100,
            'window_size': window_size,
            'load_kde': False,
            'use_kde': True
        },
        'train_gen_option': {
            'env': env,
            'num_trajs': 1000,
            'max_episode_length': 100,
            'window_size': window_size},
        'validate_gen_option': {
            'env': env,
            'num_trajs': 100,
            'max_episode_length': 100,
            'window_size': window_size},

        'cwfa_option': {
            'random_init': True,
            'rank': 10,
            'device': 'cpu',
            'init_std': 0.01,
            'out_dim': 1
        },
        'action_encoder_option': {
            'hidden_units': [10],
            'out_dim': 10,
            'final_activation': torch.nn.Tanh(),
            'inner_activation': torch.nn.Tanh()
        },
        'obs_encoder_option': {
            'hidden_units': [10, 10],
            'out_dim': 10,
            'final_activation': torch.nn.Tanh(),
            'inner_activation': torch.nn.Tanh()
        },
        'fit_option': {
            'epochs': 1000,
            'verbose': True,
            'lr': 0.001,
            'step_size': 500,
            'gamma': 1
        }
    }


    #cwfa_address = 'cwfa'+ env.unwrapped.spec.id + str(window_size) + str(option_lists['cwfa_option']['rank'])
    if hasattr(env, 'env_ID'):
        cwfa_address = 'cwfa' + env.env_ID + str(window_size) + str(option_lists['cwfa_option']['rank'])
    else:
        cwfa_address = 'cwfa'+ env.unwrapped.spec.id + str(window_size) + str(option_lists['cwfa_option']['rank'])

    if load_model:
        f = open(cwfa_address, 'rb')
        cwfa = pickle.load(f)
        f.close()
    else:
        f = open(cwfa_address, 'wb')
        cwfa = learn_CWFA(**option_lists)
        pickle.dump(cwfa, f)
        f.close()

    uqf_option = {
        'num_examples': 100000,
        'range': [-1, 1],
        'input_dim': cwfa.action_encoder.input_dim
    }

    uqf, next_A = convert_wfa_to_uqf(cwfa_ao=cwfa, **uqf_option)
    decoder_option = {
        'sample_size_train': 10000,
        'sample_size_vali': 1000,
        'lr': 0.001,
        'epochs': 1000,
        'gamma': 1,
        'step_size': 500,
        'batch_size': 256
    }
    decoder_address = 'decoder' + env.unwrapped.spec.id + str(window_size) + str(option_lists['cwfa_option']['rank'])
    if load_model:
        f = open(decoder_address, 'rb')
        action_decoder = pickle.load(f)
        f.close()
    else:
        f = open(decoder_address, 'wb')
        action_decoder = get_decoder(uqf.action_encoder, **decoder_option)
        pickle.dump(action_decoder, f)
        f.close()
    sampling_option = {
        'env': env,
        'num_trajs': 1000,
        'max_episode_length': 100,
        'uqf': None,
        'next_A': None,
        'decoder': None
    }
    observation_all, reward_all, action_all = get_trajectories(**sampling_option)
    #print(reward_all)
    reward_all = reward_all.reshape(reward_all.shape[0], -1)
    sum_rewards = np.sum(reward_all, axis = 1)
    print(sum_rewards.shape)
    mean_sum_rewards = np.mean(sum_rewards)
    print('Average sum rewards: ' + str(mean_sum_rewards))
    #print(np.mean(reward_all, axis = 1))
