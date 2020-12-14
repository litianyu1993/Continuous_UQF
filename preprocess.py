from sample_trajectories import get_trajectories, random_agent
import gym
import KDE
import numpy as np
import tensorly as tl
from Dataset import Dataset_Action_Obs_Y as Dataset
import torch
import pickle
# Specifying hyper-parameters

def sliding_window(action_obs, rewards, window_size, action_all, obs_all):
    windowed_action_obs = []
    windowed_rewards = []
    windowed_action = []
    windowed_obs = []
    assert action_obs.shape[1] == rewards.shape[1]
    for j in range(action_obs.shape[0]):
        i = 0
        while i < action_obs.shape[1] - window_size:
        #for i in range(action_obs.shape[1] - window_size):
            windowed_action_obs.append(action_obs[j, i:i+window_size, :])
            windowed_rewards.append(rewards[j, i+window_size])
            windowed_obs.append(obs_all[j, i:i+window_size, :])
            windowed_action.append(action_all[j, i:i + window_size, :])
            i+= window_size
    return np.asarray(windowed_action_obs), np.asarray(windowed_rewards), np.asarray(windowed_action), np.asarray(windowed_obs)



def construct_KDE(**option):
    option_default = {
        'env': gym.make('Pendulum-v0'),
        'num_trajs': 1000,
        'max_episode_length': 10,
        'window_size': 5,
        'agent': random_agent,
        'x': None
    }
    option = {**option_default, **option}
    if option['x'] is not None:
        x = option['x']
    else:
        _, _, x, _ = generate_data(**option)
    kde = KDE.Compute_KDE(x.reshape(x.shape[0], -1))
    return kde

def get_kde(**kde_option):
    kde_option_default = {
        'env': gym.make('Pendulum-v0'),
        'num_trajs': 100,
        'max_episode_length': 10,
        'window_size': 5,
        'load_kde': False
    }
    kde_option = {**kde_option_default, **kde_option}
    kde_address =  kde_option['env'].unwrapped.spec.id + ' ' +str(kde_option['window_size'])
    if kde_option['load_kde']:
        f = open(kde_address, "rb")
        kde = pickle.load(f)
        f.close()
    else:
        kde = construct_KDE(**kde_option)
        f = open(kde_address, "wb")
        pickle.dump(kde, f)
        f.close()
    return kde

def get_dataset(kde, **option):

    option_default = {
        'env': gym.make('Pendulum-v0'),
        'num_trajs': 1000,
        'max_episode_length': 10,
        'window_size':5}
    option = {**option_default, **option}

    action_all_pr, observation_all_pr, x_pr, new_rewards_pr = generate_data(**option)
    pr = construct_PR_target(kde, x_pr, new_rewards_pr)
    tl.set_backend('pytorch')
    new_data = Dataset(data=[tl.tensor(action_all_pr).float(), tl.tensor(observation_all_pr).float(), tl.tensor(pr).float()])


    return new_data

def get_data_generator(dataset, batch_size = 512, shuffle = True, num_workers = 0):
    generator_params = {'batch_size': batch_size,
                        'shuffle': shuffle,
                        'num_workers': num_workers}
    generator = torch.utils.data.DataLoader(dataset, **generator_params)
    return generator

def generate_data(**option):
    option_default = {
        'env': gym.make('Pendulum-v0'),
        'num_trajs': 1000,
        'max_episode_length': 10,
        'window_size': 5,
        'agent': random_agent
    }
    option = {**option_default, **option}
    window_size = option['window_size']

    option.pop('window_size', None)
    observations, rewards, actions = get_trajectories(**option)
    'Need to change this in the future for other environments'
    #reward_all += 17 #Make reward positive

    action_obs = KDE.combine_obs_action(observations, actions)
    x, new_rewards, actions_new, obss_new = sliding_window(action_obs, rewards, window_size, actions, observations)

    return actions_new, obss_new, x, new_rewards

def normalize(x):
    ori_shape = x.shape
    new_x = x.reshape(x.shape[0], -1)
    new_x = (new_x - np.mean(new_x, axis = 0))/np.std(new_x, axis=0)
    return new_x.reshape(ori_shape)

def construct_PR_target(kde, x, rewards):
    logprob = np.asarray(KDE.compute_score(kde, x.reshape(x.shape[0], -1)))
    return np.multiply(logprob, rewards)

def get_data_loaders(kde, batch_size = 256, **option_list):

    option_list_default = {
        'train_gen_option': {
            'env': gym.make('Pendulum-v0'),
            'num_trajs': 1000,
            'max_episode_length': 100,
            'window_size': 5
        },
        'validate_gen_option':{
            'env': gym.make('Pendulum-v0'),
            'num_trajs': 1000,
            'max_episode_length': 10,
            'window_size': 5
        }
    }
    option_list = {**option_list_default, **option_list}
    train_gen_option = option_list['train_gen_option']
    validate_gen_option = option_list['validate_gen_option']

    train_dataset = get_dataset(kde, **train_gen_option)
    train_loader = get_data_generator(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    validate_dataset = get_dataset(kde, **validate_gen_option)
    validate_loader = get_data_generator(validate_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_dataset, train_loader, validate_dataset, validate_loader

if __name__ == '__main__':
    options = {
        'env': gym.make('Pendulum-v0'),
        'num_trajs': 1000,
        'max_episode_length': 10,
        'window_size': 5,
        'agent': random_agent
    }
    print('Start constructing kde')
    observation_all, action_all, x, new_rewards = generate_data(**options)
    kde_test = construct_KDE(**{'x':x})
    print('Finish constructing kde')

    'Generating PR data'
    options['num_trajs'] = 1000

    observation_all_pr, action_all_pr, x_pr, new_rewards_pr = generate_data(**options)

    pr = construct_PR_target(kde_test, x_pr, new_rewards_pr)
    print(pr[:5],new_rewards_pr[:5],  x_pr.shape, action_all_pr.shape, observation_all_pr.shape)





















