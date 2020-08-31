from typing import Dict, Any

from Continuous_Hankel_GD import Hankel, Training_process
import gym
import torch
import tensorly as tl
from dataset import Dataset
import pickle
import numpy as np
from Get_Prob_reward import contruct_KDE, construct_dataset_PR, generate_data
def train_hankels(training_generator, vali_generator, step_size = 100, lr = 0.01, gamma = 0.1,
                  rank = 5, input_dim = 4, encoded_dim = 10, encoder_hidden = 10,
                 output_dim = 2, l = 3, seed=0, device='cpu', if_rescale_weights= False, encoder = None, epochs = 500):
    hankel = Hankel(rank = rank, input_dim = input_dim,
                    encoded_dim = encoded_dim, encoder_hidden = encoder_hidden,
                    output_dim = output_dim, max_length = l,
                    seed=seed, device=device, if_rescale_weights= if_rescale_weights, encoder = encoder)

    hankel, train_error, vali_error = Training_process(hankel=hankel, training_generator=training_generator
                                                       , validation_generator=vali_generator,
                                                       lr=lr, step_size=step_size, gamma=gamma, epochs= epochs)
    return hankel

def normalize(x):
    return (x - np.mean(x))/np.std(x)
def get_all_hankels(training_generators, vali_generators, step_size = 100, lr = 0.01, gamma = 0.1,
                    rank = 5, input_dim = 4, encoded_dim = 10, encoder_hidden = 10,
                    output_dim = 2, l = 3, seed=0, device='cpu', if_rescale_weights= False, epochs = 500):
    # generator_params = {'batch_size': batch_size,
    #                     'shuffle': shuffle,
    #                     'num_workers': num_workers}
    hankel_params = {'step_size': step_size,
                     'lr': lr,
                     'gamma': gamma,
                     'rank': rank,
                     'input_dim': input_dim,
                     'encoded_dim':encoded_dim,
                     'encoder_hidden': encoder_hidden,
                     'output_dim': output_dim,
                     'seed': seed,
                     'device': device,
                     'if_rescale_weights': if_rescale_weights,
                     'epochs': epochs
    }
    # training_generator = torch.utils.data.DataLoader(trainings['2l1'], **generator_params)
    # vali_generator = torch.utils.data.DataLoader(validations['2l1'], **generator_params)
    hankel_l = train_hankels(training_generators['l'], vali_generators['l'], encoder=None, **hankel_params, l=l)
    print('Start constructing hankel for l')
    hankel_2l1 = train_hankels(training_generators['2l1'], vali_generators['2l1'], encoder = None, **hankel_params, l = 2*l+1)
    encoder = [hankel_2l1.encoder1, hankel_2l1.encoder2]
    #print(encoder[0].weight)

    # training_generator = torch.utils.data.DataLoader(trainings['2l'], **generator_params)
    # vali_generator = torch.utils.data.DataLoader(validations['2l'], **generator_params)
    print('Start constructing hankel for 2l')
    hankel_2l = train_hankels(training_generators['2l'], vali_generators['2l'], encoder=encoder, **hankel_params, l = 2*l)
    #print(encoder[0].weight)

    # training_generator = torch.utils.data.DataLoader(trainings['l'], **generator_params)
    # vali_generator = torch.utils.data.DataLoader(validations['l'], **generator_params)
    print('Start constructing hankel for 2l1')
    hankel_l = train_hankels(training_generators['l'], vali_generators['l'], encoder=encoder, **hankel_params, l = l)
    return hankel_2l1, hankel_2l, hankel_l

def get_data_generator(kde, env = gym.make('Pendulum-v0'), num_trajs = 1000, max_episode_length = 10,
                       batch_size = 512, window_size=5, shuffle = True, num_workers = 0):

    generator_params = {'batch_size': batch_size,
                        'shuffle': shuffle,
                        'num_workers': num_workers}
    kde_params = {'env': env,
                  'num_trajs': num_trajs,
                  'max_episode_length': max_episode_length,
                  'window_size':window_size}

    observation_all_pr, action_all_pr, x_pr, new_rewards_pr = generate_data(**kde_params)


    pr = construct_dataset_PR(kde, x_pr, new_rewards_pr).reshape(-1, 1)
    #pr = normalize(pr)
    #print(pr[:5], new_rewards_pr[:5])

    new_data = Dataset(data=[tl.tensor(x_pr).float(), tl.tensor(pr).float()])

    # Generators
    generator = torch.utils.data.DataLoader(new_data, **generator_params)
    return generator

def get_all_kdes(kde_params, l):
    print('Start constructing kde for l')
    observation_all, action_all, x, new_rewards = generate_data(**kde_params, window_size=l)
    kde_l = contruct_KDE(x)
    print('Finish constructing kde for l')

    print('Start constructing kde for 2l')
    observation_all, action_all, x, new_rewards = generate_data(**kde_params, window_size=2*l)
    kde_2l = contruct_KDE(x)
    print('Finish constructing kde for 2l')

    print('Start constructing kde for 2l1')
    observation_all, action_all, x, new_rewards = generate_data(**kde_params, window_size=2 * l +1)
    kde_2l1 = contruct_KDE(x)
    print('Finish constructing kde for 2l1')
    return kde_l, kde_2l, kde_2l1

if __name__ == '__main__':
    l = 2
    load_kde = True
    generator_params = {'batch_size': 128,
                        'shuffle': True,
                        'num_workers': 0}
    kde_params = {'env': gym.make('Pendulum-v0'),
                  'num_trajs': 2000,
                  'max_episode_length': 10}
    if not load_kde:
        kde_l, kde_2l, kde_2l1 = get_all_kdes(kde_params, l)
        pickle.dump(kde_l, open('kde_l','wb'))
        pickle.dump(kde_2l, open('kde_2l', 'wb'))
        pickle.dump(kde_2l1, open('kde_2l1', 'wb'))
    else:
        kde_l = pickle.load(open('kde_l','rb'))
        kde_2l = pickle.load(open('kde_2l', 'rb'))
        kde_2l1 = pickle.load(open('kde_2l1', 'rb'))


    sampling_params_train = {
                      'env': gym.make('Pendulum-v0'),
                      'num_trajs': 1000,
                      'max_episode_length': 10}
    sampling_params_vali = {'env': gym.make('Pendulum-v0'),
                             'num_trajs': 1000,
                             'max_episode_length': 10}
    training_generators = {}
    vali_generators = {}

    training_generators['l'] = get_data_generator(**generator_params, **sampling_params_train, kde = kde_l, window_size= l)
    vali_generators['l'] =  get_data_generator(**generator_params, **sampling_params_vali, kde = kde_l, window_size= l)

    training_generators['2l'] = get_data_generator(**generator_params, **sampling_params_train, kde=kde_2l, window_size=2*l)
    vali_generators['2l'] = get_data_generator(**generator_params, **sampling_params_vali, kde=kde_2l, window_size= 2*l)

    training_generators['2l1'] = get_data_generator(**generator_params, **sampling_params_train, kde=kde_2l1, window_size= 2*l+1)
    vali_generators['2l1'] = get_data_generator(**generator_params, **sampling_params_vali, kde=kde_2l1, window_size= 2*l+1)

    hankel_params =  {'step_size' : 100, 'epochs': 100, 'lr' : 0.01, 'gamma' : 0.1,
                    'rank' :10, 'input_dim' : 4, 'encoded_dim' : 5, 'encoder_hidden' : 10,
                    'output_dim' : 1, 'l' : l, 'seed':1993, 'device':'cpu', 'if_rescale_weights': False}
    get_all_hankels(training_generators, vali_generators, **hankel_params)
