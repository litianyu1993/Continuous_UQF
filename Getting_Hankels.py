from typing import Dict, Any

from Continuous_Hankel_GD import Hankel, Training_process
import gym
import torch
import tensorly as tl
from Dataset import Dataset_Action_Obs_Y as Dataset
import pickle
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
#from TT_learning import TT_spectral_learning
from matplotlib import pyplot as plt
from preprocess import construct_KDE, construct_PR_target, generate_data

def train_hankel(training_generator, vali_generator, hankel,device, scheduler, optimizer, epochs, verbose = True):


    hankel, train_error, vali_error = Training_process(hankel, training_generator, vali_generator,
                                                           scheduler, optimizer, device=device,
                                                            verbose=verbose,epochs = epochs)
    plt.plot(np.log(train_error[2:]), label = 'Training')
    plt.plot(np.log(vali_error[2:]), label = 'Validation')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.show()
    return hankel

#
# def normalize(x):
#     return (x - np.mean(x))/np.std(x)
# def normalize_tensor(x):
#     ori_shape = x.shape
#     new_x = x.reshape(x.shape[0], -1)
#     return (new_x - np.mean(new_x, axis = 0)/ np.std(new_x, axis=0)).reshape(ori_shape)
#

def get_dataset(kde, env = gym.make('Pendulum-v0'), num_trajs = 1000, max_episode_length = 10,
                       window_size=5):

    kde_params = {'env': env,
                  'num_trajs': num_trajs,
                  'max_episode_length': max_episode_length,
                  'window_size':window_size}

    action_all_pr, observation_all_pr, x_pr, new_rewards_pr = generate_data(**kde_params)


    pr = construct_PR_target(kde, x_pr, new_rewards_pr)
    #pr = normalize(pr)
    tl.set_backend('pytorch')
    #print(action_all_pr.shape, observation_all_pr.shape)
    new_data = Dataset(data=[tl.tensor(action_all_pr).float(), tl.tensor(observation_all_pr).float(), tl.tensor(pr).float()])

    # Generators

    return new_data

def get_data_generator(dataset, batch_size = 512, shuffle = True, num_workers = 0):
    generator_params = {'batch_size': batch_size,
                        'shuffle': shuffle,
                        'num_workers': num_workers}
    generator = torch.utils.data.DataLoader(dataset, **generator_params)
    return generator

def get_all_kdes(kde_params, l):
    print('Start constructing kde for l')
    observation_all, action_all, x, new_rewards = generate_data(**kde_params, window_size=l)
    kde_l = construct_KDE(**{'x':x})
    print('Finish constructing kde for l')

    print('Start constructing kde for 2l')
    observation_all, action_all, x, new_rewards = generate_data(**kde_params, window_size=2*l)
    kde_2l = construct_KDE(**{'x':x})
    print('Finish constructing kde for 2l')

    print('Start constructing kde for 2l1')
    observation_all, action_all, x, new_rewards = generate_data(**kde_params, window_size=2 * l +1)
    kde_2l1 = construct_KDE(**{'x':x})
    print('Finish constructing kde for 2l1')
    return kde_l, kde_2l, kde_2l1

def extract_Hankel_and_encoder_from_NN(model):
    Han = [model.H[i].detach().numpy() for i in range(len(model.H))]
    return Han

# def spectral_learning(H2l, H2l1, Hl):
#     return TT_spectral_learning(H2l, H2l1, Hl)
def construct_all_hankels(L, lr, epochs, training_dataset, training_generators, vali_generators, Hankel_params, scheduler_params):



    Hankel_params['action_dim'] = training_dataset['l'].action.shape[-1]
    Hankel_params['obs_dim'] = training_dataset['l'].obs.shape[-1]
    Hankel_params['output_dim'] = training_dataset['l'].y.ndim
    for j in range(5):
        get_hankel_length = lambda freeze_encoder, L, encoder_action, encoder_obs: Hankel(**Hankel_params,
                                                                                          freeze_encoder=freeze_encoder,
                                                                                          max_length=L,
                                                                                          encoder_action=encoder_action,
                                                                                          encoder_obs=encoder_obs)
        # hankel_l = get_hankel_length(False, L=L, encoder_action=None,
        #                              encoder_obs=None)
        # optimizer = optim.Adamax(hankel_l.get_params(), lr=lr)
        # scheduler = StepLR(optimizer, **scheduler_params)
        # hankel_l = train_hankel(training_generators['l'], vali_generators['l'], hankel_l,
        #                         device=Hankel_params['device'],
        #                         scheduler=scheduler, optimizer=optimizer, verbose=True, epochs=epochs)


        hankel_2l1 = get_hankel_length(False, L=2 * L + 1, encoder_action=None, encoder_obs=None)
        optimizer = optim.Adamax(hankel_2l1.get_params(), lr=lr)
        scheduler = StepLR(optimizer, **scheduler_params)
        hankel_2l1 = train_hankel(training_generators['2l1'], vali_generators['2l1'], hankel_2l1,
                                  device=Hankel_params['device'],
                                  scheduler=scheduler, optimizer=optimizer, verbose=True, epochs=epochs)

        hankel_2l = get_hankel_length(True, L=2 * L, encoder_action=hankel_2l1.encoder_action,
                                      encoder_obs=hankel_2l1.encoder_obs)
        optimizer = optim.Adamax(hankel_2l.get_params(), lr=lr)
        scheduler = StepLR(optimizer, **scheduler_params)
        hankel_2l = train_hankel(training_generators['2l'], vali_generators['2l'], hankel_2l,
                                 device=Hankel_params['device'],
                                 scheduler=scheduler, optimizer=optimizer, verbose=True, epochs=epochs)

        hankel_l = get_hankel_length(True, L=L, encoder_action=hankel_2l1.encoder_action,
                                     encoder_obs=hankel_2l1.encoder_obs)
        optimizer = optim.Adamax(hankel_l.get_params(), lr=lr)
        scheduler = StepLR(optimizer, **scheduler_params)
        hankel_l = train_hankel(training_generators['l'], vali_generators['l'], hankel_l,
                                device=Hankel_params['device'],
                                scheduler=scheduler, optimizer=optimizer, verbose=True, epochs=epochs)
    return hankel_l, hankel_2l, hankel_2l1

if __name__ == '__main__':
    L = 2
    load_kde = True
    env_name = 'Pendulum-v0'
    lr = 0.01
    epochs = 100

    generator_params = {'batch_size': 512,
                        'shuffle': True,
                        'num_workers': 0}
    kde_params = {'env': gym.make(env_name),
                  'num_trajs': 500,
                  'max_episode_length': 10}

    Hankel_params = {'rank': 5,
                     'encoded_dim_action': 3,
                     'encoded_dim_obs': 3,
                     'hidden_units_action': [5],
                     'hidden_units_obs': [5],
                     'seed': 0,
                     'device': 'cpu',
                     'rescale': True}
    sampling_params_train = {'env': gym.make(env_name),
                            'num_trajs': 5000,
                            'max_episode_length': 10}
    sampling_params_vali = {'env': gym.make(env_name),
                            'num_trajs': 1000,
                            'max_episode_length': 10}
    scheduler_params = {
        'step_size': 50,
        'gamma': 0.1
    }

    #hankel_l, hankel_2l, hankel_2l1 = construct_all_hankels(L, lr, epochs, training_dataset, training_generators, vali_generators, Hankel_params, scheduler_params)
    if not load_kde:
        kde_l, kde_2l, kde_2l1 = get_all_kdes(kde_params, L)
        pickle.dump(kde_l, open('kde_l'+env_name,'wb'))
        pickle.dump(kde_2l, open('kde_2l'+env_name, 'wb'))
        pickle.dump(kde_2l1, open('kde_2l1'+env_name, 'wb'))
    else:
        kde_l = pickle.load(open('kde_l'+env_name,'rb'))
        kde_2l = pickle.load(open('kde_2l'+env_name, 'rb'))
        kde_2l1 = pickle.load(open('kde_2l1'+env_name, 'rb'))



    training_generators = {}
    vali_generators = {}
    training_dataset = {}
    vali_dataset = {}

    training_dataset['l'] = get_dataset(**sampling_params_train, kde = kde_l, window_size= L)
    training_generators['l'] = get_data_generator(dataset=training_dataset['l'], **generator_params)
    vali_dataset['l'] = get_dataset(**sampling_params_vali, kde = kde_l, window_size= L)
    vali_generators['l'] = get_data_generator(dataset=vali_dataset['l'], **generator_params)

    training_dataset['2l'] = get_dataset(**sampling_params_train, kde=kde_2l, window_size=2*L)
    training_generators['2l'] = get_data_generator(dataset=training_dataset['2l'], **generator_params)
    vali_dataset['2l'] = get_dataset(**sampling_params_vali, kde=kde_2l, window_size=2*L)
    vali_generators['2l'] = get_data_generator(dataset=vali_dataset['2l'], **generator_params)

    training_dataset['2l1'] = get_dataset(**sampling_params_train, kde=kde_2l1, window_size=2*L+1)
    training_generators['2l1'] = get_data_generator(dataset=training_dataset['2l1'], **generator_params)
    vali_dataset['2l1'] = get_dataset(**sampling_params_vali, kde=kde_2l1, window_size=2*L+1)
    vali_generators['2l1'] = get_data_generator(dataset=vali_dataset['2l1'], **generator_params)
    #print(training_dataset['l'].y.ndim)


    get_hankel_length = lambda freeze_encoder, L, encoder_action, encoder_obs: Hankel(**Hankel_params,
                                                                                      freeze_encoder=freeze_encoder,
                                                                                      max_length=L,
                                                                                      encoder_action = encoder_action,
                                                                                      encoder_obs = encoder_obs)
    hankel_2l1 = get_hankel_length(False, L = 2*L+1, encoder_action=None, encoder_obs=None)
    optimizer = optim.Adamax(hankel_2l1.get_params(), lr=lr)
    scheduler = StepLR(optimizer, **scheduler_params)
    hankel_2l1 = train_hankel(training_generators['2l1'], vali_generators['2l1'], hankel_2l1, device = Hankel_params['device'],
                 scheduler = scheduler, optimizer = optimizer, verbose=True, epochs = epochs)

    hankel_2l = get_hankel_length(True, L = 2*L, encoder_action=hankel_2l1.encoder_action,
                                  encoder_obs=hankel_2l1.encoder_obs)
    optimizer = optim.Adamax(hankel_2l.get_params(), lr=lr)
    scheduler = StepLR(optimizer, **scheduler_params)
    hankel_2l = train_hankel(training_generators['2l'], vali_generators['2l'], hankel_2l, device=Hankel_params['device'],
                 scheduler=scheduler, optimizer=optimizer, verbose=True, epochs = epochs)

    hankel_l = get_hankel_length(True, L=L, encoder_action=hankel_2l1.encoder_action,
                                  encoder_obs=hankel_2l1.encoder_obs)
    optimizer = optim.Adamax(hankel_l.get_params(), lr=lr)
    scheduler = StepLR(optimizer, **scheduler_params)
    hankel_l = train_hankel(training_generators['l'], vali_generators['l'], hankel_l,
                             device=Hankel_params['device'],
                             scheduler=scheduler, optimizer=optimizer, verbose=True, epochs = epochs)



    # hankel_params =  {'step_size' : 100, 'epochs': 500, 'lr' : 0.01, 'gamma' : 0.9,
    #                 'rank' :20, 'input_dim' : 4, 'encoded_dim' : 10, 'encoder_hidden' : 10,
    #                 'output_dim' : 1, 'l' : l, 'seed':1993, 'device':'cpu', 'if_rescale_weights': True}
    #
    # # cwfa =CWFA(rank=15, input_dim=4, encoded_dim=10, encoder_hidden=10,
    # #              output_dim=1, device='cpu', encoder=None)
    # # train_CWFA(cwfa, training_generators['2l1'], vali_generators['2l1'], lr = 0.01,
    # #            step_size = 100, gamma = 0.5, epochs = 1000, device='cpu')
    #
    #
    # hankel_2l1, hankel_2l, hankel_l = get_all_hankels(training_generators, vali_generators, **hankel_params)
    # H2l1 = extract_Hankel_and_encoder_from_NN(hankel_2l1)
    # H2l = extract_Hankel_and_encoder_from_NN(hankel_2l)
    # Hl = extract_Hankel_and_encoder_from_NN(hankel_l)
    # #CWFA = spectral_learning(H2l, H2l1, Hl)