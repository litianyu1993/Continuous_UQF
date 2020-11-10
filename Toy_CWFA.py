from CWFA_AO import CWFA_AO
import numpy as np
from dataset import  Dataset
import tensorly as tl
from Getting_Hankels import get_data_generator, construct_all_hankels
L = 2
load_kde = True
lr = 0.001
epochs = 10000

generator_params = {'batch_size': 512,
                    'shuffle': True,
                    'num_workers': 0}
Hankel_params = {'rank': 5,
                 'encoded_dim_action': 3,
                 'encoded_dim_obs': 3,
                 'hidden_units_action': [3],
                 'hidden_units_obs': [3],
                 'seed': 0,
                 'device': 'cpu',
                 'rescale': False}
scheduler_params = {
    'step_size': 500,
    'gamma': 0.5
}
rank = Hankel_params['rank']


D = 2

alpha = np.random.rand(1, rank)*2 -1
A = np.random.rand(rank, D, D, rank)*2-1
Omega = np.random.rand(rank, 1)*2-1
cwfa = CWFA_AO(alpha, A, Omega*10)
L_vec = [L, 2*L, 2*L+1]
train_generators = {}
vali_generators = {}
for T in L_vec:
    N = 10000
    act = np.random.rand(N, T, D)
    obs = np.random.rand(N, T, D)
    Y = cwfa.predict(act, obs)
    print(Y.shape)
    tl.set_backend('pytorch')
    train_data = Dataset(data=[tl.tensor(act).float(), tl.tensor(obs).float(), tl.tensor(Y).float()])

    N = 1000
    act = np.random.rand(N, T, D)
    obs = np.random.rand(N, T, D)
    Y = cwfa.predict(act, obs)
    vali_data = Dataset(data=[tl.tensor(act).float(), tl.tensor(obs).float(), tl.tensor(Y).float()])
    train_gen = get_data_generator(dataset=train_data, **generator_params)
    vali_gen = get_data_generator(dataset=vali_data, **generator_params)
    if T == L:
        label = 'l'
    elif T == 2*L:
        label = '2l'
    else:
        label = '2l1'
    train_generators[label] = train_gen
    vali_generators[label] = vali_gen
hankel_l, hankel_2l, hankel_2l1 = construct_all_hankels(L, lr, epochs, {'l':train_data}, train_generators, vali_generators, Hankel_params,
                                                            scheduler_params)
