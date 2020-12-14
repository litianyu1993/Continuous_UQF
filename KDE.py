from sample_trajectories import get_trajectories, random_agent
import numpy as np
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold
import gym
import pickle
env = gym.make('Pendulum-v0')
num_trajs = 1000
max_traj_length = 10

def combine_obs_action(observations, actions):
    return np.concatenate((actions, observations), axis = 2)

def normalize(x):
    return (x - np.mean(x, axis = 0))/np.std(x, axis = 0), np.mean(x, axis = 0), np.std(x, axis = 0)

def Compute_KDE(x, bandwidths = 10 ** np.linspace(-1, 1, 100)):
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': bandwidths},
                        cv=KFold(n_splits=2, shuffle= True))
    grid.fit(x)
    return grid

def compute_score(kde, x):
    scores = []
    for i in range(len(x)):
        scores.append(kde.score(x[i].reshape(1, -1)))
    return scores

def compute_prob(kde, x):
    scores = []
    for i in range(len(x)):
        scores.append(np.exp(kde.score(x[i].reshape(1, -1))))
    return scores



def simple_test_KDE():
    option = {
        'num_trajs': 1000,
        'max_episode_length': 10
    }
    observation_all, reward_all, action_all = get_trajectories(**option)

    action_obs = combine_obs_action(observation_all, action_all)
    # print(action_obs.shape, reward_all.shape)
    x = action_obs.reshape(num_trajs, -1)
    print(x.shape)
    kde = Compute_KDE(x)

    observation_all, reward_all, action_all = get_trajectories(**option)
    action_obs = combine_obs_action(observation_all, action_all)
    x_test = action_obs.reshape(num_trajs, -1)

    logprob = compute_score(kde, x_test)
    print(logprob)

def load_kde(kde_address):
    f = open(kde_address, "rb")
    kde = pickle.load(f)
    f.close()
    return kde
if __name__ == '__main__':
    simple_test_KDE()






