import gym
from gym import spaces
import numpy as np
from sample_trajectories import random_agent
from preprocess import generate_data, construct_KDE
from KDE import simple_test_KDE, compute_prob, compute_score
import scipy.stats
class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low = -1, high = 1, shape=[1])
        # Example for using image as input:
        self.observation_space = spaces.Box(low = -1, high = 1, shape=[1])
        self.num_states = 2
        self.mu = np.ones(self.num_states)/self.num_states
        self.current_state = self.reset()
        self.env_ID = 'toy_cpomdp_2states'
        #self.unwrapped.spec.id = 'toy'


    def step(self, action):
        # Execute one time step within the environment
        p = (action - self.action_space.low)/(self.action_space.high - self.action_space.low)
        if self.current_state == 0:
            if p > 1-p:
                self.current_state = 0
            else:
                self.current_state = 1
        else:
            if p > 1 - p:
                self.current_state = 1
            else:
                self.current_state = 0
        o = np.random.normal(self.current_state, 1, 1)[0]
        r = np.random.normal(self.current_state, 1, 1)[0]
        info = None
        done = False
        return o, r, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_state = np.random.randint(0, self.num_states)
        return np.random.normal(self.current_state, 1, 1)[0]

    def compute_likelihood(self, actions_all, obss_all):
        'actions, obss of shape: length * num_features'
        all_likelihood = []
        for i in range(len(actions_all)):
            actions = actions_all[i]
            obss = obss_all[i]
            likelihood = np.asarray([0.5, 0.5])
            for i in range(actions.shape[0]):
                likelihood *= 0.5

                p = (actions[i] - self.action_space.low) / (self.action_space.high - self.action_space.low)
                #print(self.action_space.low, p)
                T = np.asarray([[p, 1-p], [1-p, p]])
                #print('here', likelihood.shape, T.shape)
                likelihood = likelihood @ T
                O = np.asarray([scipy.stats.norm(0, 1).pdf(obss[i]), scipy.stats.norm(1, 1).pdf(obss[i])])
                likelihood = likelihood @ np.diag(O)
            all_likelihood.append(np.sum(likelihood))
        return all_likelihood