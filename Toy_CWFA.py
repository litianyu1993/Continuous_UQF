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

    def compute_log_likelihood(self, actions_all, obss_all):
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
            all_likelihood.append(np.log(np.sum(likelihood)))
        return all_likelihood

def compare_log_likehood(**option):
    toy_cpomdp = CustomEnv()
    options_default = {
        'env': toy_cpomdp,
        'num_trajs': 1000,
        'max_episode_length': 10,
        'window_size': 5,
        'agent': random_agent
    }
    option = {**options_default, **option}
    print('Start constructing kde')
    observation_all, action_all, x, new_rewards = generate_data(**option)
    kde_test = construct_KDE(**{'x': x})
    print('Finish constructing kde')
    # print(action_all.shape, observation_all.shape)
    likelihood = []
    for i in range(len(action_all)):
        likelihood.append(toy_cpomdp.compute_likelihood(action_all[i], observation_all[i]))
    # print(np.log(likelihood))
    kde_log_likelihood = compute_score(kde_test, x)
    samples = kde_test.best_estimator_.sample(n_samples = 1000)
    actions_samples = np.zeros([samples.shape[0], int(samples.shape[1]/2), 1])
    obs_samples =  np.zeros([samples.shape[0], int(samples.shape[1]/2), 1])
    for i in range(samples.shape[0]):
        for j in range(int(samples.shape[1]/2)):
            actions_samples[i, j, 0] = samples[i, j*2]
            obs_samples[i, j, 0] = samples[i, j*2+1]
    likelihood_samples = []
    for i in range(len(actions_samples)):
        likelihood_samples.append(toy_cpomdp.compute_likelihood(actions_samples[i], obs_samples[i]))

    print(samples.shape)
    return [(np.mean((kde_log_likelihood - np.log(likelihood) ** 2))), np.mean(np.log(likelihood_samples))]

if __name__ == '__main__':
    toy_cpomdp =  CustomEnv()
    options = {
        'env': toy_cpomdp,
        'num_trajs': 100,
        'max_episode_length': 20,
        'window_size': 5,
        'agent': random_agent,
    }
    window_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    error = []
    for window_size in window_sizes:
        options['max_episode_length'] = window_size+1
        options['window_size'] = window_size
        options['num_trajs'] = 20*window_size
        error.append(compare_log_likehood(**options))
        print(error)
    error = np.asarray(error)
    mse_likelihood = error[:, 0]
    avg_log_likelihood = error[:, 1]
    from matplotlib import pyplot as plt
    plt.plot(window_sizes, mse_likelihood)
    plt.xlabel('Trajectory length')
    plt.ylabel('MSE log likelihood')
    plt.title('MSE Log Likelihood w.r.t Trajectory length')
    plt.plot()
    plt.show()

    plt.plot(window_sizes, avg_log_likelihood)
    plt.xlabel('Trajectory length')
    plt.ylabel('Average sampled log likelihood')
    plt.title('Average sampled log likelihood evaluated on the \n true model w.r.t Trajectory length')
    plt.plot()
    plt.show()

    error = []
    for window_size in window_sizes:
        options['max_episode_length'] = 21
        options['window_size'] = window_size
        options['num_trajs'] = 20 * window_size
        error.append(compare_log_likehood(**options))
        print(error)
    error = np.asarray(error)
    mse_likelihood = error[:, 0]
    avg_log_likelihood = error[:, 1]
    from matplotlib import pyplot as plt

    plt.plot(window_sizes, mse_likelihood)
    plt.xlabel('Trajectory length')
    plt.ylabel('MSE log likelihood')
    plt.title('MSE Log Likelihood w.r.t window_size, \n with 20 maximum trajectory length')
    plt.plot()
    plt.show()

    plt.plot(window_sizes, avg_log_likelihood)
    plt.xlabel('Trajectory length')
    plt.ylabel('Average sampled log likelihood')
    plt.title('Average sampled log likelihood evaluated \n on the true model w.r.t window_size, \nwith 20 maximum trajectory length')
    plt.plot()
    plt.show()