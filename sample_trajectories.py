
import gym
import numpy as np


def random_agent(env, state):
    return env.action_space.sample()

def get_trajectories(**option):
    option_default = {
        'agent': random_agent,
        'env': gym.make('Pendulum-v0'),
        'num_trajs': 100,
        'max_episode_length': 100,
        'uqf': None,
        'next_A': None,
        'decoder': None
    }
    option = {**option_default, **option}
    observation_all = []
    reward_all = []
    action_all = []
    for i_episode in range(option['num_trajs']):
        observation = option['env'].reset()
        obs_epi = []
        reward_epi = []
        action_epi = []
        for t in range(option['max_episode_length']):
            #env.render()
            if option['uqf'] is not None:
                history = [action_epi, obs_epi]
                from acting import UQF_agent
                action = UQF_agent(option['uqf'], option['next_A'], option['decoder'], history, **option)
            else:
                action = option['agent'](option['env'], observation)
            #print(action)
            observation, reward, done, info = option['env'].step(action)
            obs_epi.append(observation)
            action_epi.append(action)
            reward_epi.append(reward)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
        observation_all.append(obs_epi)
        reward_all.append(reward_epi)
        action_all.append(action_epi)
    option['env'].close()
    return np.asarray(observation_all), np.asarray(reward_all), np.asarray(action_all)

if __name__ == '__main__':
    option = {
        'agent': random_agent,
        'env': gym.make('Pendulum-v0'),
        'num_trajs': 1000,
        'max_episode_length': 100
    }
    observation_all, reward_all, action_all = get_trajectories(**option)
    print(observation_all.shape)
    sum_rewards = np.sum(reward_all, axis=1)
    mean_sum_rewards = np.mean(sum_rewards)
    print('Average sum rewards: ' + str(mean_sum_rewards))