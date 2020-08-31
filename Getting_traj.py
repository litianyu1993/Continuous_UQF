
import gym
import numpy as np


def random_agent(env, state):
    return env.action_space.sample()

def get_trajectories(policy, env = gym.make('Pendulum-v0'), num_trajs = 100, max_episode_length = 100):
    observation_all = []
    reward_all = []
    action_all = []
    for i_episode in range(num_trajs):
        observation = env.reset()
        obs_epi = []
        reward_epi = []
        action_epi = []
        for t in range(max_episode_length):
            #env.render()
            action = policy(env, observation)
            observation, reward, done, info = env.step(action)
            obs_epi.append(observation)
            action_epi.append(action)
            reward_epi.append(reward)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
        observation_all.append(obs_epi)
        reward_all.append(reward_epi)
        action_all.append(action_epi)
    env.close()
    return np.asarray(observation_all), np.asarray(reward_all), np.asarray(action_all)

#observation_all, reward_all, action_all = get_trajectories(random_agent)