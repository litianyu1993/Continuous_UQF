
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
        'decoder': None,
        'range': (-1, 1),
        'actions_mins': [-2],
        'actions_maxs': [2]
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
            #option['env'].render()
            if option['uqf'] is not None:
                history = [action_epi, obs_epi]
                from acting import UQF_agent
                #print(option['uqf'])
                agent_option = {'range': option['range']}
                action = UQF_agent(uqf=option['uqf'], next_A=option['next_A'], decoder=option['decoder'], history = history, **agent_option)
                #print(type(action), action.detach())
                action = action.detach().numpy()
                for i in range(len(action)):
                    if action[i] > option['actions_maxs'][i]:
                        action[i] = option['actions_maxs'][i]
                    if action[i] < option['actions_mins'][i]:
                        action[i] = option['actions_mins'][i]
            else:
                action = option['agent'](option['env'], observation)

            #print(action)
            observation, reward, done, info = option['env'].step(action)
            #print(action, reward)
            obs_epi.append(observation)
            action_epi.append(action)
            reward_epi.append(reward)
            #print(reward)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
        observation_all.append(obs_epi)
        reward_all.append(reward_epi)
        #print(np.sum(np.asarray(reward_epi)))
        action_all.append(action_epi)
    option['env'].close()
    return np.asarray(observation_all), np.asarray(reward_all), np.asarray(action_all)

if __name__ == '__main__':
    option = {
        'agent': random_agent,
        'env': gym.make('Pendulum-v0'),
        'num_trajs': 1,
        'max_episode_length': 100
    }
    observation_all, reward_all, action_all = get_trajectories(**option)
    print(observation_all.shape)