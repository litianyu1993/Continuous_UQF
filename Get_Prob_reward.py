from Getting_traj import get_trajectories, random_agent
import gym
import KDE
import numpy as np
# Specifying hyper-parameters

def sliding_window(action_obs, rewards, window_size, action_all, obs_all):
    windowed_action_obs = []
    windowed_rewards = []
    windowed_action = []
    windowed_obs = []
    assert action_obs.shape[1] == rewards.shape[1]
    for j in range(action_obs.shape[0]):
        for i in range(action_obs.shape[1] - window_size):
            windowed_action_obs.append(action_obs[j, i:i+window_size, :])
            windowed_rewards.append(rewards[j, i+window_size])
            windowed_obs.append(obs_all[j, i:i+window_size, :])
            windowed_action.append(action_all[j, i:i + window_size, :])
    #print(action_all.shape, obs_all.shape)
    #print(np.asarray(windowed_action).shape, np.asarray(windowed_action).shape)
    return np.asarray(windowed_action_obs), np.asarray(windowed_rewards), np.asarray(windowed_action), np.asarray(windowed_obs)



def contruct_KDE(x):
    '''
    :param env:
    :param num_trajs:
    :param max_episode_length: Max_episode_length/window_size needs to be an integer for now
    :param windwo_size:
    :return:
    '''
    # observation_all, reward_all, action_all = get_trajectories(random_agent, env=env,
    #                                                            num_trajs=num_trajs,
    #                                                            max_episode_length=max_episode_length)
    # action_obs = KDE.combine_obs_action(observation_all, action_all)
    #
    # x, new_rewards = sliding_window(action_obs, reward_all, window_size)
    kde = KDE.Compute_KDE(x.reshape(x.shape[0], -1))
    return kde
def generate_data(env = gym.make('Pendulum-v0'), num_trajs = 1000, max_episode_length = 10, window_size = 5):
    observation_all, reward_all, action_all = get_trajectories(random_agent, env=env,
                                                               num_trajs=num_trajs,
                                                               max_episode_length=max_episode_length)
    'Need to change this in the future for other environments'
    reward_all += 17 #Make reward positive

    action_obs = KDE.combine_obs_action(observation_all, action_all)
    x, new_rewards, action_all_new, obs_all_new = sliding_window(action_obs, reward_all, window_size, action_all, observation_all)
    return action_all_new, obs_all_new, x, new_rewards


def construct_dataset_PR(kde, x, new_rewards):
    # observation_all, reward_all, action_all = get_trajectories(random_agent, env=env,
    #                                                            num_trajs=num_trajs,
    #                                                            max_episode_length=max_episode_length)
    # action_obs = KDE.combine_obs_action(observation_all, action_all)
    #
    # x, new_rewards = sliding_window(action_obs, reward_all, window_size)

    logprob = np.asarray(KDE.compute_score(kde, x.reshape(x.shape[0], -1)))
    return np.multiply(logprob, new_rewards)

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    'Training the KDE'
    num_trajs = 1000
    max_episode_length = 10
    print('Start constructing kde')
    observation_all, action_all, x, new_rewards = generate_data(env=gym.make('Pendulum-v0'),
                                                                num_trajs=num_trajs,
                                                                max_episode_length=max_episode_length,
                                                                window_size=5)
    kde = contruct_KDE(x)
    print('Finish constructing kde')

    'Generating PR data'
    num_trajs = 1000
    max_episode_length = 10
    observation_all_pr, action_all_pr, x_pr, new_rewards_pr = generate_data(env=gym.make('Pendulum-v0'),
                                                                num_trajs=num_trajs,
                                                                max_episode_length=max_episode_length,
                                                                window_size=5)

    pr = construct_dataset_PR(kde, x_pr, new_rewards_pr)
    print(pr[:5],new_rewards_pr[:5],  x_pr.shape, action_all_pr.shape, observation_all_pr.shape)





















