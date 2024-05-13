import gym
import pickle
import numpy as np
import random
import time
import matplotlib.pyplot as plt
def q_learning(episodes=1000, training=True, render=True):

    env = gym.make('CartPole-v1', render_mode='human' if render is True else None)
    state = env.reset()
    pos_bins = np.linspace(-2.4, 2.4, 10)
    vl_bins = np.linspace(-4, 4, 10)
    ang_bins = np.linspace(-0.2095, 0.2095, 10)
    angvl_bins = np.linspace(-4, 4, 10)
    if training:
        q = np.zeros((len(pos_bins) + 1, len(vl_bins) + 1, len(ang_bins) + 1, len(angvl_bins) + 1, env.action_space.n))
    else:
        f = open('Qlearning.pkl', 'rb')
        q= pickle.load(f)
        f.close()
    # print(q.shape)
    alpha = 0.1
    gamma = 0.99
    epsilon = 1
    epsilon_decay = 0.00001
    episodes_rewards_mean=[]
    rewards_per_episode = []
    for i in range(episodes):
        state, _ = env.reset()
        state_pos = np.digitize(state[0], pos_bins)
        state_vl = np.digitize(state[1], vl_bins)
        state_ang = np.digitize(state[2], ang_bins)
        state_angvl = np.digitize(state[3], angvl_bins)

        terminated = False
        rewards = 0
        while not terminated and rewards < 250:
            if training and random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_pos, state_vl, state_ang, state_angvl, :])
            new_state, reward, terminated, _, _ = env.step(action)
            new_state_pos = np.digitize(new_state[0], pos_bins)
            new_state_vl = np.digitize(new_state[1], vl_bins)
            new_state_ang = np.digitize(new_state[2], ang_bins)
            new_state_angvl = np.digitize(new_state[3], angvl_bins)

            if training:
                q[state_pos, state_vl, state_ang, state_angvl, action] = q[
                                                                             state_pos, state_vl, state_ang, state_angvl, action] + alpha * (
                                                                                 reward + gamma * np.max(
                                                                             q[new_state_pos, new_state_vl,
                                                                             new_state_ang, new_state_angvl, :]) - q[
                                                                                     state_pos, state_vl, state_ang, state_angvl, action]
                                                                         )
            rewards += reward
            state_pos = new_state_pos
            state_vl = new_state_vl
            state_ang = new_state_ang
            state_angvl = new_state_angvl
            # if terminated:
            #     time.sleep(2)

        rewards_per_episode.append(rewards)
        mean = np.mean(rewards_per_episode[-100:])
        episodes_rewards_mean.append(mean)
        if not training:
            print(f"episode:{i}, rewards:{rewards},epsilon:{epsilon:0.2f},last 100:{mean}")

        if training and i % 100 == 0:
            print(f"episode:{i}, rewards:{rewards},epsilon:{epsilon:0.2f},last 100:{mean}")

        if training and np.min(rewards_per_episode[-100:]) > 195:
            print(f"episode:{i}, rewards:{rewards},epsilon:{epsilon:0.2f},last 100:{mean}")
            break

        epsilon = max(epsilon - epsilon_decay, 0)
    env.close()
    if training:
        f = open('Qlearning.pkl', 'wb')
        pickle.dump(q, f)
        f.close()
    return episodes_rewards_mean


avg = q_learning(100000, training=False, render=True)

plt.plot(avg, label='Q Learning')
plt.xlabel('episode in 100s')
plt.ylabel('average reward')
plt.savefig('qlearning.png')
plt.show()

