import gym
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pickle


env = gym.make('CartPole-v1', render_mode=None)

pos_bins = np.linspace(-2.4, 2.4, 10)
vl_bins = np.linspace(-4, 4, 10)
ang_bins = np.linspace(-0.2095, 0.2095, 10)
angvl_bins = np.linspace(-4, 4, 10)

policy = np.random.rand(
    len(pos_bins)+1,
    len(vl_bins)+1,
    len(ang_bins)+1,
    len(angvl_bins)+1,
    env.action_space.n
)

def run_episode(
        env,
        policy=policy,
        display=True
):
    state, _ = env.reset()
    state_pos = np.digitize(state[0], pos_bins)
    state_vl = np.digitize(state[1], vl_bins)
    state_ang = np.digitize(state[2], ang_bins)
    state_angvl = np.digitize(state[3], angvl_bins)

    episode = []
    alive = True
    total = 0
    while alive:
        if display:
            env.render()

        action = np.argmax(policy[state_pos, state_vl, state_ang, state_angvl,:])
        state, reward, alive, _, _ = env.step(action)

        state_pos = np.digitize(state[0], pos_bins)
        state_vl = np.digitize(state[1], vl_bins)
        state_ang = np.digitize(state[2], ang_bins)
        state_angvl = np.digitize(state[3], angvl_bins)

        total += reward
        episode.append([(state_pos, state_vl, state_ang, state_angvl, action), reward])
        alive = not alive

    if display:
        clear_output(True)
        env.render()
    return episode, total


def monte_carlo(
        env,
        episodes=100000,
        policy=policy,
        gamma=0.1,
        lr=0.9,
        display=True
):

    returns = {state: {0:[], 1:[]} for state in range(len(policy.reshape(-1, 2)))}
    hist = list()
    for i in range(episodes):

        episode, total = run_episode(env=env, policy=policy, display=display)
        hist.append(total)
        if i%100 == 0 : print(f"[{i}]\tReward: {np.mean(hist[-100:])}")

        G = 0
        for i in reversed(range(0, len(episode))):
            (state_pos, state_vl, state_ang, state_angvl, action), reward = episode[i]
            # discount = gamma ** i
            # G +=  discount * reward

            G = gamma * G + reward
            # policy[state_pos, state_vl, state_ang, state_angvl, action] += lr*(G - policy[state_pos, state_vl, state_ang, state_angvl, action]) ## gamma=0.1, lr=0.9,

            returns[state_pos * state_vl * state_ang * state_angvl][action].append(G)
            policy[state_pos, state_vl, state_ang, state_angvl, action] = np.mean(returns[state_pos * state_vl * state_ang * state_angvl][action])

    return policy, hist






# policy = create_random_policy(env)
# print(policy)
# print(create_state_action_dictionary(env, policy))
# for _ in range(100):
#     print(run_game(env, policy))
# print(env.step(1))

policy, hist = monte_carlo(env, episodes=50000, display=False)


plt.plot(hist)
plt.xlabel("Episodes")
plt.ylabel("Total Rewards")
plt.title("Total Rewards Over Episodes")
plt.savefig('MonteCarlo_all_hist.png')
plt.show()

plt.plot(np.mean(np.array(hist).reshape(-1, 100), axis=1))
plt.xlabel("Episodes")
plt.ylabel("AVG Rewards")
plt.title("AVG Rewards Over Episodes")
plt.savefig('MonteCarlo_avgs.png')
plt.show()

f = open('MonteCarlo.pkl', 'wb')
pickle.dump(policy, f)
f.close()




