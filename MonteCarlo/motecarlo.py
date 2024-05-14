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


def create_random_policy(
        env,
        pos_bins=pos_bins,
        vl_bins=vl_bins,
        ang_bins=ang_bins,
        angvl_bins=angvl_bins
):
    policy = {}
    for pos_bin in pos_bins:
        for vl_bin in vl_bins:
            for ang_bin in ang_bins:
                for angvl_bin in angvl_bins:
                    # policy[pos_bin, vl_bin, ang_bin, angvl_bin] = np.random.randint(1, 10, 2).astype(np.float32)
                    policy[pos_bin, vl_bin, ang_bin, angvl_bin] = np.random.randn(2).astype(np.float32)
    return policy



def run_episode(
        env,
        policy,
        display=True
):
    state, _ = env.reset()
    state_pos = pos_bins[np.digitize(state[0], pos_bins) - 1]
    state_vl = vl_bins[np.digitize(state[1], vl_bins) - 1]
    state_ang = ang_bins[np.digitize(state[2], ang_bins) - 1]
    state_angvl = angvl_bins[np.digitize(state[3], angvl_bins) - 1]

    episode = []
    alive = True
    total = 0
    while alive:
        if display:
            env.render()

        action = np.argmax(list(policy[state_pos, state_vl, state_ang, state_angvl]))
        state, reward, alive, _, _ = env.step(action)

        state_pos = pos_bins[np.digitize(state[0], pos_bins)-1]
        state_vl = vl_bins[np.digitize(state[1], vl_bins)-1]
        state_ang = ang_bins[np.digitize(state[2], ang_bins)-1]
        state_angvl = angvl_bins[np.digitize(state[3], angvl_bins)-1]

        total += reward
        episode.append([(state_pos, state_vl, state_ang, state_angvl), action, reward])
        alive = not alive

    if display:
        clear_output(True)
        env.render()

    return episode, total

def monte_carlo(
        env,
        episodes=100000,
        policy=None,
        gamma=0.95,
        lr=0.1,
        display=True
):

    if not policy:
        policy = create_random_policy(env)

    returns = {state: {0:[], 1:[]} for state in policy.keys()}
    hist = list()
    for i in range(episodes):

        episode, total = run_episode(env=env, policy=policy, display=display)
        hist.append(total)
        if i%100 == 0 : print(f"[{i}]\tReward: {np.mean(hist[-100:])}")

        G = 0
        for i in reversed(range(0, len(episode))):
            s_t, a_t, r_t = episode[i]
            G = gamma * G + r_t
            # policy[s_t][a_t] += lr*(G - policy[s_t][a_t]) ## gamma=0.1, lr=0.9,

            returns[s_t][a_t].append(G)
            policy[s_t][a_t] = np.mean(returns[s_t][a_t])

    return policy, hist






# policy = create_random_policy(env)
# print(policy)
# print(create_state_action_dictionary(env, policy))
# for _ in range(100):
#     print(run_game(env, policy))
# print(env.step(1))

policy, hist = monte_carlo(env, episodes=100000, display=False)


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




