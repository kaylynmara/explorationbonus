import numpy as np
import os
import random
from Env.environment import RaceTrack
import matplotlib
import matplotlib.pyplot as plt
import pickle
matplotlib.use('TkAgg')


episodes = 100
gamma = .9  # discount factor - change this as you see fit
epsilon = .1  # exploration rate for E-Greedy policy. Change as you see fit
num_runs = 100
map_choice = "maps/track_c.npy"  # pick what map you want to use from track_builder.py
map_name = os.path.splitext(os.path.basename(map_choice))[0]

#  assigning tags for visualization tracking
""" If you're doing different iterations of this experiment ie changing variables, maps etc. you'll
need to assign these tags prior to running the experiment"""

training_session = 1 # manually assign this depending on what run you're doing
experiment_name = "baseline" # manually assign this depending on what experiment you're running

"""note that these will dynamically change the output of your .pkl file later for when you're doing
visual analysis. So make sure you're descriptive in your naming convention so that you know which visualizations
you're working with later"""

save_filename = f"training_results_{map_name}_{training_session}_{experiment_name}.pkl"
print(f"training results will be saved as: {save_filename}")

all_returns = []
Q_all_runs = []
all_rewards = []
all_steps = []

# All Possible Actions (Acceleration Choices)
actions = [(-1, -1), (-1, 0), (-1, 1), (0, 0), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def epsilon_greedy_policy(state, Q):
    """select an action using epsilon greedy policy"""
    if state not in Q:
        Q[state] = {a: 0 for a in actions}
    if random.random() < epsilon:
        return random.choice(actions)  # explore with prob epsilon

    return max(Q[state], key=Q[state].get)


def generate_episode(env, Q):
    """ run one episode and collect states, actions, rewards"""
    env.reset()
    episode = []
    done = False
    total_reward = 0
    steps = 0

    while not done:
        state = (env.position, env.velocity)
        action = epsilon_greedy_policy(state, Q)
        next_state, next_velocity, reward, done = env.step(action)
        episode.append((state, action, reward))
        total_reward += reward
        steps += 1
    return episode, total_reward, steps


def monte_carlo_control(seed):
    """ Train The Agent Using MC Control with a given seed"""
    random.seed(seed)
    np.random.seed(seed)

    Q = {}
    returns = {}
    episode_rewards = []
    episode_steps = []

    for episode in range(episodes):
        env = RaceTrack(map_choice)
        episode_data, total_reward, steps = generate_episode(env, Q)

        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        G = 0


        visited = set()
        for t in reversed(range(len(episode_data))):
            state, action, reward = episode_data[t]
            G = gamma * G + reward

            if (state, action) not in visited:
                visited.add((state, action))

                if (state, action) not in returns:
                    returns[(state, action)] = []

                returns[(state, action)].append(G)

                if state not in Q:
                    Q[state] = {a: 0 for a in actions}

                Q[state][action] = np.mean(returns[(state, action)])

        if episode % 500 == 0:
            print(f"Run {seed} - Episode {episode}/{episodes} Completed")

    print(f"Monte Carlo Training Complete for Seed:{seed}")
    return Q, returns, episode_rewards, episode_steps


if __name__ == "__main__":
    for run in range(num_runs):
        print(f"Starting Run{run + 1}/{num_runs} with Seed {run}")
        Q, returns, rewards, steps = monte_carlo_control(run)

        all_returns.append(returns)
        Q_all_runs.append(Q)
        all_rewards.append(rewards)
        all_steps.append(steps)

    print("All Runs Complete!")

    with open(save_filename, "wb") as f:
        pickle.dump({
            "all_rewards": all_rewards,
            "all_steps": all_steps,
            "Q_all_runs": Q_all_runs
        }, f)

    print(f"Training results saved to {save_filename}")

    # compute average rewards and steps across multiple runs
    avg_rewards = np.mean(all_rewards, axis=0)
    avg_steps = np.mean(all_steps, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    std_steps = np.std(all_steps, axis=0)

    plt.figure(figsize=(12, 5))

    # reward per episode
    plt.subplot(1, 2, 1)
    plt.plot(avg_rewards, label="Average Rewards Per Episode", color='blue')
    plt.fill_between(
        range(len(avg_rewards)),
        avg_rewards - std_rewards,
        avg_rewards + std_rewards,
        color='blue',
        alpha=0.2
    )
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Monte Carlo Learning: Reward per Episode")
    plt.legend()

    # steps per episode
    plt.subplot(1, 2, 2)
    plt.plot(avg_steps, label="Average Steps Per Episode", color='orange')
    plt.fill_between(
        range(len(avg_steps)),
        avg_steps - std_steps,
        avg_steps + std_steps,
        color='orange',
        alpha=0.2
    )
    plt.xlabel("Episodes")
    plt.ylabel("Steps Taken")
    plt.title("Monte Carlo Learning: Steps per Episode")
    plt.legend()

    plt.tight_layout()
    plt.show()