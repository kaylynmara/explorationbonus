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
epsilon_init = 0.1  # starting rate for epsilon decay
epsilon_dec = .99
epsilon_target = 0.01
decay_episodes = 80
num_runs = 100
map_choice = "maps/track_d.npy"  # pick what map you want to use from track_builder.py
map_name = os.path.splitext(os.path.basename(map_choice))[0]
c = 1

#  assigning tags for visualization tracking
""" If you're doing different iterations of this experiment ie changing variables, maps etc. you'll
need to assign these tags prior to running the experiment"""

training_session = 2  # manually assign this depending on what run you're doing
experiment_name = "off_policy"  # manually assign this depending on what experiment you're running

"""note that these will dynamically change the output of your .pkl file later for when you're doing
visual analysis. So make sure you're descriptive in your naming convention so that you know which visualizations
you're working with later"""

save_filename = f"training_results_{map_name}_{training_session}_{experiment_name}.pkl"
print(f"training results will be saved as: {save_filename}")


Q_all_runs = []
all_rewards = []
all_steps = []

# All Possible Actions (Acceleration Choices)
actions = [(-1, -1), (-1, 0), (-1, 1), (0, 0), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def behavior_policy(state, Q, eps_behavior):
    """epsilon greedy behavior policy with epsilon decay"""
    if state not in Q:
        Q[state] = {a: 0 for a in actions}
    if random.random() < eps_behavior:
        return random.choice(actions)
    else:
        return max(Q[state], key=Q[state].get)


def generate_episode(env, Q, eps_behavior):
    env.reset()
    episode = []
    done = False
    total_reward = 0
    steps = 0

    while not done:
        state = (env.position, env.velocity)
        action = behavior_policy(state, Q, eps_behavior)
        next_state, next_velocity, reward, done = env.step(action)
        episode.append((state, action, reward))
        total_reward += reward
        steps += 1
    return episode, total_reward, steps


def op_monte_carlo_control(seed):
    """off policy mc control w/ importance sampling and eps decay eps_behavior decays over
    episode and eps_target remains fixed"""
    random.seed(seed)
    np.random.seed(seed)
    Q = {}
    C = {}  # cum importance samp weights
    visits = {}
    episode_rewards = []
    episode_steps = []

    for episode in range(episodes):
        eps_behavior = epsilon_init * (epsilon_dec ** episode)
        env = RaceTrack(map_choice)
        episode_data, total_reward, steps = generate_episode(env, Q, eps_behavior)
        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        G = 0
        W = 1.0

        for (state, action, reward) in reversed(episode_data):
            visits[(state, action)] = visits.get((state, action), 0) + 1
            bonus = c / np.sqrt(visits[(state,action)])
            G = gamma * G + reward + bonus

            if (state, action) not in C:
                C[(state, action)] = 0.0
            C[(state, action)] += W

            if state not in Q:
                Q[state] = {a: 0 for a in actions}

            Q[state][action] += (W / C[(state, action)] * (G - Q[state][action]))

            num_actions = len(actions)
            best_action = max(Q[state], key=Q[state].get)
            if action == best_action:
                pi_target = (1-epsilon_target) + epsilon_target / num_actions
                pi_behavior = (1 - eps_behavior) + eps_behavior / num_actions
            else:
                pi_target = epsilon_target / num_actions
                pi_behavior = eps_behavior / num_actions

            if pi_behavior == 0:
                break

            rho = pi_target / pi_behavior
            W *= rho

            if W == 0:
                break

        if episode % 50 == 0:
            print(f"Run{seed} - Episode{episodes} Completed")

    print(f"Off Policy MC Complete for Seed {seed}")
    return Q, episode_rewards, episode_steps


if __name__ == "__main__":
    for run in range(num_runs):
        print(f"Starting Run{run + 1}/{num_runs} with Seed {run}")
        Q, rewards, steps = op_monte_carlo_control(run)
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
