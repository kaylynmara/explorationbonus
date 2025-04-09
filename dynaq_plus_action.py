import numpy as np
import os
import random
from Env.environment import RaceTrack
import matplotlib
import matplotlib.pyplot as plt
import pickle

matplotlib.use('TkAgg')

episodes = 100
alpha = 0.1
gamma = .9  # discount factor - change this as you see fit
epsilon = .1  # exploration rate for E-Greedy policy. Change as you see fit
kappa = 0.001 # tunable coefficient for exploratiomn bonus - set this higher for more exploring
num_runs = 100
planning_steps = 5

map_choice = "maps/track_a.npy"  # pick what map you want to use from track_builder.py
map_name = os.path.splitext(os.path.basename(map_choice))[0]

#  assigning tags for visualization tracking
""" If you're doing different iterations of this experiment ie changing variables, maps etc. you'll
need to assign these tags prior to running the experiment"""

training_session = 1  # manually assign this depending on what run you're doing
experiment_name = "baseline"  # manually assign this depending on what experiment you're running

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


def epsilon_greedy_policy(state, Q, tau):
    """select an action using epsilon greedy policy - modified to incorporate exploration bonus"""
    if state not in Q:
        Q[state] = {a: 0 for a in actions}
    if random.random() < epsilon:
        return random.choice(actions)  # explore with prob epsilon

    augmented_Q = {}
    for a in actions:
        tau_val = tau.get((state, a),1) # solve sqrt(0)
        bonus = kappa * np.sqrt(tau_val)
        augmented_Q[a] = Q[state][a] + bonus

    return max(augmented_Q, key=augmented_Q.get)

def dyna_q_control(seed):
    """exploration bonus implemented in action"""
    random.seed(seed)
    np.random.seed(seed)

    Q = {}
    model = {}
    tau = {} # time since state-action pair was last tried
    episode_rewards = []
    episode_steps = []

    for ep in range(episodes):
        env = RaceTrack(map_choice)
        env.reset()
        total_reward = 0
        steps = 0

        while True:
            state = (env.position, env.velocity)
            action = epsilon_greedy_policy(state, Q, tau)
            next_pos, next_vel, reward, done = env.step(action)
            next_state = (next_pos, next_vel)

            # Initializing Q table entries
            if state not in Q:
                Q[state] = {a: 0 for a in actions}
            if next_state not in Q:
                Q[next_state] = {a: 0 for a in actions}

            Q[state][action] += alpha * (reward + gamma * max(Q[next_state].values()) - Q[state][action])

            # save the trans. in the model
            model[(state, action)] = (next_state, reward)

            # planning steps
            for _ in range(planning_steps):
                s, a = random.choice(list(model.keys()))
                s_prime, r = model[(s,a)]

                if s not in Q:
                    Q[s] = {act: 0 for act in actions}
                if s_prime not in Q:
                    Q[s_prime] = {act: 0 for act in actions}

                Q[s][a] += alpha * (r + gamma * max(Q[s_prime].values()) - Q[s][a])

            total_reward += reward
            steps += 1
            if done:
                break

        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        if ep % 10 == 0:
            print(f"Seed{seed}: Episode {ep + 1}/{episodes} complete")

    return Q, episode_rewards, episode_steps

if __name__== "__main__":
    for run in range(num_runs):
        print(f"Starting Run {run + 1}/{num_runs} with Seed {run}")
        Q, rewards, steps = dyna_q_control(run)

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

    print(f"Training Results saved to {save_filename}")

    # compute average rewards and steps across multiple runs
    avg_rewards = np.mean(all_rewards, axis=0)
    avg_steps = np.mean(all_steps, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    std_steps = np.std(all_steps, axis=0)
    cumulative_rewards = np.cumsum(-avg_rewards)
    """ we flip cumulative_rewards negative so that the curve rises with better performing agents"""

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
    plt.title("Dyna-Q Reward per Episode")
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
    plt.title("Dyna-Q Steps Per Epsiode")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Cumulative reward plot (like the one on page 167 of the txtbk)
    plt.figure(figsize= (6, 4))
    plt.plot(cumulative_rewards, color="green")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Rewards")
    plt.title("Cumulative Reward Over Time")
    plt.tight_layout()
    plt.show()
