import numpy as np
import matplotlib
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
matplotlib.use("TkAgg")

"""your .pkl files will be used to develop the vis. You'll need to match the proper
experiment names and training sessions as well as the track/map you used"""
training_session = 1 # change this to match which training session you'd like to visualize
experiment_name = "baseline" # change this to match the experiment you want to visualize
map_name = "track_a" # change this to the track you want to visualize

training_file = f"training_results_{map_name}_{training_session}_{experiment_name}.pkl"

if not os.path.exists(training_file):
    raise FileNotFoundError(f"Training file {training_file} not found. Check your path and try again")

print(f"Loading training data from: {training_file}")

with open(training_file, "rb") as f:
    data = pickle.load(f)

all_rewards = data["all_rewards"]
all_steps = data["all_steps"]
Q_all_runs = data["Q_all_runs"]

# load track size
track_map = np.load(f"maps/{map_name}.npy")
track_height, track_width = track_map.shape
print(f"Track Map Size: {track_map.shape}")

# Heat Map For State Visit Frequency
state_visit_counts = {}

for Q in Q_all_runs:
    for state in Q:
        if state not in state_visit_counts:
            state_visit_counts[state] = 0
        state_visit_counts[state] += 1

heatmap_data = np.zeros((track_height, track_width))
finish_line_coords = []
start_line_coords = []

for x in range(track_height):
    for y in range(track_width):
        if track_map[x, y] == 3:
            finish_line_coords.append((x, y))

for x in range(track_height):
    for y in range(track_width):
        if track_map[x, y] == 2:
            start_line_coords.append((x, y))

for(pos, vel), count in state_visit_counts.items():
    x, y = pos
    if 0 <= x < track_height and 0 <= y < track_width:
        heatmap_data[x, y] += count

visited_x = [int(state[0][0]) for state in state_visit_counts.keys()]
visited_y = [int(state[0][1]) for state in state_visit_counts.keys()]

if visited_x and visited_y:
    min_x = int(min(visited_x, default=0))
    max_x = int(max(visited_x, default=track_height - 1))
    min_y = int(min(visited_y, default=0))
    max_y = int(max(visited_y, default=track_width - 1))
    heatmap_data = heatmap_data[min_x:max_x+1, min_y:max_y+1]
    finish_line_coords = [(x - min_x, y - min_y) for (x, y) in finish_line_coords]
    start_line_coords = [(x - min_x, y - min_y) for (x, y) in start_line_coords]
else:
    print("Warning: No visited states detected. Using full track")

plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, cmap="coolwarm", annot=False)

for x, y in finish_line_coords:
    plt.text(y + 0.5, x + 0.5, "#", fontsize=12, ha="center", va="center", color="white", fontweight="bold")

for x, y in start_line_coords:
    plt.text(y + 0.5, x + 0.5, "S", fontsize=12, ha="center", va="center", color="white", fontweight="bold")


finish_patch = mpatches.Patch(color="white", label="### = Finish Line")
start_patch = mpatches.Patch(color="white", label="SSS = Start Line")
plt.legend(handles=[finish_patch, start_patch], loc='best', fontsize=10, frameon=True)

plt.title(f"State Visit Frequency Heatmap\nSession {training_session} - {experiment_name} - {map_name}")
plt.xlabel("Track Width (Y)")
plt.ylabel("Track Height (X)")
plt.show()

# Monte Carlo Policy Visualization
Q_optimal = {}

for Q in Q_all_runs:
    for state, action_values in Q.items():
        if state not in Q_optimal:
            Q_optimal[state] = {a: [] for a in action_values.keys()}
        for action, value in action_values.items():
            Q_optimal[state][action].append(value)

for state in Q_optimal:
    for action in Q_optimal[state]:
        Q_optimal[state][action] = np.mean(Q_optimal[state][action])


visited_x = [int(state[0][0]) for state in Q_optimal.keys()]
visited_y = [int(state[0][1]) for state in Q_optimal.keys()]

if visited_x and visited_y:
    min_x = max(0, min(visited_x))
    max_x = min(track_height - 1, max(visited_x))
    min_y = max(0, min(visited_y))
    max_y = min(track_width - 1, max(visited_y))

    cropped_track_map = track_map[min_x:max_x + 1, min_y:max_y + 1]
else:
    print("Warning: No visited states detected. Using full track.")
    cropped_track_map = track_map

plt.figure(figsize=(8, 6))
sns.heatmap(cropped_track_map, cmap="gray", alpha=0.3, annot=False, cbar=False, square=True)

for i, state in enumerate(Q_optimal):
    if i % 3 == 0:
        pos, vel = state
        x, y = pos
        best_action = max(Q_optimal[state], key=Q_optimal[state].get)  # Best action

        dx, dy = best_action
        cropped_x, cropped_y = x - min_x, y - min_y

        if (0 <= cropped_x < cropped_track_map.shape[0]) and (0 <= cropped_y < cropped_track_map.shape[1]):
            confidence = Q_optimal[state][best_action]
            color = "red" if confidence > -1 else "blue"
            plt.arrow(cropped_y + 0.5, cropped_x + 0.5, dy * 0.3, -dx * 0.3, head_width=0.2, color=color)

plt.title("Optimized Monte Carlo Policy Visualization")
plt.xlabel("Track Width (Y)")
plt.ylabel("Track Height (X)")

legend_patches = [
    mpatches.Patch(color="red", label="High Confidence Actions"),
    mpatches.Patch(color="blue", label="Low Confidence Actions"),
]
plt.legend(handles=legend_patches, loc="best", fontsize=10, frameon=True)
plt.show()








