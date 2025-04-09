# This builds four racetracks. The first are simple, the second two are complex
# You can modify either of the two tracks as you see fit
# Upon execution, it should save the two tracks and display the result

import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure the 'maps' directory exists
os.makedirs("../maps", exist_ok=True)


def create_track_a():
    """Creates a simple racetrack with a clear start and finish line."""
    track = np.zeros((10, 15), dtype=int)
    track[3:9, 2:13] = 1  # Track surface
    track[8, 3:6] = 2  # Start line
    track[3, 9:12] = 3  # Finish line
    return track


def create_track_b():
    """Creates a tougher racetrack for the agent."""
    track = np.zeros((12, 18), dtype=int)
    track[2:11, 3:15] = 1  # track surface
    track[10, 4:7] = 2  # Start line
    track[2, 10:13] = 3  # Finish line
    return track


def create_track_c():
    """Creates a more complex racetrack with sharp turns to further challenge the agent"""
    track = np.zeros((16, 22), dtype=int)
    track[4:13, 3:19] = 1  # track surface
    track[8, 5:11] = 0  # passage for agent to navigate
    track[5:8, 13:15] = 0  # sharp turn
    track[12, 4:7] = 2  # start line
    track[4, 16:19] = 3  # finish line
    return track

def create_track_d():
    """another complex track with maze patterns"""
    track = np.zeros((21, 26), dtype=int)
    track[6:16, 4:23] = 1  # track surface
    track[11, 6:11] = 0  # blocked area to force detour
    track[6:10, 15:19] = 0  # force another obstacle
    track[15, 5:8] = 2  # start line
    track[6, 19:23] = 3  # finish line
    return track


def load_and_display_track(track_path):
    track = np.load(track_path)

    import matplotlib
    matplotlib.use("TkAgg")

    plt.imshow(track, cmap="gray")
    plt.colorbar()
    plt.title(f"Track: {track_path}")
    plt.show()


if __name__ == "__main__":
    # Create and save the track files
    track_a = create_track_a()
    track_b = create_track_b()
    track_c = create_track_c()
    track_d = create_track_d()

    np.save("../maps/track_a.npy", track_a)
    np.save("../maps/track_b.npy", track_b)
    np.save("../maps/track_c.npy", track_c)
    np.save("../maps/track_d.npy", track_d)

    print("Track maps saved successfully!")

    # Display the tracks
    load_and_display_track("../maps/track_a.npy")
    load_and_display_track("../maps/track_b.npy")
    load_and_display_track("../maps/track_c.npy")
    load_and_display_track("../maps/track_d.npy")
