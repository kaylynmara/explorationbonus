import numpy as np
import random

class RaceTrack:
    def __init__(self, track_file):
        """ Initialize the racetrack environment"""
        self.track = np.load(track_file)
        self.start_positions = np.argwhere(self.track == 2).tolist()
        self.finish_positions = np.argwhere(self.track == 3)

        self.velocity = (0, 0) #vx, vy
        self.position = None
        self.reset()

    def reset(self):
        """Reset the car to a random start position"""
        self.position = tuple(random.choice(self.start_positions))
        self.velocity = (random.choice([-1, 1]), random.choice([-1, 1])) # small push after a reset :)
        return self.position, self.velocity

    def step(self, action):
        """take a step in the environment"""
        """action = (ax, ay) where ax and ay are in (-1, 0, 1)"""
        ax, ay = action

        if random.random() < .1:
            ax, ay = 0,0 # complies with assignment request for random velocity failure

        vx, vy = self.velocity

        """Not a requirement of the problem, but added bc agent kept getting stuck"""
        if vx == 0 and vy == 0 and ax ==0 and ay ==0:
            ax, ay = random.choice([-1,1]), random.choice([-1,1])
            print("Forced Acceleration To Prevent Getting Stuck")


        vx = min(max(vx + ax, 0), 4)
        vy = min(max(vy + ay, 0), 4)

        x, y = self.position
        new_x, new_y = x - vy, y + vx # neg vy allows for movement to the finish line at the top

        if self.check_collision((x,y), (new_x, new_y)):
            print(f"Collision Detected. Resetting To Start")
            self.reset()
            return self.position, self.velocity, -1, False # collision penalty - continue episode

        if self.track[new_x, new_y] == 3:
            print(f"Finish Line Reached! Position: {new_x, new_y}")
            return (new_x, new_y), (vx, vy), 0, True # no penalty and the episode ends

        self.position = (new_x, new_y)
        self.velocity = (vx, vy)
        return self.position, self.velocity, -1, False # -1 reward per step

    def check_collision(self, old_pos, new_pos):
        """ using Bresenham's algo to check collisions along the way"""
        x1, y1 = old_pos
        x2, y2 = new_pos

        dx = abs(x2-x1)
        dy = abs(y2-y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx -dy

        print(f"Checking Collision From {x1, y1}, {x2, y2}")

        while True:
            if not (0 <= x1 < self.track.shape[0] and 0 <= y1 < self.track.shape[1]):
                print(f"Collision! Out Of Bounds at {(x1, y1)}")
                return True

            if self.track[x1, y1] ==0:
                return True

            if (x1, y1) == (x2, y2):
                break

            e2 = 2*err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        return False


if __name__ == "__main__":
    env = RaceTrack('../maps/track_a.npy')
    print("Start Position:", env.position)

    for _ in range(5):
        action = (random.choice([-1,0,1]), random.choice([-1,0,1]))
        pos, vel, reward, done = env.step(action)
        print(f"Action:{action}, Position:{pos}, Velocity:{vel}, Reward:{reward}, Done:{done}")
        if done:
            break
