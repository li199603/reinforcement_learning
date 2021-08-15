import gym
import time
import matplotlib.pyplot as plt
import numpy as np


class Env_Breakout():
    def __init__(self, height_rang=[32, 194], width_range=[8, 151]):
        self.env = gym.make("Breakout-v0")
        self.height_range = height_rang
        self.width_range = width_range
        
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        h1, h2 = self.height_range
        w1, w2 = self.width_range
        observation = observation[h1:h2+1, w1:w2+1]
        gray = 0.2989 * observation[:, :, 0] + 0.5870 * observation[:, :, 1] + 0.1140 * observation[:, :, 2]
        return gray, reward, done, info
    
    def render(self):
        self.env.render()
    
    def reset(self):
        self.env.reset()
    


if __name__ == "__main__":
        env = Env_Breakout()
        env.reset()
        for _ in range(100):
            observation, reward, done, info = env.step(2)

        plt.imshow(observation, cmap="gray")
        plt.axis('off')
        plt.show()


