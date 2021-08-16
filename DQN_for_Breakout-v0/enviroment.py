import gym
import time
import matplotlib.pyplot as plt
import numpy as np


class Env_Breakout():
    def __init__(self, height_rang, width_range, skip_steps=4):
        self.env = gym.make("Breakout-v0")
        self.height_range = height_rang
        self.width_range = width_range
        self.skip_steps = skip_steps
        
    def step(self, action):
        for _ in self.skip_steps:
            state, reward, done, info = self.env.step(action)
            state = self.state_process(state)
        return state, reward, done, info
    
    def get_action_dim(self):
        return self.env.action_space.n
    
    def render(self):
        self.env.render()
    
    def reset(self):
        state = self.env.reset()
        state = self.state_process(state)
        return state
        
    def state_process(self, state):
        h1, h2 = self.height_range
        w1, w2 = self.width_range
        state = state[h1:h2+1, w1:w2+1]
        state = 0.2989 * state[:, :, 0] + 0.5870 * state[:, :, 1] + 0.1140 * state[:, :, 2]
        return state

    def close(self):
        self.env.close()
    


if __name__ == "__main__":
    HEIGHT_RANGE = [32, 194]
    WIDTH_RANGE = [8, 151]
    env = Env_Breakout(HEIGHT_RANGE, WIDTH_RANGE)
    # env.reset()
    # for _ in range(10):
    #     observation, reward, done, info = env.step(1)

    # plt.imshow(observation, cmap="gray")
    # plt.axis('off')
    # plt.show()
    
    # env = Env_Breakout()
    for i in range(5):
        env.reset()
        done = False
        while not done:
            action = env.env.action_space.sample()
            next_state, _, done, _ = env.step(action)
            state = next_state
            env.render()

    env.close()


