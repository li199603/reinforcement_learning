import gym
import matplotlib.pyplot as plt
import numpy as np
import cv2
import breakout_wrapper


class Env_Breakout():
    def __init__(self, height_range=[32, 194], width_range=[8, 151], 
                 height_resize=84, width_resize=84, skip_steps=3):
        self.env = breakout_wrapper.wrap(gym.make('Breakout-v0'))
        self.height_range = height_range
        self.width_range = width_range
        self.height_resize = height_resize
        self.width_resize = width_resize
        self.skip_steps = skip_steps
        
    def step(self, action):
        total_state = np.zeros((self.height_resize, self.width_resize, 3))
        total_reward = 0
        total_done = False
        for i in range(self.skip_steps):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            total_done = total_done or done
            if i == 0:
                state = self.state_process(state)
                total_state[:, :, 0] = state
            if i == self.skip_steps // 2:
                state = self.state_process(state)
                total_state[:, :, 1] = state
            if i == self.skip_steps-1:
                state = self.state_process(state)
                total_state[:, :, 2] = state
        return total_state, total_reward, total_done, info
    
    def get_action_dim(self):
        return self.env.action_space.n
    
    def get_featrues_dim(self):
        state = self.reset()
        return tuple(state.shape)
    
    def render(self):
        self.env.render()
    
    def reset(self):
        state = self.env.reset()
        state = self.state_process(state)
        state = np.repeat(np.expand_dims(state, 2), 3, 2)
        return state
        
    def state_process(self, state):
        h1, h2 = self.height_range
        w1, w2 = self.width_range
        state = state[h1:h2+1, w1:w2+1]
        state = 0.2989 * state[:, :, 0] + 0.5870 * state[:, :, 1] + 0.1140 * state[:, :, 2]
        state = cv2.resize(state, (84, 84), interpolation=cv2.INTER_LINEAR)
        return state

    def close(self):
        self.env.close()
    


if __name__ == "__main__":
    # HEIGHT_RANGE = [32, 194]
    # WIDTH_RANGE = [8, 151]
    # env = Env_Breakout(HEIGHT_RANGE, WIDTH_RANGE)
    # env.reset()
    # for _ in range(1):
    #     observation, reward, done, info = env.step(3)

    # plt.imshow(observation, cmap="gray")
    # plt.axis('off')
    # plt.show()
    
    env = Env_Breakout()
    for i in range(5):
        env.reset()
        done = False
        while not done:
            # action = env.env.action_space.sample()
            action = int(input())
            next_state, _, done, _ = env.step(action)
            state = next_state
            env.render()

    env.close()


