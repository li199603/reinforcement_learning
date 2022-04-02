import numpy as np

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, buffer_size, sample_size):
        self.buffer_size = buffer_size
        self.sample_size = sample_size
        self.state_buffer = np.zeros((buffer_size, state_dim))
        self.action_buffer = np.zeros((buffer_size, action_dim))
        self.reward_buffer = np.zeros((buffer_size, 1))
        self.done_buffer = np.zeros((buffer_size, 1))
        self.next_state_buffer = np.zeros((buffer_size, state_dim))
        self.count = 0
    
    def store(self, state, action, reward, done, next_state):
        index = self.count % self.buffer_size
        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.done_buffer[index] = done
        self.next_state_buffer[index] = next_state
        self.count += 1
    
    def sample(self):
        if self.count < self.sample_size:
            raise Exception("There are not enough data.")
        indices = np.random.choice(min(self.count, self.buffer_size),
                                   self.sample_size,
                                   False)
        return self.state_buffer[indices], \
               self.action_buffer[indices], \
               self.reward_buffer[indices], \
               self.done_buffer[indices], \
               self.next_state_buffer[indices]
               
class ReplayBufferNStep(ReplayBuffer):
    def __init__(self, state_dim, action_dim, buffer_size, sample_size, n_step, gamma):
        super().__init__(state_dim, action_dim, buffer_size, sample_size)
        self.n_step = n_step
        self.gamma = gamma
        self.queue = [] # 辅助计算 n_step 的 discounted_reward, look_ahead_done, look_ahead_state
    
    def store(self, state, action, reward, done, next_state):
        self.queue.append([state, action, reward, done, next_state])
        if done:
            discounted_reward = 0
            look_ahead_done = True
            look_ahead_state = next_state
            while len(self.queue) > 0:
                state_i, action_i, reward_i, done_i, next_state_i = self.queue.pop()
                discounted_reward = reward_i + self.gamma * discounted_reward
                super().store(state_i, action_i, discounted_reward, look_ahead_done, look_ahead_state)
        elif len(self.queue) == self.n_step:
            discounted_reward = 0
            look_ahead_done = False
            look_ahead_state = next_state
            for i in range(self.n_step-1, -1, -1):
                discounted_reward = self.queue[i][2] + self.gamma * discounted_reward
            state_i, action_i, reward_i, done_i, next_state_i = self.queue.pop(0)
            super().store(state_i, action_i, discounted_reward, look_ahead_done, look_ahead_state)


if __name__ == "__main__":
    buffer = ReplayBuffer(2, 5, 3)
    for i in range(10):
        data = [i, i]
        buffer.store(data)
        if i >= 3:
            print(buffer.sample())
    
        
        