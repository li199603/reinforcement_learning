import numpy as np
import pickle

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

class PriorityReplayBuffer(ReplayBuffer):
    def __init__(self, state_dim, action_dim, buffer_size, sample_size):
        super().__init__(state_dim, action_dim, buffer_size, sample_size)
        self.sum_tree = SumTree(buffer_size)
        self.abs_error_upper = 1.0
        self.epsilon = 0.01  # 避免零error
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001
    
    def store(self, state, action, reward, done, next_state):
        priority = np.power(self.abs_error_upper, self.alpha)
        self.sum_tree.add(priority)
        super().store(state, action, reward, done, next_state)
    
    def sample(self):
        if self.count < self.sample_size:
            raise Exception("There are not enough data.")
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        node_indices = np.zeros((self.sample_size,), dtype=np.int32)
        data_indices = np.zeros((self.sample_size,), dtype=np.int32)
        priorities = np.zeros((self.sample_size,), dtype=np.float64)
        segment = self.sum_tree.total() / self.sample_size
        for i in range(self.sample_size):
            a, b = segment * i, segment * (i + 1)
            value = np.random.uniform(a, b)
            node_index, data_index, priority = self.sum_tree.get(value)
            node_indices[i] = node_index
            data_indices[i] = data_index
            priorities[i] = priority
        data_batch = (self.state_buffer[data_indices], 
                      self.action_buffer[data_indices], 
                      self.reward_buffer[data_indices], 
                      self.done_buffer[data_indices], 
                      self.next_state_buffer[data_indices])
        sampling_probabilities = priorities / self.sum_tree.total()
        n_entity = min(self.count, self.buffer_size)
        importance_sampling_weights = np.power(n_entity * sampling_probabilities, -self.beta)
        importance_sampling_weights /= np.max(importance_sampling_weights)
        return data_batch, node_indices, importance_sampling_weights
    
    def errors_update(self, tree_indices, abs_errors):
        abs_errors += self.epsilon
        abs_errors = np.minimum(abs_errors, self.abs_error_upper)
        priorities = np.power(abs_errors, self.alpha)
        for i, p in zip(tree_indices, priorities):
            self.sum_tree.update(i, p)
        

class SumTree(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.n_entries = 0
        self.write = 0 # 记录当前写入的第几个数据
        

    def retrieve(self, idx, value):
        # 从编号idx节点开始检索value值对应的节点
        value = min(self.tree[idx], value)
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if value <= self.tree[left]:
            return self.retrieve(left, value)
        else:
            return self.retrieve(right, value - self.tree[left])

    def update(self, idx, new_value):
        # 更新idx节点值（优先级）
        assert 2 * idx + 1 >= len(self.tree) # 确保更新的节点是叶子节点
        change = new_value - self.tree[idx]
        self.tree[idx] = new_value
        self.propagate_changes(idx, change)

    def propagate_changes(self, idx, change):
        # 迭代更新父节点值
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self.propagate_changes(parent, change)

    def add(self, value):
        idx = self.write + self.capacity - 1
        self.update(idx, value)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def get(self, value):
        idx = self.retrieve(0, value)
        data_idx = idx - self.capacity + 1
        return idx, data_idx, self.tree[idx]

    def total(self):
        return self.tree[0]
    


if __name__ == "__main__":
    buffer = ReplayBuffer(2, 5, 3)
    for i in range(10):
        data = [i, i]
        buffer.store(data)
        if i >= 3:
            print(buffer.sample())
    
        
        