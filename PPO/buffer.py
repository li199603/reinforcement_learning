import numpy as np
import scipy.signal

def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class Buffer:
    # Buffer for storing trajectories
    def __init__(self, state_dim, buffer_size, gamma=0.99, lam=0.95, action_dim=None):
        # Buffer initialization
        self.buffer_size = buffer_size
        self.state_buffer = np.zeros((buffer_size, state_dim), dtype=np.float32)
        if action_dim is not None:
            self.action_buffer = np.zeros((buffer_size, action_dim), dtype=np.float32)
        else:
            self.action_buffer = np.zeros(buffer_size, dtype=np.int32)
        self.advantage_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.reward_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.return_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.value_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.probability_buffer = np.zeros(buffer_size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, state, action, reward, value, probability):
        # Append one step of agent-environment interaction
        self.state_buffer[self.pointer] = state
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.probability_buffer[self.pointer] = probability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        assert self.pointer == self.buffer_size # buffer has to be full before you can get
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.state_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.probability_buffer,
        )