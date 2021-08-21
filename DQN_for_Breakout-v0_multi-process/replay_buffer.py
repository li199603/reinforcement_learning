from copy import Error
import numpy as np

class Replay_Buffer:
    def __init__(self, featrue_dim, buffer_size=100000, frame_history_len=3, batch_size=64):
        if isinstance(featrue_dim, tuple):
            self.featrue_dim = featrue_dim
        elif isinstance(featrue_dim, int):
            self.featrue_dim = (featrue_dim, )
        else:
            raise Error
        self.buffer_size = buffer_size
        # 一个样本数据中，states包含的帧数。这样的特征能够反映历史信息，即能够让模型捕捉游戏中物体运动的方向和速度
        self.frame_history_len = frame_history_len
        self.batch_size = batch_size
        self.buffer_s_pre = np.zeros((self.buffer_size, ) + self.featrue_dim)
        self.buffer_s_cur = np.zeros((self.buffer_size, ) + self.featrue_dim)
        self.buffer_action = np.zeros((self.buffer_size, ), dtype=np.int8)
        self.buffer_reward = np.zeros((self.buffer_size, ))
        self.buffer_done = np.zeros((self.buffer_size, ), dtype=np.int8)
        self.buffer_counter = 0

    def store_data(self, s_pre, action, reward, s_cur, done):
        index = self.buffer_counter % self.buffer_size
        self.buffer_s_pre[index] = s_pre
        self.buffer_action[index] = int(action)
        self.buffer_reward[index] = reward
        self.buffer_s_cur[index] = s_cur
        self.buffer_done[index] = int(done)
        self.buffer_counter += 1
    
    def sample_batch_data(self):
        if self.buffer_counter < self.batch_size:
            raise Error
        sample_index = np.random.choice(min(self.buffer_size, self.buffer_counter), size=self.batch_size)
        batch_s_pre = self.buffer_s_pre[sample_index]
        batch_action = self.buffer_action[sample_index]
        batch_reward = self.buffer_reward[sample_index]
        batch_s_cur = self.buffer_s_cur[sample_index]
        batch_done = self.buffer_done[sample_index]
        
        return batch_s_pre, batch_action, batch_reward, batch_s_cur, batch_done
