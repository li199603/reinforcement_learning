from tensorflow.keras import models, layers, optimizers, Model
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.engine.input_layer import InputLayer
import time

def print_run_time(func):  
    def wrapper(*args, **kw):  
        local_time = time.time()  
        res = func(*args, **kw) 
        print("current Function [%s] run time is %.2f" % (func.__name__ ,time.time() - local_time))
        return res 
    return wrapper 

class DQN():
    def __init__(self, featrue_shape, action_dim, lr, gamma, epsilon_max,
                 hidden_dim, buffer_size, batch_size, update_frequency,
                 epsilon_increment):
        self.action_dim = action_dim
        self.featrue_shape = featrue_shape
        self.lr = lr
        self.gamma = gamma
        self.epsilon_max = epsilon_max
        self.epsilon_increment = epsilon_increment
        self.epsilon = 0 if self.epsilon_increment is not None else epsilon_max
        self.hidden_dim = hidden_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.buffer_s_pre = np.zeros((self.buffer_size, self.featrue_shape[0], self.featrue_shape[1], 1))
        self.buffer_s_cur = np.zeros((self.buffer_size, self.featrue_shape[0], self.featrue_shape[1], 1))
        self.buffer_action = np.zeros((self.buffer_size, ))
        self.buffer_reward = np.zeros((self.buffer_size, ))
        self.buffer_counter = 0
        self.learn_step_counter = 0
        self.policy_net = self._build_net()
        self.target_net = self._build_net()
        self._update_param()
        opt = optimizers.Adam(learning_rate=self.lr)
        self.policy_net.compile(loss="mse", optimizer=opt)
        

    def _build_net(self):
        net = models.Sequential([
            layers.InputLayer(input_shape=(self.featrue_shape[0], self.featrue_shape[1], 1)),
            # layers.Conv2D(32, (10, 10), input_shape=(self.featrue_shape[0], self.featrue_shape[1], 1), activation='relu'),
            # layers.Conv2D(8, (5, 5), activation='relu'),
            # layers.MaxPool2D(pool_size=(5, 5)),
            layers.Flatten(),
            layers.Dense(units=self.hidden_dim, activation="relu"),
            layers.Dense(units=self.action_dim)
        ])
        return net

    # @print_run_time
    def choose_action(self, s):
        s = s[np.newaxis, :, :, np.newaxis]
        if np.random.uniform() < self.epsilon:
            q_values = self.policy_net(s).numpy().reshape([self.action_dim])
            max_q = np.max(q_values)
            action = np.random.choice(np.where(q_values == max_q)[0])
        else:
            action = np.random.choice(self.action_dim)
        return action

    # @print_run_time
    def _update_param(self):
        self.target_net.set_weights(self.policy_net.get_weights())

    # @print_run_time
    def learn(self):
        if self.buffer_counter < self.batch_size:
            return
        sample_index = np.random.choice(min(self.buffer_size, self.buffer_counter), size=self.batch_size)
        batch_q_cur = self.target_net(self.buffer_s_cur[sample_index]).numpy()
        batch_q_pre = self.target_net(self.buffer_s_pre[sample_index]).numpy()
        batch_action = self.buffer_action[sample_index].astype(int)
        batch_reward = self.buffer_reward[sample_index]
        
        x = self.buffer_s_pre[sample_index]
        y = batch_q_pre.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        y[batch_index, batch_action] = batch_reward + self.gamma * np.max(batch_q_cur, axis=1)

        loss = self.policy_net.fit(x, y, verbose=0)
        self.learn_step_counter += 1
        if self.learn_step_counter % self.update_frequency == 0:
            self._update_param()
        self.epsilon = min(self.epsilon + self.epsilon_increment, self.epsilon_max)
        return loss

    # @print_run_time
    def store_data(self, s_pre, action, reward, s_cur):
        index = self.buffer_counter % self.buffer_size
        self.buffer_s_pre[index, :] = s_pre[:, :, np.newaxis]
        self.buffer_action[index] = action
        self.buffer_reward[index] = reward
        self.buffer_s_cur[index, :] = s_cur[:, :, np.newaxis]
        self.buffer_counter += 1

    def save(self, path):
        self.policy_net.save_weights(path)

    def load(self, path):
        self.policy_net.load_weights(path)


class Dueling_DQN(DQN):
    def _build_net(self):
        x = tf.keras.Input((self.state_dim,))
        a = layers.Dense(units=self.hidden_dim, activation="relu")(x)
        state_value = layers.Dense(units=1, activation="tanh")(a)
        action_advantage = layers.Dense(units=self.action_dim, activation="linear")(a)
        y = state_value + (action_advantage - tf.reduce_mean(action_advantage, keepdims=True))
        net = Model(inputs=x, outputs=y)
        return net

        
    
    