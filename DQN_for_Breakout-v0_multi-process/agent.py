from tensorflow.keras import models, layers, optimizers, Model
import tensorflow as tf
import numpy as np
import time

from tensorflow.python.keras.backend import pool2d
from tensorflow.python.ops.gen_nn_ops import MaxPool

class DQN():
    def __init__(self, featrue_dim, action_dim, lr, gamma, epsilon_max,
                 update_frequency, epsilon_increment):
        self.action_dim = action_dim
        if isinstance(featrue_dim, tuple):
            self.featrue_dim = featrue_dim
        elif isinstance(featrue_dim, int):
            self.featrue_dim = (featrue_dim, )
        self.lr = lr
        self.gamma = gamma
        self.epsilon_max = epsilon_max
        self.epsilon_increment = epsilon_increment
        self.epsilon = 0 if self.epsilon_increment is not None else epsilon_max
        self.update_frequency = update_frequency
        self.learn_step_counter = 0
        self.policy_net = self._build_net()
        self.target_net = self._build_net()
        self._update_param()
        opt = optimizers.Adam(learning_rate=self.lr)
        self.policy_net.compile(loss="mse", optimizer=opt)
        cur_time = time.strftime("DQN_for_Breakout-v0_multi-process/logs/%Y-%m-%d-%Hh%Mm%Ss", time.localtime()) 
        self.summary_writer = tf.summary.create_file_writer(cur_time)
        print(self.policy_net.summary())

    def _build_net(self):
        net = models.Sequential([
            layers.InputLayer(self.featrue_dim),
            layers.Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), padding="same", activation='relu'),
            layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same", activation='relu'),
            layers.MaxPool2D(pool_size=(2,2)),
            layers.Flatten(),
            layers.Dense(units=50, activation="relu"),
            layers.Dense(units=self.action_dim)
        ])
        return net
    
    def choose_action(self, state):
        state = np.expand_dims(state, 0)
        if np.random.uniform() < self.epsilon:
            q_values = self.policy_net(state).numpy().reshape([self.action_dim])
            max_q = np.max(q_values)
            action = np.random.choice(np.where(q_values == max_q)[0])
        else:
            action = np.random.choice(self.action_dim)
        return action

    def _update_param(self):
        self.target_net.set_weights(self.policy_net.get_weights())

    def learn(self, batch_s_pre, batch_action, batch_reward, batch_s_cur, batch_done):
        batch_size = batch_s_pre.shape[0]
        batch_q_cur = self.target_net(batch_s_cur).numpy()
        batch_q_pre = self.policy_net(batch_s_pre).numpy()
        
        x = batch_s_pre
        y = batch_q_pre.copy()
        batch_index = np.arange(batch_size)
        y[batch_index, batch_action] = batch_reward + (1 - batch_done) * self.gamma * np.max(batch_q_cur, axis=1)

        hist = self.policy_net.fit(x, y, verbose=0, batch_size=x.shape[0], epochs=5)
        self.learn_step_counter += 1
        if self.learn_step_counter % self.update_frequency == 0:
            self._update_param()
        self.epsilon = min(self.epsilon + self.epsilon_increment, self.epsilon_max)
        return hist

    def write_scalar(self, name, scalar, step):
        with self.summary_writer.as_default():
            tf.summary.scalar(name, scalar, step)
                
    
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

        
    
    