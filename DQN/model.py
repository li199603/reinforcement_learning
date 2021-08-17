from tensorflow.keras import models, layers, optimizers, Model
import tensorflow as tf
import numpy as np


class DQN():
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon_max,
                 hidden_dim, buffer_size, batch_size, update_frequency,
                 epsilon_increment):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon_max = epsilon_max
        self.epsilon_increment = epsilon_increment
        self.epsilon = 0 if self.epsilon_increment is not None else epsilon_max
        self.hidden_dim = hidden_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.buffer = np.zeros((self.buffer_size, self.state_dim*2+2))
        self.buffer_counter = 0
        self.learn_step_counter = 0
        self.policy_net = self._build_net()
        self.target_net = self._build_net()
        self._update_param()
        opt = optimizers.Adam(learning_rate=self.lr)
        self.policy_net.compile(loss="mse", optimizer=opt)
        print(self.policy_net.summary())
        

    def _build_net(self):
        net = models.Sequential([
            layers.Dense(units=self.hidden_dim, input_dim=self.state_dim, activation="relu"),
            layers.Dense(units=self.action_dim, input_dim=self.hidden_dim)
        ])
        return net

    def choose_action(self, s):
        s = s[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            q_values = self.policy_net.predict(s).reshape([self.action_dim])
            max_q = np.max(q_values)
            action = np.random.choice(np.where(q_values == max_q)[0])
        else:
            action = np.random.choice(self.action_dim)
        return action

    def _update_param(self):
        self.target_net.set_weights(self.policy_net.get_weights())

    def learn(self):
        if self.buffer_counter < self.batch_size:
            return
        sample_index = np.random.choice(min(self.buffer_size, self.buffer_counter), size=self.batch_size)
        batch_data = self.buffer[sample_index, :]

        batch_action = batch_data[:, self.state_dim].astype(int)
        batch_reward = batch_data[:, self.state_dim+1]
        batch_q_cur = self.target_net.predict(batch_data[:, -self.state_dim:])
        batch_q_pre = self.policy_net.predict(batch_data[:, :self.state_dim])

        x = batch_data[:, :self.state_dim]
        y = batch_q_pre.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        y[batch_index, batch_action] = batch_reward + self.gamma * np.max(batch_q_cur, axis=1)

        loss = self.policy_net.fit(x, y, verbose=0)
        self.learn_step_counter += 1
        if self.learn_step_counter % self.update_frequency == 0:
            self._update_param()
        self.epsilon = min(self.epsilon + self.epsilon_increment, self.epsilon_max)
        return loss


    def store_data(self, s_pre, action, reward, s_cur):
        data = np.hstack((s_pre, [action, reward], s_cur))
        index = self.buffer_counter % self.buffer_size
        self.buffer[index, :] = data
        self.buffer_counter += 1

    def save(self, path):
        pass

    def load(self, path):
        pass

class DQN_2():
    def __init__(self, featrue_dim, action_dim, lr, gamma, epsilon_max,
                 hidden_dim, buffer_size, batch_size, update_frequency,
                 epsilon_increment):
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
        self.hidden_dim = hidden_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.buffer_s_pre = np.zeros((self.buffer_size, ) + self.featrue_dim)
        self.buffer_s_cur = np.zeros((self.buffer_size, ) + self.featrue_dim)
        self.buffer_action = np.zeros((self.buffer_size, ))
        self.buffer_reward = np.zeros((self.buffer_size, ))
        self.buffer_counter = 0
        self.learn_step_counter = 0
        self.policy_net = self._build_net()
        self.target_net = self._build_net()
        self._update_param()
        opt = optimizers.Adam(learning_rate=self.lr)
        self.policy_net.compile(loss="mse", optimizer=opt)
        print(self.policy_net.summary())

    def _build_net(self):
        net = models.Sequential([
            layers.InputLayer(self.featrue_dim),
            layers.Dense(units=self.hidden_dim, activation="relu"),
            layers.Dense(units=self.action_dim)
        ])
        return net

    # @print_run_time
    def choose_action(self, state):
        state = np.expand_dims(state, 0)
        if np.random.uniform() < self.epsilon:
            q_values = self.policy_net(state).numpy().reshape([self.action_dim])
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
        self.buffer_s_pre[index] = s_pre
        self.buffer_action[index] = action
        self.buffer_reward[index] = reward
        self.buffer_s_cur[index] = s_cur
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
