from tensorflow.keras import models, layers, optimizers
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
        self._update_param(self)
        opt = optimizers.Adam(learning_rate=self.lr)
        self.policy_net.compile(loss="mse", optimizer=opt)
        

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
            _action = np.argmax(q_values)
            if _action != action:
                print("_action != action")
                print(action)
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

        loss = self.policy_net.train_on_batch(x, y)
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
