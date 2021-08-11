from tensorflow import keras
from tensorflow.keras import models, layers, optimizers, losses
import numpy as np
import tensorflow as tf

tf.keras.backend.set_floatx('float64')
class Policy_Gradient():
    def __init__(self, state_dim, action_dim, lr, gamma, hidden_dim):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.lr = lr
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.policy_net = self._build_net()
        self.state_list, self.action_list, self.reward_list = [], [], []

    def _build_net(self):
        policy_net = models.Sequential([
            layers.Dense(units=self.hidden_dim, input_dim=self.state_dim, activation="relu"),
            layers.Dense(units=self.action_dim, input_dim=self.hidden_dim, activation="softmax")
        ])
        opt = optimizers.Adam(learning_rate=self.lr)
        policy_net.compile(loss="categorical_crossentropy", optimizer=opt)
        return policy_net

    def choose_action(self, s):
        s = s[np.newaxis, :]
        prob = self.policy_net(s).numpy().flatten()
        action = np.random.choice(self.action_dim, 1, p=prob)[0]
        return action

    def learn(self):
        episode_length = len(self.state_list)
        discount_rewards = self._discount_and_norm_rewards()
        x = np.vstack(self.state_list)
        y = np.zeros((episode_length, self.action_dim))
        episode_index = np.arange(episode_length)
        episode_action = np.array(self.action_list)
        y[episode_index, episode_action] = discount_rewards
        self.policy_net.fit(x, y, verbose=0)
        self.state_list, self.action_list, self.reward_list = [], [], []

    def store_data(self, state, action, reward):
        self.state_list.append(state)
        self.action_list.append(action)
        self.reward_list.append(reward)

    def _discount_and_norm_rewards(self):
        discounted_rewards = np.zeros(len(self.reward_list))
        running_add = 0
        for t in reversed(range(0, len(self.reward_list))):
            running_add = self.reward_list[t] + self.gamma * running_add
            discounted_rewards[t] = running_add
        # 标准化
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        return discounted_rewards


    def save(self, path):
        pass

    def load(self, path):
        pass


class Policy_Gradient_2(Policy_Gradient):
    def __init__(self, state_dim, action_dim, lr, gamma, hidden_dim):
        Policy_Gradient.__init__(self, state_dim, action_dim, lr, gamma, hidden_dim)
        self.opt = optimizers.Adam(self.lr)
    
    def _build_net(self):
        policy_net = models.Sequential([
            layers.Dense(units=self.hidden_dim, input_dim=self.state_dim, activation="relu"),
            layers.Dense(units=self.action_dim, input_dim=self.hidden_dim, activation="softmax")
        ])
        return policy_net

    def choose_action(self, s):
        s = s[np.newaxis, :]
        prob = self.policy_net(s).numpy().flatten()
        action = np.random.choice(self.action_dim, 1, p=prob)[0]
        return action
    
    def learn(self):
        episode_state = np.vstack(self.state_list)
        episode_action = np.array(self.action_list)
        discount_rewards = self._discount_and_norm_rewards()
        with tf.GradientTape() as tape:
            episode_action_prob = self.policy_net(episode_state)
            cross_entropy = tf.losses.sparse_categorical_crossentropy(y_true=episode_action,
                                                                      y_pred=episode_action_prob)
            loss = tf.reduce_mean(cross_entropy * discount_rewards)
        grads = tape.gradient(loss, self.policy_net.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.policy_net.trainable_variables))
        self.state_list, self.action_list, self.reward_list = [], [], []
        
        
        
class Policy_Gradient_3(Policy_Gradient):
    def __init__(self, state_dim, action_dim, lr, gamma, hidden_dim):
        Policy_Gradient.__init__(self, state_dim, action_dim, lr, gamma, hidden_dim)
        self.opt = optimizers.Adam(self.lr)
    
    def _build_net(self):
        policy_net = models.Sequential([
            layers.Dense(units=self.hidden_dim, input_dim=self.state_dim, activation="relu"),
            layers.Dense(units=self.action_dim, input_dim=self.hidden_dim, activation="softmax")
        ])
        return policy_net

    def choose_action(self, s):
        s = s[np.newaxis, :]
        prob = self.policy_net(s).numpy().flatten()
        action = np.random.choice(self.action_dim, 1, p=prob)[0]
        return action
    
    def learn(self):
        episode_length = len(self.state_list)
        discount_rewards = self._discount_and_norm_rewards()
        episode_state = np.vstack(self.state_list)
        episode_action = np.array(self.action_list)
        episode_index = np.arange(episode_length)
        y = np.zeros((episode_length, self.action_dim))
        y[episode_index, episode_action] = discount_rewards
        
        with tf.GradientTape() as tape:
            episode_action_prob = self.policy_net(episode_state)
            cross_entropy = tf.losses.categorical_crossentropy(y_true=episode_action,
                                                               y_pred=episode_action_prob)                                                    
            loss = tf.reduce_mean(cross_entropy * discount_rewards)
        grads = tape.gradient(loss, self.policy_net.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.policy_net.trainable_variables))
        self.state_list, self.action_list, self.reward_list = [], [], []