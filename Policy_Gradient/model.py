from tensorflow import keras
from tensorflow.keras import models, layers, optimizers, losses
import numpy as np
import tensorflow as tf


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
            layers.Dense(units=self.hidden_dim, input_dim=self.state_dim, activation="relu", kernel_initializer="he_normal"),
            layers.Dense(units=self.action_dim, input_dim=self.hidden_dim, activation="softmax", kernel_initializer="he_normal")
        ])
        opt = optimizers.Adam(self.lr)
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
        # self.policy_net.fit(x, y, verbose=0)
        self.policy_net.train_on_batch(x, y)
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
        
        
        
class Policy_Gradient_3(Policy_Gradient_2):
    def __init__(self, state_dim, action_dim, lr, gamma, hidden_dim):
        Policy_Gradient_2.__init__(self, state_dim, action_dim, lr, gamma, hidden_dim)
        
    def learn(self):
        episode_length = len(self.state_list)
        discount_rewards = self._discount_and_norm_rewards()
        episode_state = np.vstack(self.state_list)
        y = np.zeros((episode_length, self.action_dim))
        episode_index = np.arange(episode_length)
        episode_action = np.array(self.action_list)
        y[episode_index, episode_action] = discount_rewards

        
        with tf.GradientTape() as tape:
            episode_action_prob = self.policy_net(episode_state)
            cross_entropy1 = tf.losses.categorical_crossentropy(y_true=tf.one_hot(episode_action, self.action_dim),
                                                               y_pred=episode_action_prob)
            loss1 = tf.reduce_mean(cross_entropy1 * discount_rewards)
            # print("loss1 = ", end="")
            # print(loss1.numpy())
            # cross_entropy2 = tf.losses.sparse_categorical_crossentropy(y_true=episode_action,
            #                                                           y_pred=episode_action_prob)
            
            # loss2 = tf.reduce_mean(cross_entropy2 * discount_rewards)
            # print("loss2 = ", end="")
            # print(loss2.numpy())
            # cross_entropy3 = tf.losses.categorical_crossentropy(y_true=y,
            #                                                     y_pred=episode_action_prob)
            # loss3 = tf.reduce_mean(cross_entropy3)
            # print("loss3 = ", end="")
            # print(loss3.numpy())
            
        grads = tape.gradient(loss1, self.policy_net.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.policy_net.trainable_variables))
        self.state_list, self.action_list, self.reward_list = [], [], []
        


class Policy_Gradient_4(Policy_Gradient):
    def __init__(self, state_dim, action_dim, lr, gamma, hidden_dim):
        super(Policy_Gradient_4, self).__init__(state_dim, action_dim, lr, gamma, hidden_dim)
    
    def learn(self):
        # episode_length = len(self.state_list)
        # discount_rewards = self._discount_and_norm_rewards()
        # episode_state = np.vstack(self.state_list)
        # y = np.zeros((episode_length, self.action_dim))
        # episode_index = np.arange(episode_length)
        # episode_action = np.array(self.action_list)
        # y[episode_index, episode_action] = discount_rewards
        
        
        
        episode_length = len(self.state_list)
        discount_rewards = self._discount_and_norm_rewards()
        x = np.vstack(self.state_list)
        y = np.zeros((episode_length, self.action_dim))
        episode_index = np.arange(episode_length)
        episode_action = np.array(self.action_list)
        y[episode_index, episode_action] = discount_rewards
        
        
        episode_action_prob = self.policy_net(x)
        cross_entropy = tf.losses.categorical_crossentropy(y_true=tf.one_hot(episode_action, self.action_dim),
                                                           y_pred=episode_action_prob)
        loss2 = tf.reduce_mean(cross_entropy * discount_rewards)
        print("loss2 = ", end="")
        print(loss2.numpy())
        
        
        loss1 = self.policy_net.train_on_batch(x, y)
        print("loss1 = ", end="")
        print(loss1)
        
        self.state_list, self.action_list, self.reward_list = [], [], []
        


class Policy_Gradient_two_models():
    def __init__(self, state_dim, action_dim, lr, gamma, hidden_dim):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.lr = lr
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.policy_net1 = self._build_net()
        self.policy_net2 = self._build_net()
        self.policy_net1.set_weights(self.policy_net2.get_weights())
        self.opt1 = optimizers.Adam(self.lr)
        self.policy_net1.compile(loss="categorical_crossentropy", optimizer=self.opt1)
        self.opt2 = optimizers.Adam(self.lr)
        self.state_list, self.action_list, self.reward_list = [], [], []

    def _build_net(self):
        policy_net = models.Sequential([
            layers.Dense(units=self.hidden_dim, input_dim=self.state_dim, activation="relu", kernel_initializer="he_normal"),
            layers.Dense(units=self.action_dim, input_dim=self.hidden_dim, activation="softmax", kernel_initializer="he_normal")
        ])
        return policy_net

    def choose_action(self, s):
        s = s[np.newaxis, :]
        prob = self.policy_net1(s).numpy().flatten()
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
        loss1 = self.policy_net1.train_on_batch(x, y)
        print("loss1 = ", end="")
        print(loss1)


        with tf.GradientTape() as tape:
            episode_action_prob = self.policy_net2(x)
            cross_entropy = tf.losses.categorical_crossentropy(y_true=tf.one_hot(episode_action, self.action_dim),
                                                               y_pred=episode_action_prob)
            loss2 = tf.reduce_mean(cross_entropy * discount_rewards)
        grads = tape.gradient(loss2, self.policy_net2.trainable_variables)
        self.opt2.apply_gradients(zip(grads, self.policy_net2.trainable_variables))
        print("loss2 = ", end="")
        print(loss2.numpy())
        
        
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

