import time
import tensorflow as tf
import tensorflow.keras as keras
from replay_buffer import ReplayBuffer
import numpy as np
import tensorboardX

class DDPG:
    def __init__(self, action_dim, state_dim, action_bound,
                 actor_lr, critic_lr, gamma, tau,
                 buffer_size, batch_size):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_bound = action_bound
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        self.std = 0.3
        self.std_min = 0.01
        self.std_decay = 0.9995
        
        self.actor, self.actor_target = self._build_actor(), self._build_actor()
        self.critic, self.critic_target = self._build_critic(), self._build_critic()
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())
        self.actor.summary()
        self.critic.summary()
        
        self.buffer = ReplayBuffer(2 * state_dim + action_dim + 2,
                                   self.buffer_size,
                                   self.batch_size)
        
    def policy(self, state, add_noise=False):
        state = np.expand_dims(state, 0)
        action = np.squeeze(self.actor(state).numpy(), axis=0)
        if add_noise: 
            noise = np.random.normal(scale=self.std, size=action.shape) * self.action_bound
            action = np.clip(action + noise, -self.action_bound, self.action_bound)
        return action
    
    def learn(self):
        try:
            states, actions, rewards, dones, next_states = self._sample()
        except Exception:
            return
        self._critic_learn(states, actions, rewards, dones, next_states)
        self._actor_learn(states)
        self.std = max(self.std_min, self.std * self.std_decay)
        self._update_target_model()
    
    def _critic_learn(self, states, actions, rewards, dones, next_states):
        next_actions = self.actor_target(next_states)
        next_q = self.critic_target([next_states, next_actions])
        td_target = rewards + (1 - dones) * self.gamma * next_q
        loss = self.critic.fit([states, actions], td_target, verbose=0)

    def _actor_learn(self, states):
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            q = self.critic([states, actions])
            actor_loss = -tf.math.reduce_mean(q)
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.opt_actor.apply_gradients(zip(grads, self.actor.trainable_variables))
    
    def _update_target_model(self):
        def _update(model, target_model):
            weights = model.get_weights()
            target_weights = target_model.get_weights()
            for i in range(len(weights)):
                target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
            target_model.set_weights(target_weights)
        _update(self.actor, self.actor_target)
        _update(self.critic, self.critic_target)
    
    def store_transition(self, state, action, reward, done, next_state):
        data = np.hstack((state, action, [reward, int(done)], next_state))
        self.buffer.store(data)
    
    def _sample(self):
        data = self.buffer.sample()
        states = data[:, :self.state_dim]
        actions = data[:, self.state_dim:self.state_dim+self.action_dim]
        rewards = data[:, -self.state_dim-2:-self.state_dim-1]
        dones = data[:, -self.state_dim-1:-self.state_dim]
        next_states = data[:, -self.state_dim:]
        return states, actions, rewards, dones, next_states
    
    # def _build_actor(self):
    #     inputs = keras.Input((self.state_dim,), name="state_inputs")
    #     x = keras.layers.Dense(20, activation="relu")(inputs)
    #     x = keras.layers.Dense(10, activation="relu")(x)
    #     x = keras.layers.Dense(self.action_dim, activation="tanh")(x)
    #     outputs = keras.layers.multiply([x, self.action_bound])
    #     model = keras.Model(inputs=inputs, outputs=outputs)
    #     self.opt_actor = keras.optimizers.Adam(self.actor_lr)
    #     return model
    
    # def _build_critic(self):
    #     state_inputs = keras.Input((self.state_dim,), name="state_inputs")
    #     action_inputs = keras.Input((self.action_dim,), name="action_inputs")
    #     x1 = keras.layers.Dense(20, activation="relu")(state_inputs)
    #     x2 = keras.layers.Dense(5, activation="relu")(action_inputs)
    #     x = keras.layers.concatenate([x1, x2])
    #     outputs = keras.layers.Dense(1, activation="linear")(x)
    #     model = keras.Model(inputs=[state_inputs, action_inputs], outputs=outputs)
    #     model.compile(loss="mse", optimizer=keras.optimizers.Adam(self.critic_lr))
    #     return model
    
    def _build_actor(self):
        inputs = keras.Input((self.state_dim,), name="state_inputs")
        x = keras.layers.Dense(256, activation="relu")(inputs)
        x = keras.layers.Dense(256, activation="relu")(x)
        x = keras.layers.Dense(self.action_dim, activation="tanh")(x)
        outputs = keras.layers.multiply([x, self.action_bound])
        model = keras.Model(inputs, outputs)
        self.opt_actor = keras.optimizers.Adam(self.actor_lr)
        return model
    
    def _build_critic(self):
        state_inputs = keras.Input((self.state_dim,), name="state_inputs")
        x1 = keras.layers.Dense(16, activation="relu")(state_inputs)
        x1 = keras.layers.Dense(32, activation="relu")(x1)
        action_inputs = keras.Input((self.action_dim,), name="action_inputs")
        x2 = keras.layers.Dense(32, activation="relu")(action_inputs)
        x = keras.layers.Concatenate()([x1, x2])
        x = keras.layers.Dense(256, activation="relu")(x)
        x = keras.layers.Dense(256, activation="relu")(x)
        outputs = keras.layers.Dense(1)(x)
        model = keras.Model([state_inputs, action_inputs], outputs)
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(self.critic_lr))
        return model
        
        
        
        