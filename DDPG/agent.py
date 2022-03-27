import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
from replay_buffer import ReplayBuffer
import os

class DDPG:
    def __init__(self,
                 action_dim,
                 state_dim,
                 action_bound,
                 actor_lr,
                 critic_lr,
                 gamma, tau,
                 buffer_size,
                 batch_size,
                 summary_writer):
        
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_bound = action_bound
        self.gamma = gamma
        self.tau = tau
        
        self.actor, self.actor_target = self._build_actor(), self._build_actor()
        self.critic, self.critic_target = self._build_critic(), self._build_critic()
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())
        self.critic_opt = keras.optimizers.Adam(critic_lr)
        self.actor_opt = keras.optimizers.Adam(actor_lr)
        self.actor.summary()
        self.critic.summary()
        
        self.buffer = ReplayBuffer(2*state_dim+action_dim+2, buffer_size, batch_size)
        self.summary_writer = summary_writer
        self.learn_step_count = 0
    
    def policy(self, state):
        state = np.expand_dims(state, 0)
        action = np.squeeze(self.actor(state).numpy(), axis=0)
        return action
    
    def learn(self):
        try:
            states, actions, rewards, dones, next_states = self._sample_transition()
        except Exception:
            return
        critic_loss = self._critic_learn(states, actions, rewards, dones, next_states)
        actor_loss = self._actor_learn(states)
        self.summary_writer.add_scalar("agent/critic_loss", critic_loss.numpy(), self.learn_step_count)
        self.summary_writer.add_scalar("agent/actor_loss", actor_loss.numpy(), self.learn_step_count)
        self._update_target_model()
        self.learn_step_count += 1
        
    @tf.function
    def _critic_learn(self, states, actions, rewards, dones, next_states):
        with tf.GradientTape() as tape:
            next_actions = self.actor_target(next_states)
            next_q = self.critic_target([next_states, next_actions])
            td_target = rewards + (1 - dones) * self.gamma * next_q
            cur_q = self.critic([states, actions])
            loss = tf.math.reduce_mean(tf.math.square(cur_q - td_target))
        grad = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grad, self.critic.trainable_variables))
        return loss
    
    @tf.function
    def _actor_learn(self, states):
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            q = self.critic([states, actions])
            loss = -tf.math.reduce_mean(q)
        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))
        return loss
        
    @tf.function  
    def _update_target_model(self):
        def _update(model, target_model):
            weights = model.variables
            target_weights = target_model.variables
            for (w1, w2) in zip(weights, target_weights):
                w2.assign((1 - self.tau) * w2 + self.tau * w1)
        _update(self.actor, self.actor_target)
        _update(self.critic, self.critic_target)
    
    def store_transition(self, state, action, reward, done, next_state):
        data = np.hstack((state, action, [reward, int(done)], next_state))
        self.buffer.store(data)
    
    def _sample_transition(self):
        data = self.buffer.sample()
        states = data[:, :self.state_dim]
        actions = data[:, self.state_dim:self.state_dim+self.action_dim]
        rewards = data[:, -self.state_dim-2:-self.state_dim-1]
        dones = data[:, -self.state_dim-1:-self.state_dim]
        next_states = data[:, -self.state_dim:]
        
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states)
        return states, actions, rewards, dones, next_states
    
    def _build_actor(self):
        inputs = layers.Input((self.state_dim,), name="state_inputs")
        x = layers.Dense(256, activation="relu")(inputs)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(self.action_dim, activation="tanh")(x)
        outputs = layers.multiply([x, self.action_bound])
        model = keras.Model(inputs, outputs)
        return model
    
    def _build_critic(self):
        state_inputs = layers.Input((self.state_dim,), name="state_inputs")
        x1 = layers.Dense(16, activation="relu")(state_inputs)
        x1 = layers.Dense(32, activation="relu")(x1)
        
        action_inputs = layers.Input((self.action_dim,), name="action_inputs")
        x2 = layers.Dense(32, activation="relu")(action_inputs)
        
        x = layers.Concatenate()([x1, x2])
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(256, activation="relu")(x)
        outputs = layers.Dense(1)(x)
        model = keras.Model([state_inputs, action_inputs], outputs)
        return model
    
    def save(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        actor_path = os.path.join(dir, "actor_model.h5")
        critic_path = os.path.join(dir, "critic_model.h5")
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
    
    def load(self, dir):
        actor_path = os.path.join(dir, "actor_model.h5")
        critic_path = os.path.join(dir, "critic_model.h5")
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)
        
        
        