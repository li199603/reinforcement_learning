from cgitb import grey
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
from replay_buffer import ReplayBuffer, ReplayBufferNStep, PriorityReplayBuffer, PriorityReplayBufferNStep
import os

class DDPG:
    def __init__(self,
                 action_dim,
                 state_dim,
                 action_bound,
                 actor_lr,
                 critic_lr,
                 gamma,
                 tau,
                 buffer_size,
                 batch_size,
                 n_step,
                 priority_replay,
                 summary_writer):
        
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_bound = action_bound
        self.gamma = gamma
        self.tau = tau
        self.n_step = n_step
        self.priority_replay = priority_replay
        
        self.actor, self.actor_target = self._build_actor(), self._build_actor()
        self.critic, self.critic_target = self._build_critic(), self._build_critic()
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())
        self.critic_opt = keras.optimizers.Adam(critic_lr)
        self.actor_opt = keras.optimizers.Adam(actor_lr)
        self.actor.summary()
        self.critic.summary()
        
        if priority_replay:
            self.buffer = PriorityReplayBufferNStep(state_dim, action_dim, buffer_size, batch_size, n_step, gamma)
        else:
            self.buffer = ReplayBufferNStep(state_dim, action_dim, buffer_size, batch_size, n_step, gamma)
        self.summary_writer = summary_writer
        self.learn_step_count = 0
    
    def policy(self, state):
        state = np.expand_dims(state, 0)
        action = np.squeeze(self.actor(state).numpy(), axis=0)
        return action
    
    def learn(self):
        try:
            if self.priority_replay:
                states, actions, rewards, dones, next_states, node_indices, importance_sampling_weights = self._sample_transition()
            else:
                states, actions, rewards, dones, next_states = self._sample_transition()
                importance_sampling_weights = tf.convert_to_tensor([1.0]*len(states))
        except Exception as e:
            return
        critic_loss, abs_errors = self._critic_learn(states, actions, rewards, dones, next_states, importance_sampling_weights)
        actor_loss = self._actor_learn(states, importance_sampling_weights)
        if self.priority_replay:
            self.buffer.errors_update(node_indices, abs_errors.numpy())
        self.summary_writer.add_scalar("agent/critic_loss", critic_loss.numpy(), self.learn_step_count)
        self.summary_writer.add_scalar("agent/actor_loss", actor_loss.numpy(), self.learn_step_count)
        self._update_target_model()
        self.learn_step_count += 1
        
    @tf.function
    def _critic_learn(self, states, actions, rewards, dones, next_states, importance_sampling_weights):
        with tf.GradientTape() as tape:
            next_actions = self.actor_target(next_states)
            next_q = self.critic_target([next_states, next_actions])
            td_target = rewards + (1 - dones) * np.power(self.gamma, self.n_step) * next_q
            cur_q = self.critic([states, actions])
            costs = tf.math.square(cur_q - td_target)
            abs_errors = tf.math.sqrt(costs)
            loss = tf.math.reduce_mean(costs * importance_sampling_weights)
        grad = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grad, self.critic.trainable_variables))
        return loss, abs_errors
    
    @tf.function
    def _actor_learn(self, states, importance_sampling_weights):
        with tf.GradientTape() as tape:
            actions = self.actor(states)
            q = self.critic([states, actions])
            loss = -tf.math.reduce_mean(q * importance_sampling_weights)
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
        done = int(done)
        self.buffer.store(state, action, reward, done, next_state)
    
    def _sample_transition(self):
        if self.priority_replay:
            data_batch, node_indices, importance_sampling_weights = self.buffer.sample()
            states, actions, rewards, dones, next_states = data_batch
            importance_sampling_weights = tf.convert_to_tensor(importance_sampling_weights, dtype=tf.float32)
        else:
            states, actions, rewards, dones, next_states = self.buffer.sample()
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states)
        if self.priority_replay:
            return states, actions, rewards, dones, next_states, node_indices, importance_sampling_weights
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
        
        
        