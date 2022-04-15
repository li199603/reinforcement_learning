from random import seed
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
from tensorflow.keras import layers
from buffer import Buffer
from tensorboardX import SummaryWriter

EPS = 1e-8
SEED = 7
np.random.seed(SEED)
tf.random.set_seed(SEED)

class Agent:
    def __init__(self,
                 state_dim,
                 num_actions,
                 actor_lr,
                 critic_lr,
                 gamma,
                 lam,
                 buffer_size,
                 clip_ratio,
                 target_kl,
                 actor_learn_iterations,
                 critic_learn_iterations,
                 summary_writer):
        
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.actor_learn_iterations = actor_learn_iterations
        self.critic_learn_iterations = critic_learn_iterations
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.critic_opt = keras.optimizers.Adam(critic_lr)
        self.actor_opt = keras.optimizers.Adam(actor_lr)
        self.actor.summary()
        self.critic.summary()
        
        self.buffer = Buffer(state_dim, buffer_size, gamma, lam)
        self.summary_writer = summary_writer
    
    @tf.function
    def policy(self, state):
        state = tf.reshape(state, (1, self.state_dim)) # (1, state_dim)
        logits = self.actor(state) # (1, num_actions)
        action = tf.squeeze(tf.random.categorical(logits, 1)) # scalar
        logprob = tf.squeeze(self._get_logprobabilities(logits, action)) # scalar
        return action, logprob
    
    def store_transition(self, state, action, reward, logprob):
        state_tensor = tf.reshape(state, (1, self.state_dim)) # (1, state_dim)
        value = np.squeeze(self.critic(state_tensor).numpy()) # scalar
        self.buffer.store(state, action, reward, value, logprob)
    
    def finish_trajectory(self, last_value):
        self.buffer.finish_trajectory(last_value)
    
    def learn(self):
        (state_buffer,
         action_buffer,
         advantage_buffer,
         return_buffer,
         logprobability_buffer,
        ) = self.buffer.get()
        
        # Update the policy and implement early stopping using KL divergence
        for _ in range(self.actor_learn_iterations):
            kl = self._actor_learn(state_buffer,
                                   action_buffer,
                                   logprobability_buffer,
                                   advantage_buffer)
            if kl > 1.5 * self.target_kl:
                # Early Stopping
                break

        # Update the value function
        for _ in range(self.critic_learn_iterations):
            self._critic_learn(state_buffer, return_buffer)
    
    @tf.function
    def _actor_learn(self, state_buffer, action_buffer, logprobability_buffer, advantage_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            ratio = tf.exp(
                self._get_logprobabilities(self.actor(state_buffer), action_buffer)
                - logprobability_buffer
            ) # (None, )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            ) # (None, )
            cost = tf.minimum(ratio * advantage_buffer, min_advantage) # (None, )
            loss = -tf.reduce_mean(cost) # scalar
        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

        kl = tf.reduce_mean(
            logprobability_buffer
            - self._get_logprobabilities(self.actor(state_buffer), action_buffer)
        )
        return kl
    
    @tf.function
    def _critic_learn(self, state_buffer, return_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            cost = (return_buffer - self.critic(state_buffer)) ** 2 # (None, )
            loss = tf.reduce_mean(cost) # scalar
        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))
        
    def _build_actor(self):
        inputs = layers.Input(shape=(self.state_dim,), dtype=tf.float32)
        x = layers.Dense(64, "tanh", kernel_initializer=keras.initializers.GlorotUniform(SEED))(inputs)
        x = layers.Dense(64, "tanh")(x)
        outputs = layers.Dense(self.num_actions, name="logits")(x)
        model = keras.Model(inputs, outputs)
        return model
    
    def _build_critic(self):
        inputs = layers.Input(shape=(self.state_dim,), dtype=tf.float32)
        x = layers.Dense(64, "tanh")(inputs)
        x = layers.Dense(64, "tanh")(x)
        outputs = layers.Dense(1, name="state_value")(x)
        outputs = tf.squeeze(outputs, axis=1)
        model = keras.Model(inputs, outputs)
        return model
    
    @tf.function
    def _get_logprobabilities(self, logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits) # (None, num_actions)
        logprobability = tf.reduce_sum(
            tf.one_hot(a, self.num_actions) * logprobabilities_all, axis=1
        )
        return logprobability # (None,)


class AgentContinuousAction:
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_lr,
                 critic_lr,
                 gamma,
                 lam,
                 buffer_size,
                 clip_ratio,
                 target_kl,
                 actor_learn_iterations,
                 critic_learn_iterations,
                 action_bound,
                 summary_writer):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_learn_iterations = actor_learn_iterations
        self.critic_learn_iterations = critic_learn_iterations
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.action_bound = action_bound
        
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.critic_opt = keras.optimizers.Adam(critic_lr)
        self.actor_opt = keras.optimizers.Adam(actor_lr)
        self.actor.summary()
        self.critic.summary()
        
        self.buffer = Buffer(state_dim, buffer_size, gamma, lam, action_dim)
        self.summary_writer = summary_writer
        
        # self.init_params()
        
        
    # def init_params(self):
    #     all_params = np.random.uniform(-0.1, 0.1, 400)
    #     # print("***** all_params *****")
    #     # print("sum: %.5f, head: %.5f, end: %.5f" % (np.sum(all_params), all_params[0], all_params[1]))
    #     # print("***** all_params - end *****")
    #     all_params = tf.convert_to_tensor(all_params, tf.float32)
    #     def get_params(shape):
    #         num = 1
    #         for d in shape:
    #             num *= d
    #         return tf.reshape(all_params[:num], shape)
    #     self.a_params = self.actor.trainable_variables
    #     [a_p.assign(get_params(a_p.shape)) for a_p in self.a_params]
    #     self.c_params = self.critic.trainable_variables
    #     [c_p.assign(get_params(c_p.shape)) for c_p in self.c_params]
        
    
    @tf.function
    def policy(self, state):
        state = tf.reshape(state, (1, self.state_dim)) # (1, state_dim)
        mean, std = self.actor(state) # (1, action_dim)
        norm_dist = tfp.distributions.Normal(loc=mean, scale=std)
        action = norm_dist.sample(seed=SEED) # (1, action_dim)
        action = tf.clip_by_value(action, -self.action_bound, self.action_bound)
        prob = norm_dist.prob(action) # (1,)
        action, prob = tf.squeeze(action, axis=0), tf.squeeze(prob, axis=0) # (action_dim, )  scalar
        return action, prob
    
    def store_transition(self, state, action, reward, prob):
        state_tensor = tf.reshape(state, (1, self.state_dim)) # (1, state_dim)
        value = np.squeeze(self.critic(state_tensor).numpy()) # scalar
        self.buffer.store(state, action, reward, value, prob)
    
    def finish_trajectory(self, last_value):
        self.buffer.finish_trajectory(last_value)
    
    def learn(self):
        (state_buffer,
         action_buffer,
         advantage_buffer,
         return_buffer,
         probability_buffer,
        ) = self.buffer.get()
        # Update the policy and implement early stopping using KL divergence
        for _ in range(self.actor_learn_iterations):
            kl = self._actor_learn(state_buffer,
                                   action_buffer,
                                   probability_buffer,
                                   advantage_buffer)
            if kl > 1.5 * self.target_kl:
                # Early Stopping
                break

        # Update the value function
        for _ in range(self.critic_learn_iterations):
            self._critic_learn(state_buffer, return_buffer)
    
    @tf.function
    def _actor_learn(self, state_buffer, action_buffer, probability_buffer, advantage_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            mean, std = self.actor(state_buffer)
            prob = self._get_probabilities(mean, std, action_buffer)
            ratio = prob / (probability_buffer + EPS)
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            ) # (None, )
            cost = tf.minimum(ratio * advantage_buffer, min_advantage) # (None, )
            loss = -tf.reduce_mean(cost) # scalar
        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

        mean, std = self.actor(state_buffer)
        prob = self._get_probabilities(mean, std, action_buffer)
        kl = tf.reduce_mean(
            tf.math.log(probability_buffer) - tf.math.log(prob)
        )
        return kl
    
    @tf.function
    def _critic_learn(self, state_buffer, return_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            cost = (return_buffer - self.critic(state_buffer)) ** 2 # (None, )
            loss = tf.reduce_mean(cost) # scalar
        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))
        
    def _build_actor(self):
        inputs = layers.Input(shape=(self.state_dim,), dtype=tf.float32)
        x = layers.Dense(100, "relu")(inputs)
        mean = layers.Dense(self.action_dim, "tanh")(x)
        mean = layers.multiply([mean, self.action_bound])
        std = layers.Dense(self.action_dim, "softplus")(x)
        model = keras.Model(inputs, [mean, std])
        return model
    
    def _build_critic(self):
        inputs = layers.Input(shape=(self.state_dim,), dtype=tf.float32)
        x = layers.Dense(100, "relu")(inputs)
        outputs = layers.Dense(1, name="state_value")(x)
        outputs = tf.squeeze(outputs, axis=1)
        model = keras.Model(inputs, outputs)
        return model
    
    @tf.function
    def _get_probabilities(self, mean, std, x):
        norm_dist = tfp.distributions.Normal(loc=mean, scale=std)
        prob = norm_dist.prob(x)
        prob = tf.squeeze(prob, axis=1)
        return prob # (None, )
        

