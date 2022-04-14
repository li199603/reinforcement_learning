import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from buffer import Buffer
from tensorboardX import SummaryWriter

EPS = 1e-8

# np.random.seed(2333)
# tf.random.set_seed(2333)

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
        x = layers.Dense(64, "tanh")(inputs)
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
    
    # @tf.function
    def policy(self, state):
        state = tf.reshape(state, (1, self.state_dim)) # (1, state_dim)
        mean, log_std = self.actor(state) # (1, action_dim)
        std = tf.math.exp(log_std) # (1, action_dim)
        action = mean + tf.random.normal(tf.shape(mean)) * std # (1, action_dim)
        
        action = tf.clip_by_value(action, -self.action_bound, self.action_bound)
        
        logprob = self._get_logprobabilities(mean, log_std, action) # (1,)
        action, logprob = tf.squeeze(action, axis=0), tf.squeeze(logprob, axis=0) # (action_dim, )  scalar
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
            # print(_)
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
    
    # @tf.function
    def _actor_learn(self, state_buffer, action_buffer, logprobability_buffer, advantage_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            mean, log_std = self.actor(state_buffer)
            
            log_prob = self._get_logprobabilities(mean, log_std, action_buffer)
            ratio = tf.exp(
                log_prob
                - logprobability_buffer
            ) # (None, )
            # ratio = tf.clip_by_value(ratio, 0.01, 10)
            
            # ratio = tf.exp(
            #     self._get_logprobabilities(mean, log_std, action_buffer)
            #     - logprobability_buffer
            # ) # (None, )
            
            
            
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            ) # (None, )
            cost = tf.minimum(ratio * advantage_buffer, min_advantage) # (None, )
            loss = -tf.reduce_mean(cost) # scalar
        grads = tape.gradient(loss, self.actor.trainable_variables)
        tmp = [tf.math.reduce_all(tf.math.is_finite(grads[i])) for i in range(len(grads))]
        if not tf.math.reduce_all(tmp):
            if not tf.math.reduce_all(tf.math.is_finite(cost)):
                print("cost")
                print(cost.numpy()[31:37])
            if not tf.math.reduce_all(tf.math.is_finite(min_advantage)):
                print("min_advantage")
            if not tf.math.reduce_all(tf.math.is_finite(ratio)):
                print("ratio")
                print(ratio.numpy()[31:37])
                print(log_prob.numpy()[31:37])
                print(logprobability_buffer[31:37])
                print((log_prob - logprobability_buffer)[31:37])
                print("****************")
                print(mean.numpy()[31:37])
                print(log_std.numpy()[31:37])
                print(log_prob.numpy()[31:37])
                print(logprobability_buffer[31:37])
                print(action_buffer[31:37])
                
            if not tf.math.reduce_all(tf.math.is_finite(log_prob)):
                print("log_prob")
                print(log_prob.numpy()[31:37])
            if not tf.math.reduce_all(tf.math.is_finite(advantage_buffer)):
                print("advantage_buffer")
            if not tf.math.reduce_all(tf.math.is_finite(logprobability_buffer)):
                print("logprobability_buffer")
            if not tf.math.reduce_all(tf.math.is_finite(mean)):
                print("mean")  
            if not tf.math.reduce_all(tf.math.is_finite(log_std)):
                print("log_std")
            
            
            tmp1 = [tf.math.reduce_all(tf.math.is_finite(self.actor.trainable_variables[i])) for i in range(len(self.actor.trainable_variables))]
            if not tf.math.reduce_all(tmp1):
            
            # tmp1 = [tf.math.reduce_any(tf.math.is_nan(self.actor.trainable_variables[i])) for i in range(len(self.actor.trainable_variables))]
            # if tf.math.reduce_any(tmp1):
                print(222)
                print(tmp1)
                exit()
            
            
            # if tf.math.reduce_any(tf.math.is_nan(cost)):
            #     print("**")
            #     print(cost)
            # print(log_prob[:3].numpy())
            # print(mean[:3].numpy())
            # print(log_std[:3].numpy())
            # print(state_buffer[:3])
            print(111)
            print(grads[0][0,:3])
            print(loss)
            exit()
            
            # exit()
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

        mean, log_std = self.actor(state_buffer)
        kl = tf.reduce_mean(
            logprobability_buffer
            - self._get_logprobabilities(mean, log_std, action_buffer)
        )
        return kl
    
    # @tf.function
    def _critic_learn(self, state_buffer, return_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            cost = (return_buffer - self.critic(state_buffer)) ** 2 # (None, )
            loss = tf.reduce_mean(cost) # scalar
        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))
        
    def _build_actor(self):
        inputs = layers.Input(shape=(self.state_dim,), dtype=tf.float32)
        x = layers.Dense(64, "tanh")(inputs)
        # x = layers.Dense(128, "tanh")(x)
        x = layers.Dense(64, "tanh")(x)
        mean = layers.Dense(self.action_dim)(x)
        mean = layers.multiply([mean, self.action_bound * 0.8])
        log_std = layers.Dense(self.action_dim)(inputs)
        model = keras.Model(inputs, [mean, log_std])
        return model
    
    def _build_critic(self):
        inputs = layers.Input(shape=(self.state_dim,), dtype=tf.float32)
        x = layers.Dense(64, "tanh")(inputs)
        # x = layers.Dense(128, "tanh")(x)
        x = layers.Dense(64, "tanh")(x)
        outputs = layers.Dense(1, name="state_value")(x)
        outputs = tf.squeeze(outputs, axis=1)
        model = keras.Model(inputs, outputs)
        return model
    
    @tf.function
    def _get_logprobabilities(self, mean, log_std, x):
        logprob = -0.5 * (((x-mean)/(tf.math.exp(log_std)+EPS))**2 + 2*log_std + tf.math.log(2*np.pi)) # (None, action_dim)
        logprob = tf.reduce_sum(logprob, axis=1) # (None, )
        return logprob
        

