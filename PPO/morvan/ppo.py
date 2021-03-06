"""
A simple version of Proximal Policy Optimization (PPO) using single thread.
Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]
View more on my tutorial website: https://morvanzhou.github.io/tutorials
Dependencies:
tensorflow r1.2
gym 0.9.2
"""

import tensorflow.compat.v1 as tf
import tensorflow as tf2
import numpy as np
import matplotlib.pyplot as plt
import gym
import tensorflow_probability as tfp
tf.disable_eager_execution()

EP_MAX = 2000
# EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 128
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 3, 1
SEED = 7
EPS = 1e-8
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization

np.random.seed(SEED)
tf2.random.set_seed(SEED)
tf.set_random_seed(SEED)

class PPO(object):

    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(seed=SEED), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                self.ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + EPS)
                surr = self.ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tfp.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.min_advantage = tf.clip_by_value(self.ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv
                self.cost = tf.minimum(surr, self.min_advantage)
                self.aloss = -tf.reduce_mean(self.cost)

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        
        self.sess.run(tf.global_variables_initializer())
        self.params_init()
        self.sess.run(self.update_oldpi_op)
        
    def params_init(self):
        # ????????????
        all_params = np.random.uniform(-0.1, 0.1, 400)
        print("***** all_params *****")
        print("sum: %.5f, head: %.5f, end: %.5f" % (np.sum(all_params), all_params[0], all_params[1]))
        print("***** all_params - end *****")
        all_params = tf2.convert_to_tensor(all_params, tf2.float32)
        def get_params(shape):
            num = 1
            for d in shape:
                num *= d
            return tf2.reshape(all_params[:num], shape)
        
        self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="pi")
        with tf.variable_scope('a_init'):
            a_init = [a_p.assign(get_params(a_p.shape)) for a_p in self.a_params]
            
        self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic")
        with tf.variable_scope('c_init'):
            c_init = [c_p.assign(get_params(c_p.shape)) for c_p in self.c_params]
        
        self.sess.run([a_init, c_init])

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv, v = self.sess.run([self.advantage, self.v], {self.tfs: s, self.tfdc_r: r})
        adv = (adv - adv.mean())/(adv.std())     # sometimes helpful
        
        # print("***** s a adv r*****")
        print(s[:3])
        print(a[3:6])
        print(adv[88:96])
        print(r[66:69])
        # print(v[:3])
        # exit(0)

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            loss, rartio, cost, min_advantage = \
                self.sess.run([self.aloss, self.ratio, self.cost, self.min_advantage], 
                            {self.tfs: s, self.tfa: a, self.tfadv: adv})
            print(loss)
            print(rartio[88:96])
            print(cost[88:96])
            # exit(0)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

            
            
        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            self.mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            self.sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tfp.distributions.Normal(loc=self.mu, scale=self.sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a, tmp_mu, tmp_sigma = self.sess.run([self.sample_op, self.mu, self.sigma], {self.tfs: s})
        print("state: ", s[0])
        print("mean: %.5f, std: %.5f, action: %.5f" %(tmp_mu, tmp_sigma, a))
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

env = gym.make('Pendulum-v1').unwrapped
env.seed(SEED)
ppo = PPO()
all_ep_r = []

for ep in range(EP_MAX):
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(BATCH):    # in one episode
        print("****** %d ******" % t)
        if ep % 100 == 1 or ep >= EP_MAX - 10:
            env.render()
        a = ppo.choose_action(s)
        s_, r, done, _ = env.step(a)
        print("reward: %.5f" % r)
        if ep == 10 and t == 4:
                exit(0)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)    # normalize reward, find to be useful
        s = s_
        ep_r += r

        # update ppo
        if t == BATCH - 1:
            v_s_ = ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br)
    if ep == 0: all_ep_r.append(ep_r)
    else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
    print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
        ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
    )
