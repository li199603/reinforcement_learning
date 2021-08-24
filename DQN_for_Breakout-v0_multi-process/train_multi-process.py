from multiprocessing import Queue, Process
import multiprocessing
import tensorflow as tf
import time
import enviroment
from tensorflow.keras import models, layers
import replay_buffer
import numpy as np
import argparse
import agent
import tqdm


parser = argparse.ArgumentParser("Ues DQN playing Breakout-v0. Train in a multi-process way.")
parser.add_argument("--render", action="store_true")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--episodes", type=int, default=10000)
parser.add_argument("--epsilon", type=float, default=0.9)
parser.add_argument("--buffer_size", type=int, default=5000)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--update_frequency", type=int, default=100)
parser.add_argument("--learn_frequency", type=int, default=10)
parser.add_argument("--epsilon_increment", type=float, default=0.0002)
parser.add_argument("--learning_step", type=int, default=2000)
args = parser.parse_args()



def train():
    env = enviroment.Env_Breakout()
    featrue_dim = env.get_featrues_dim()
    action_dim = env.get_action_dim()
    # data_buffer = replay_buffer.Replay_Buffer(featrue_dim)
    agt = agent.DQN(featrue_dim, action_dim, args.lr, args.gamma, args.epsilon,
                    args.update_frequency, args.epsilon_increment)
    init_weights = agt.policy_net.get_weights()
    init_epsilon = agt.epsilon
    reward_queue = Queue()
    loss_queue = Queue()
    data_queue = Queue()
    
    sd = Process(target=store_data, args=(featrue_dim, args.buffer_size, data_queue))
    pg = Process(target=play_game, args=(reward_queue, data_queue, init_weights, init_epsilon))
    wt = Process(target=write_tensorboard, args=(reward_queue, loss_queue))
    
    for p in [sd, pg, wt]:
        p.start()
    
    while True:
        loss = 1
        loss_queue.put(loss)
    

def collect_data(env, data_buffer):
    s_cur = env.reset()
    for _ in tqdm.trange(args.learning_step):
        action_dim = env.get_action_dim()
        action = np.random.randint(action_dim)
        s_pre = s_cur
        s_cur, reward, done, _ = env.step(action)
        data_buffer.store_data(s_pre, action, reward, s_cur, done)
        if done:
            s_cur = env.reset() 

def store_data(featrue_dim, buffer_size, data_queue: Queue):
    data_buffer = replay_buffer.Replay_Buffer(featrue_dim, buffer_size)
    while True:
        s_pre, action, reward, s_cur, done = data_queue.get()
        data_buffer.store_data(s_pre, action, reward, s_cur, done)
        
def build_net(featrue_dim, action_dim):
    net = models.Sequential([
        layers.InputLayer(featrue_dim),
        layers.Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), padding="same", activation='relu'),
        layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same", activation='relu'),
        layers.MaxPool2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dense(units=50, activation="relu"),
        layers.Dense(action_dim)
    ])
    return net

def choose_action(policy_net, state, epsilon, action_dim):
    state = np.expand_dims(state, 0)
    if np.random.uniform() < epsilon:
        q_values = policy_net(state).numpy().reshape([action_dim])
        max_q = np.max(q_values)
        action = np.random.choice(np.where(q_values == max_q)[0])
    else:
        action = np.random.choice(action_dim)
    return action

def play_game(reward_queue: Queue, data_queue: Queue, init_weights, init_epsilon):
    env = enviroment.Env_Breakout()
    featrue_dim = env.get_featrues_dim()
    action_dim = env.get_action_dim()
    # policy_net = build_net(featrue_dim, action_dim)
    # policy_net.set_weights(init_weights)
    epsilon = init_epsilon
    while True:
        s_cur = env.reset()
        s_pre = None
        total_reward = 0
        while True:
            # action = choose_action(policy_net, s_cur, epsilon, action_dim)
            action = np.random.randint(action_dim)
            s_pre = s_cur
            s_cur, reward, done, _ = env.step(action)
            if done:
                reward -= 10
            data_queue.put([s_pre, action, reward, s_cur, done])
            ############################
            # policy_net  epsilon 更新 #
            ############################
            total_reward += reward
            if done:
                break
        reward_queue.put(total_reward)
           
def write_tensorboard(reward_queue: Queue, loss_queue: Queue, max_count=10):
    path = time.strftime("DQN_for_Breakout-v0_multi-process/logs/%Y-%m-%d-%Hh%Mm%Ss", time.localtime()) 
    summary_writer = tf.summary.create_file_writer(path)
    reward_count = 0
    reward_sum = 0
    reward_step = 1
    loss_count = 0
    loss_sum = 0
    loss_step = 1
    
    while True:
        if not reward_queue.empty():
            reward_sum += reward_queue.get()
            reward_count += 1
            print(reward_count)
        if not  loss_queue.empty():
            loss_sum += loss_queue.get()
            loss_count += 1
            print(loss_count)
        with summary_writer.as_default():
            if reward_count == max_count:
                tf.summary.scalar("arg_reward", reward_sum/max_count, reward_step)
                reward_count = 0
                reward_sum = 0
                reward_step += 1
            if loss_count == max_count:
                tf.summary.scalar("arg_loss", loss_sum/max_count, loss_step)
                loss_count = 0
                loss_sum = 0
                loss_step += 1
            
if __name__ == "__main__":
    train()