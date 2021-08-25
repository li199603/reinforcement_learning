import multiprocessing as mp
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
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--epsilon", type=float, default=0.9)
parser.add_argument("--buffer_size", type=int, default=5000)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--update_frequency", type=int, default=100)
parser.add_argument("--epsilon_increment", type=float, default=0.0002)
parser.add_argument("--learning_times", type=int, default=10000)
args = parser.parse_args()



def train():
    env = enviroment.Env_Breakout()
    featrue_dim = env.get_featrues_dim()
    action_dim = env.get_action_dim()
    agt = agent.DQN(featrue_dim, action_dim, args.lr, args.gamma, args.epsilon,
                    args.update_frequency, args.epsilon_increment)
    init_weights = agt.policy_net.get_weights()
    init_epsilon = 0
    reward_queue = mp.Queue()
    loss_queue = mp.Queue()
    step_data_queue = mp.Queue()
    batch_data_pipe = mp.Pipe()
    agent_pipe = mp.Pipe()
    
    sd = mp.Process(target=store_data, args=(featrue_dim, args.buffer_size, step_data_queue, batch_data_pipe[0]))
    pg = mp.Process(target=play_game, args=(reward_queue, step_data_queue, agent_pipe[0], init_weights, init_epsilon))
    wt = mp.Process(target=write_tensorboard, args=(reward_queue, loss_queue))
    
    for p in [sd, pg, wt]:
        p.start()
    
    
    for i in tqdm.tgrange(1, args.learning_times+1):
        batch_data_pipe[1].send(True)
        batch_data = batch_data_pipe[1].recv()
        hist = agt.learn(*batch_data)
        loss = sum(hist.history["loss"]) / len(hist.history["loss"])
        loss_queue.put(loss)
        if agent_pipe[1].poll():
            _ = agent_pipe[1].recv()
            x = {"weights": agt.policy_net.get_weights(), "epsilon": agt.epsilon}
            agent_pipe[1].send(x)

    

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

def store_data(featrue_dim, buffer_size, step_data_queue: mp.Queue, batch_data_pipe):
    data_buffer = replay_buffer.Replay_Buffer(featrue_dim, buffer_size)
    while True:
        s_pre, action, reward, s_cur, done = step_data_queue.get()
        data_buffer.store_data(s_pre, action, reward, s_cur, done)
        if batch_data_pipe.poll():
            try:
                batch_data = data_buffer.sample_batch_data()
                batch_data_pipe.send(batch_data)
                _ = batch_data_pipe.recv()
            except BufferError as e:
                print("********************* " + str(e) + " *********************")
            
        
def build_net(featrue_dim, action_dim):
    net = models.Sequential([
        layers.InputLayer(featrue_dim),
        layers.Conv2D(filters=8, kernel_size=(8, 8), strides=(4, 4), padding="same", activation='relu'),
        layers.Conv2D(filters=16, kernel_size=(4, 4), strides=(2, 2), padding="same", activation='relu'),
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

def play_game(reward_queue: mp.Queue, step_data_queue: mp.Queue, agent_pipe, init_weights, init_epsilon, update_frequency=50):
    env = enviroment.Env_Breakout()
    featrue_dim = env.get_featrues_dim()
    action_dim = env.get_action_dim()
    policy_net = build_net(featrue_dim, action_dim)
    policy_net.set_weights(init_weights)
    epsilon = init_epsilon
    episode_count = 0
    while True:
        s_cur = env.reset()
        s_pre = None
        total_reward = 0
        while True:
            action = choose_action(policy_net, s_cur, epsilon, action_dim)
            s_pre = s_cur
            s_cur, reward, done, _ = env.step(action)
            if done:
                reward -= 10
            step_data_queue.put([s_pre, action, reward, s_cur, done])
            total_reward += reward
            if done:
                break
        reward_queue.put(total_reward)
        episode_count += 1
        if episode_count == update_frequency:
            agent_pipe.send(True)
            x = agent_pipe.recv()
            policy_net.set_weights(x["weights"])
            epsilon = x["epsilon"]
            episode_count = 0
           
def write_tensorboard(reward_queue: mp.Queue, loss_queue: mp.Queue, max_count=50):
    path = time.strftime("DQN_for_Breakout-v0_multi-process/logs/%Y-%m-%d-%Hh%Mm%Ss", time.localtime()) 
    summary_writer = tf.summary.create_file_writer(path)
    reward_count = 0
    reward_sum = 0
    reward_step = 1
    loss_count = 0
    loss_sum = 0
    loss_step = 1
    
    while True:
        reward_sum += reward_queue.get()
        reward_count += 1
        loss_sum += loss_queue.get()
        loss_count += 1
        with summary_writer.as_default():
            if reward_count == max_count:
                tf.summary.scalar("avg_reward", reward_sum/max_count, reward_step)
                reward_count = 0
                reward_sum = 0
                reward_step += 1
            if loss_count == max_count:
                tf.summary.scalar("avg_loss", loss_sum/max_count, loss_step)
                loss_count = 0
                loss_sum = 0
                loss_step += 1
            
if __name__ == "__main__":
    train()