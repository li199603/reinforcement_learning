from xmlrpc.client import boolean
import gym
import argparse
from agent import DDPG
import numpy as np
import time
import tqdm
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser("Playing gym's game of Pendulum by DDPG")
parser.add_argument("--render", action="store_true")
parser.add_argument("--actor_lr", type=float, default=0.001)
parser.add_argument("--critic_lr", type=float, default=0.002)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--buffer_size", type=int, default=50000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_step", type=int, default=1)
parser.add_argument("--priority_replay", action="store_true")
parser.add_argument("--max_episode", type=int, default=200)
args = parser.parse_args()


def run():
    env = gym.make("Pendulum-v1")
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    action_bound = env.action_space.high
    
    dir_name = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
    summary_writer = SummaryWriter("DDPG/summary/" + dir_name)
    
    ddpg = DDPG(action_dim,
                state_dim,
                action_bound,
                args.actor_lr,
                args.critic_lr,
                args.gamma,
                args.tau,
                args.buffer_size,
                args.batch_size,
                args.n_step,
                args.priority_replay,
                summary_writer)
    
    ep_reward_list = []
    for episode in tqdm.trange(args.max_episode):
        state = env.reset()
        ep_reward = 0
        start_time = time.time()
        while True:
            if args.render:
                env.render()
            action = ddpg.policy(state)
            next_state, reward, done, _ = env.step(action)
            ddpg.store_transition(state, action, reward, done, next_state)
            ddpg.learn()
            ep_reward += reward
            state = next_state
            if done:
                break
        end_time = time.time()
        ep_reward_list.append(ep_reward)
        avg_ep_reward = np.mean(ep_reward_list[-20:])
        ep_time = end_time - start_time
        summary_writer.add_scalar("env/ep_reward", ep_reward, episode)
        summary_writer.add_scalar("env/avg_ep_reward", avg_ep_reward, episode)
        summary_writer.add_scalar("env/ep_time", ep_time, episode)
    ddpg.save("DDPG/model_save")

def eval():
    env = gym.make("Pendulum-v1")
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    action_bound = env.action_space.high
    summary_writer = None
    
    ddpg = DDPG(action_dim,
                state_dim,
                action_bound,
                args.actor_lr,
                args.critic_lr,
                args.gamma,
                args.tau,
                args.buffer_size,
                args.batch_size,
                args.n_step,
                args.priority_replay,
                summary_writer)
    
    print("************** before load **************")
    ep_reward_list = []
    for episode in range(0):
        state = env.reset()
        ep_reward = 0
        while True:
            env.render()
            action = ddpg.policy(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state
            if done:
                break
        ep_reward_list.append(ep_reward)
        print("episode: %d, ep_reward: %.2f" % (episode, ep_reward))
    avg_ep_reward = np.mean(ep_reward_list)
    print("avg_ep_reward: %.2f" % (avg_ep_reward))
    
    
    print("************** after load **************")
    ddpg.load("DDPG/model_save")
    ep_reward_list = []
    for episode in range(3):
        state = env.reset()
        ep_reward = 0
        while True:
            env.render()
            action = ddpg.policy(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state
            if done:
                break
        ep_reward_list.append(ep_reward)
        print("episode: %d, ep_reward: %.2f" % (episode, ep_reward))
    avg_ep_reward = np.mean(ep_reward_list)
    print("avg_ep_reward: %.2f" % (avg_ep_reward))
    
            
            
if __name__ == "__main__":
    run()
    eval()