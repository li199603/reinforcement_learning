import gym
import argparse
from agent import DDPG
import time

parser = argparse.ArgumentParser("Playing gym's game of Pendulum by DDPG")
parser.add_argument("--render", action="store_true")
parser.add_argument("--actor_lr", type=float, default=0.001)
parser.add_argument("--critic_lr", type=float, default=0.002)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--buffer_size", type=int, default=50000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--max_episode", type=int, default=50)
args = parser.parse_args()


def run():
    env = gym.make("Pendulum-v1")
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    action_bound = env.action_space.high
    ddpg = DDPG(action_dim,
                state_dim,
                action_bound,
                args.actor_lr,
                args.critic_lr,
                args.gamma,
                args.tau,
                args.buffer_size,
                args.batch_size)
    
    for episode in range(args.max_episode):
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
        print("episode: %d, reward: %.3f, time: %.3fs" % (episode, ep_reward, end_time - start_time))
            
            
if __name__ == "__main__":
    run()