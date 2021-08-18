import gym
import argparse
import matplotlib.pyplot as plt
from tensorflow.python.keras.backend_config import epsilon
import tqdm
import agent
import enviroment
import time

parser = argparse.ArgumentParser("Ues DQN play Breakout-v0")
parser.add_argument("--render", action="store_true")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--episodes", type=int, default=500)
parser.add_argument("--epsilon", type=float, default=0.9)
parser.add_argument("--hidden_dim", type=int, default=50)
parser.add_argument("--buffer_size", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--update_frequency", type=int, default=100)
parser.add_argument("--epsilon_increment", type=float, default=0.0002)
parser.add_argument("--aggregate_step", type=int, default=10)
args = parser.parse_args()


HEIGHT_RANGE = [32, 194]
WIDTH_RANGE = [8, 151]


def train():
    env = enviroment.Env_Breakout(HEIGHT_RANGE, WIDTH_RANGE)
    featrue_dim = env.get_featrues_dim()
    action_dim = env.get_action_dim()
    agt = agent.DQN(featrue_dim, action_dim, args.lr, args.gamma, args.epsilon, args.hidden_dim,
                    args.buffer_size, args.batch_size, args.update_frequency, args.epsilon_increment)
    ep_rewards = []
    aggr_ep_rewards = {'ep':[],'avg':[],'min':[],'max':[]}
    for i in range(1, args.episodes+1):
        start_time = time.time()
        s_cur = env.reset()
        total_reward = 0
        while True:
            if args.render:
                env.render()
            action = agt.choose_action(s_cur)
            s_pre = s_cur
            s_cur, reward, done, _ = env.step(action)
            agt.store_data(s_pre, action, reward, s_cur)
            agt.learn()
            total_reward += reward
            if done:
                break
        ep_rewards.append(total_reward)
        if i % args.aggregate_step == 0 or i == 1:
            average_reward = sum(ep_rewards[-args.aggregate_step:])/len(ep_rewards[-args.aggregate_step:])
            min_reward = min(ep_rewards[-args.aggregate_step:])
            max_reward = max(ep_rewards[-args.aggregate_step:])
            aggr_ep_rewards['ep'].append(i)
            aggr_ep_rewards['avg'].append(average_reward)
            aggr_ep_rewards['min'].append(min_reward)
            aggr_ep_rewards['max'].append(max_reward)
        if i % 50 == 0:
            cur_time = time.strftime("%Y-%m-%d-%Hh%Mm%Ss", time.localtime()) 
            agt.save("DQN_for_Breakout-v0\checkpoints\\" + cur_time + ".h5")
        end_time = time.time()
        print("Episode [%3d / %d]    Total reward: %.2f    Current epsilon: %.4f    Play time: %.2fs" % (i, args.episodes, total_reward, agt.epsilon, end_time-start_time))
    env.close()


    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label = 'avg')
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label = 'min')
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label = 'max')
    plt.legend(loc='upper left')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.show()


    agt.epsilon = args.epsilon
    for i in range(5):
        state = env.reset()
        done = False
        while not done:
            action = agt.choose_action(state)
            next_state, _, done, _ = env.step(action)
            state = next_state
            env.render()

    env.close()




if __name__ == "__main__":
    train()
    
    # env = enviroment.Env_Breakout(HEIGHT_RANGE, WIDTH_RANGE)
    # featrue_dim = env.get_featrues_dim()
    # action_dim = env.get_action_dim()
    # agt = agent.DQN(featrue_dim, action_dim, args.lr, args.gamma, args.epsilon, args.hidden_dim,
    #                 args.buffer_size, args.batch_size, args.update_frequency, args.epsilon_increment)
    # agt.load(r"DQN_for_Breakout-v0\checkpoints\2021-08-17-03h46m17s.h5")
    # for i in range(5):
    #     state = env.reset()
    #     done = False
    #     while not done:
    #         action = agt.choose_action(state)
    #         next_state, _, done, _ = env.step(action)
    #         state = next_state
    #         env.render()


    # env.close()
    




