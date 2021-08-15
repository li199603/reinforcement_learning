import gym
import argparse
from gym.envs.registration import make
import matplotlib.pyplot as plt
import tqdm
import agent

parser = argparse.ArgumentParser("Ues DQN play Breakout-v0")
parser.add_argument("--render", action="store_true")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--epsilon", type=float, default=0.9)
parser.add_argument("--hidden_dim", type=int, default=50)
parser.add_argument("--buffer_size", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--update_frequency", type=int, default=100)
parser.add_argument("--epsilon_increment", type=float, default=0.001)
parser.add_argument("--aggregate_step", type=int, default=5)
args = parser.parse_args()


HEIGHT_RANGE = [32, 194]
WIDTH_RANGE = [8, 151]

def state_preprocess(state):
    h1, h2 = HEIGHT_RANGE
    w1, w2 = WIDTH_RANGE
    state = state[h1:h2+1, w1:w2+1]
    new_state = 0.2989 * state[:, :, 0] + 0.5870 * state[:, :, 1] + 0.1140 * state[:, :, 2]
    return new_state

def train():
    env = gym.make("Breakout-v0")
    featrue_shape = [HEIGHT_RANGE[1]-HEIGHT_RANGE[0]+1, WIDTH_RANGE[1]-WIDTH_RANGE[0]+1]
    action_dim = env.action_space.n
    print(action_dim)
    agt = agent.DQN(featrue_shape, action_dim, args.lr, args.gamma, args.epsilon, args.hidden_dim,
                    args.buffer_size, args.batch_size, args.update_frequency, args.epsilon_increment)
    ep_rewards = []
    aggr_ep_rewards = {'ep':[],'avg':[],'min':[],'max':[]}
    for i in tqdm.trange(1, args.episodes+1, ascii=True, unit='episodes'):
        s_cur = env.reset()
        s_cur = state_preprocess(s_cur)
        reward_sum = 0
        while True:
            if args.render:
                env.render()
            action = agt.choose_action(s_cur)
            s_pre = s_cur
            s_cur, reward, done, _ = env.step(action)
            s_cur = state_preprocess(s_cur)
            agt.store_data(s_pre, action, reward, s_cur)
            agt.learn()
            reward_sum += reward
            if done:
                break
        ep_rewards.append(reward_sum)
        if i % args.aggregate_step == 0 or i == 1:
            average_reward = sum(ep_rewards[-args.aggregate_step:])/len(ep_rewards[-args.aggregate_step:])
            min_reward = min(ep_rewards[-args.aggregate_step:])
            max_reward = max(ep_rewards[-args.aggregate_step:])
            aggr_ep_rewards['ep'].append(i)
            aggr_ep_rewards['avg'].append(average_reward)
            aggr_ep_rewards['min'].append(min_reward)
            aggr_ep_rewards['max'].append(max_reward)   

    env.close()

    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label = 'avg')
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label = 'min')
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label = 'max')
    plt.legend(loc='upper left')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.show()

    for i in range(5):
        state = env.reset()
        state = state_preprocess(state)
        done = False
        while not done:
            action = agt.choose_action(state)
            next_state, _, done, _ = env.step(action)
            state = next_state
            state = state_preprocess(state)
            env.render()

    env.close()




if __name__ == "__main__":
    train()




