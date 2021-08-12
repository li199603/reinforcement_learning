import time
import gym
import argparse
import model
import matplotlib.pyplot as plt
import tqdm


parser = argparse.ArgumentParser("Policy Gradient")
parser.add_argument("--render", action="store_true")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--gamma", type=float, default=0.95)
parser.add_argument("--episodes", type=int, default=2000)
parser.add_argument("--hidden_dim", type=int, default=5)
args = parser.parse_args()

AGGREGATE_STATS_EVERY = 50

def get_env(env_id):
    env = gym.make(env_id)
    env = env.unwrapped
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    return env, state_dim, action_dim


def run():
    env, state_dim, action_dim = get_env("CartPole-v0")
    PG_model = model.Policy_Gradient_3(state_dim, action_dim, args.lr, args.gamma, args.hidden_dim)
    ep_rewards = []
    aggr_ep_rewards = {'ep':[],'avg':[],'min':[],'max':[]}
    for i in tqdm.trange(1, args.episodes+1, ascii=True, unit='episodes'):
        s_cur = env.reset()
        reward_sum = 0
        while True:
            if args.render:
                env.render()
            action = PG_model.choose_action(s_cur)
            s_pre = s_cur
            s_cur, reward, done, _ = env.step(action)
            x, x_dot, theta, theta_dot = s_cur
            pre_x, _, pre_theta, _ = s_pre
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2
            if abs(theta) < abs(pre_theta):
                reward += 0.5
            else:
                reward -= 0.5
            if theta * pre_theta < 0:
                reward += 1
            PG_model.store_data(s_pre, action, reward)
            reward_sum += reward
            if done:
                break
        PG_model.learn()
        ep_rewards.append(reward_sum)
        if i % AGGREGATE_STATS_EVERY == 0 or i == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
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

    state = env.reset()
    for i in range(5):
        state = env.reset()
        done = False
        while not done:
            action = PG_model.choose_action(state)
            next_state, _, done, _ = env.step(action)
            state = next_state
            env.render()

    env.close()
    
    


if __name__ == "__main__":
    run()




