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
parser.add_argument("--episodes", type=int, default=3000)
parser.add_argument("--hidden_dim", type=int, default=5)
args = parser.parse_args()

def get_env(env_id):
    env = gym.make(env_id)
    env = env.unwrapped
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    return env, state_dim, action_dim


def run():
    env, state_dim, action_dim = get_env("CartPole-v0")
    PG_model = model.Policy_Gradient_2(state_dim, action_dim, args.lr, args.gamma, args.hidden_dim)
    reward_list = []
    for i in tqdm.trange(args.episodes, ascii=True, unit='episodes'):
        start_time = time.time()
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
        reward_list.append(reward_sum)
        end_time = time.time()
        # print("Episode [%d / %d]\tsum reward: %.2f\ttime: %.2fs" %
        #       (i, args.episodes, reward_sum, end_time-start_time))

    env.close()

    plt.plot(range(args.episodes), reward_list)
    plt.show()




if __name__ == "__main__":
    run()




