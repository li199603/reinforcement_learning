import time
import gym
import argparse
import model
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("Sarsa")
parser.add_argument("--render", action="store_true")
parser.add_argument("--lr", type=float, default=0.85)
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--episodes", type=int, default=1000)
parser.add_argument("--epsilon", type=float, default=1)
args = parser.parse_args()

def run():
    action_map = {0: "up", 1: "right", 2: "down", 3: "left"}
    env = gym.make("CliffWalking-v0")
    Sarsa_model = model.Sarsa(env.observation_space.n, env.action_space.n,
                                   args.lr, args.gamma, args.epsilon)
    reward_list = []

    for i in range(args.episodes):
        start_time = time.time()
        s_cur = env.reset()
        a_cur = Sarsa_model.choose_action(s_cur)
        reward_sum = 0
        while True:
            if args.render:
                env.render()
            s_pre = s_cur
            print(action_map[a_cur])
            s_cur, reward, done, _ = env.step(a_cur)
            a_pre = a_cur
            a_cur = Sarsa_model.choose_action(s_cur)
            Sarsa_model.learn(s_pre, a_pre, reward, s_cur, a_cur, done)
            reward_sum += reward
            if done:
                break
        reward_list.append(reward_sum)
        end_time = time.time()
        print("Episode [%d / %d]\tsum reward: %.2f\ttraning_time: %.2fs" %
              (i, args.episodes, reward_sum, end_time-start_time))

    print("Final q_table values:")
    print(Sarsa_model.Q_table)
    env.close()

    plt.plot(range(args.episodes), reward_list)
    plt.show()


if __name__ == "__main__":
    run()




