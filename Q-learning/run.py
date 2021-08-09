import time
import gym
import argparse
import model
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("Q-learning")
parser.add_argument("--render", action="store_true")
parser.add_argument("--lr", type=float, default=0.85)
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--epsilon", type=float, default=1)
args = parser.parse_args()

def run():
    action_map = {0: "up", 1: "right", 2: "down", 3: "left"}
    env = gym.make("CliffWalking-v0")
    Q_learning_model = model.Q_learning(env.observation_space.n, env.action_space.n,
                                        args.lr, args.gamma, args.epsilon)
    reward_list = []

    for i in range(args.episodes):
        start_time = time.time()
        s_cur = env.reset()
        reward_sum = 0
        while True:
            if args.render:
                env.render()
            action = Q_learning_model.choose_action(s_cur)
            print(action_map[action])
            s_pre = s_cur
            s_cur, reward, done, _ = env.step(action)
            Q_learning_model.learn(s_pre, action, reward, s_cur, done)
            reward_sum += reward
            if done:
                break
        reward_list.append(reward_sum)
        end_time = time.time()
        print("Episode [%d / %d]\tsum reward: %.2f\ttraning_time: %.2fs" %
              (i, args.episodes, reward_sum, end_time-start_time))

    print("Final q_table values:")
    print(Q_learning_model.Q_table)
    env.close()

    plt.plot(range(args.episodes), reward_list)
    plt.show()




if __name__ == "__main__":
    run()




