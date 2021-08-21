import gym
import argparse
import agent
import enviroment
import time
import tqdm
import replay_buffer
import matplotlib.pyplot as plt
import tensorboard

parser = argparse.ArgumentParser("Ues DQN playing Breakout-v0. Train in a multi-process way.")
parser.add_argument("--render", action="store_true")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--gamma", type=float, default=0.9)
parser.add_argument("--episodes", type=int, default=30)
parser.add_argument("--epsilon", type=float, default=0.9)
parser.add_argument("--buffer_size", type=int, default=50000)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--update_frequency", type=int, default=100)
parser.add_argument("--learn_frequency", type=int, default=100)
parser.add_argument("--epsilon_increment", type=float, default=0.005)
parser.add_argument("--aggregate_step", type=int, default=10)
parser.add_argument("--learning_step", type=int, default=5000)
args = parser.parse_args()



def train():
    env = enviroment.Env_Breakout()
    featrue_dim = env.get_featrues_dim()
    action_dim = env.get_action_dim()
    agt = agent.DQN(featrue_dim, action_dim, args.lr, args.gamma, args.epsilon, args.hidden_dim,
                    args.buffer_size, args.batch_size, args.update_frequency, args.epsilon_increment)
    ep_rewards = []
    aggr_ep_rewards = {'ep':[],'avg':[],'min':[],'max':[]}
    total_step = 0
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
            agt.store_data(s_pre, action, reward, s_cur, done)
            total_step += 1
            if total_step % args.learn_frequency == 0:
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
        if i % 5000 == 0:
            cur_time = time.strftime("%Y-%m-%d-%Hh%Mm%Ss", time.localtime()) 
            agt.save(r"DQN_for_Breakout-v0/checkpoints/" + cur_time + ".h5")
        end_time = time.time()
        print("Episode [%3d / %d]    Total reward: %.2f    Current epsilon: %.4f    Play time: %.2fs" % (i, args.episodes, total_reward, agt.epsilon, end_time-start_time))
    env.close()




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

def collect_data(env, data_buffer):
    print("-------------------collect data------------------------")
    s_cur = env.reset()
    for _ in tqdm.trange(args.learning_step):
        action = env.action_space.sample()
        s_pre = s_cur
        s_cur, reward, done, _ = env.step(action)
        data_buffer.store_data(s_pre, action, reward, s_cur, done)
        if done:
            s_cur = env.reset()
    

def train2():
    def get_env(env_id):
        env = gym.make(env_id)
        env = env.unwrapped
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        return env, state_dim, action_dim
    env, state_dim, action_dim = get_env("CartPole-v0")
    data_buffer = replay_buffer.Replay_Buffer(state_dim)
    collect_data(env, data_buffer)
    DQN_model = agent.DQN(state_dim, action_dim, args.lr, args.gamma, args.epsilon,
                          args.update_frequency, args.epsilon_increment)

    ep_rewards = []
    aggr_ep_rewards = {'ep':[],'avg':[],'min':[],'max':[]}
    
    for i in range(1, args.episodes+1):
        start_time = time.time()
        s_cur = env.reset()
        total_reward = 0
        while True:
            if args.render:
                env.render()
            action = DQN_model.choose_action(s_cur)
            s_pre = s_cur
            s_cur, reward, done, _ = env.step(action)
            x, _, theta, _ = s_cur
            pre_x, _, pre_theta, _ = s_pre
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2
            if abs(theta) < abs(pre_theta):
                reward += 0.5
            if theta * pre_theta < 0:
                reward += 1
            data_buffer.store_data(s_pre, action, reward, s_cur, done)
            DQN_model.learn(*data_buffer.sample_batch_data())
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
        end_time = time.time()
        print("Episode [%2d / %d]\tTotal reward: %.2f\tPlay time: %.2fs" % (i, args.episodes, total_reward, end_time-start_time))

    env.close()

    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label = 'avg')
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label = 'min')
    plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label = 'max')
    plt.legend(loc='upper left')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.show()

    state = env.reset()
    DQN_model.epsilon = args.epsilon
    for i in range(5):
        state = env.reset()
        done = False
        while not done:
            action = DQN_model.choose_action(state)
            next_state, _, done, _ = env.step(action)
            state = next_state
            env.render()

    env.close()
    




if __name__ == "__main__":
    train2()
    
    # env = enviroment.Env_Breakout(HEIGHT_RANGE, WIDTH_RANGE)
    # featrue_dim = env.get_featrues_dim()
    # action_dim = env.get_action_dim()
    # agt = agent.DQN(featrue_dim, action_dim, args.lr, args.gamma, args.epsilon, args.hidden_dim,
    #                 args.buffer_size, args.batch_size, args.update_frequency, args.epsilon_increment)
    # agt.load(r"DQN_for_Breakout-v0\checkpoints\2021-08-20-20h57m16s.h5")
    # agt.epsilon = 1
    # for i in range(5):
    #     state = env.reset()
    #     done = False
    #     while not done:
    #         action = agt.choose_action(state)
    #         next_state, _, done, _ = env.step(action)
    #         state = next_state
    #         env.render()
    #         time.sleep(0.2)


    # env.close()
    




