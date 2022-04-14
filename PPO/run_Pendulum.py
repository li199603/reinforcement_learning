import numpy as np
import tensorflow as tf
import gym
import argparse
from tensorboardX import SummaryWriter
from agent import AgentContinuousAction


SEED = 7
np.random.seed(SEED)
tf.random.set_seed(SEED)

parser = argparse.ArgumentParser("Playing gym's game of CartPole by PPO")
parser.add_argument("--steps_per_epoch", type=int, default=200)
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--clip_ratio", type=float, default=0.2)
parser.add_argument("--actor_lr", type=float, default=2e-3)
parser.add_argument("--critic_lr", type=float, default=1e-3)
parser.add_argument("--actor_learn_iterations", type=int, default=80)
parser.add_argument("--critic_learn_iterations", type=int, default=80)
parser.add_argument("--target_kl", type=float, default=0.01)
parser.add_argument("--render", action="store_true")
args = parser.parse_args()


def run():
    # Initialize the environment and get the dimensionality of the
    # observation space and the number of possible actions
    env = gym.make("Pendulum-v1")
    env.seed(SEED)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high
    
    # dir_name = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
    # summary_writer = SummaryWriter("PPO/summary/" + dir_name)

    # Initialize the Agent
    agent = AgentContinuousAction(state_dim,
                                  action_dim,
                                  args.actor_lr,
                                  args.critic_lr,
                                  args.gamma,
                                  args.lam,
                                  args.steps_per_epoch,
                                  args.clip_ratio,
                                  args.target_kl,
                                  args.actor_learn_iterations,
                                  args.critic_learn_iterations,
                                  action_bound,
                                  summary_writer=None)

    # Initialize the observation, episode return and episode length
    state, episode_return, episode_length = env.reset(), 0, 0


    # Iterate over the number of epochs
    for epoch in range(args.epochs):
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0

        # Iterate over the steps of each epoch
        for t in range(args.steps_per_epoch):
            if epoch % 100 == 1:
                env.render()

            # Get the logits, action, and take one step in the environment
            action, logprob = agent.policy(state)
            action, logprob = action.numpy(), logprob.numpy()
            # if not (-2.0 < float(action) < 2.0):
            #     print(action)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, logprob)
            state = next_state
            episode_return += reward
            episode_length += 1
            
            if done or (t == args.steps_per_epoch - 1):
                last_value = 0
                if not done:
                    state_tensor = tf.reshape(state, (1, state_dim))
                    last_value = np.squeeze(agent.critic(state_tensor).numpy())
                agent.finish_trajectory(last_value)
                sum_return += episode_return
                sum_length += episode_length
                num_episodes += 1
                state, episode_return, episode_length = env.reset(), 0, 0
                
        agent.learn()
        # Print mean return and length for each epoch
        print(
            f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
        )

if __name__ == "__main__":
    run()
