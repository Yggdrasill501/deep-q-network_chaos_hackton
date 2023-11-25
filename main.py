import gym
import numpy as np
import matplotlib.pyplot as plt
from agents.dqn_agent import DQNAgent
from envs.system_env import SystemEnv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "production"], default="train")
parser.add_argument("--target_state", default=None)
args = parser.parse_args()

# Register the custom environment
gym.envs.register(
    id='SystemEnv-v0',
    entry_point=SystemEnv,
    max_episode_steps=500
)

EPISODES = 500

# Create the environment
env = gym.make('SystemEnv-v0', target_state=args.target_state)
state_size = np.prod(env.observation_space.shape)
action_size = env.action_space.nvec.prod()
agent = DQNAgent(state_size, action_size)
done = False
batch_size = 32

if args.mode == "train":
    # List to store total rewards for each episode
    rewards = []

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        # Store the total reward for this episode
        rewards.append(total_reward)

        # Save the model every 10 episodes
        if e % 10 == 0:
            agent.save("models/system-dqn.h5")

    # Plotting the rewards
    plt.plot(rewards)
    plt.title("Rewards per episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()

elif args.mode == "production":
    agent.load("models/system-dqn.h5")
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = np.reshape(next_state, [1, state_size])
        if done:
            print("Reached the target state:", args.target_state)
