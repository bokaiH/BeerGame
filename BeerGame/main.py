# main.py
import gym
from gym import spaces
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from env import BeerGame
from dqn import DQN, DQNAgent
import matplotlib.pyplot as plt
from ppo import PPOPolicyNetwork, PPOValueNetwork, PPOAgent

def plot_learning_curve(rewards_per_episode, losses_per_episode):
    # Plot learning curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')

    plt.subplot(1, 2, 2)
    plt.plot(losses_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    plt.title('Average Loss per Episode')

    plt.tight_layout()
    plt.show()

def run_dqn():
    # Initialize environment and DQN agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = BeerGame()
    dqn_agent = DQNAgent(state_dim=30, action_dim=5, device=device)  # Adjust state_dim as per observation space

    episodes = 100
    rewards_per_episode = []
    losses_per_episode = []

    # Training the DQN agent
    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        losses = []

        while not done:
            action = [0] * env.n_agents
            action[0] = dqn_agent.act(state[0])  # DQN agent for the first agent
            rnd_action = env.action_space.sample()
            for i in range(1, env.n_agents):
                action[i] = rnd_action[i]

            next_state, reward, done_list, _ = env.step(action)
            done = all(done_list)
            total_reward += reward[0]

            dqn_agent.remember(state[0], action[0], reward[0], next_state[0], done)
            state = next_state
            step_losses = dqn_agent.replay()
            losses.extend(step_losses)

            env.render(log_file_path=f'./dqn_results/log_episode_{e}_render.txt', episode=e)
        
        dqn_agent.update_target_model()
        rewards_per_episode.append(total_reward)
        if losses:
            losses_per_episode.append(np.mean(losses))
        else:
            losses_per_episode.append(0)
        print(f"Episode {e+1}/{episodes} finished.")
    print("Training finished.")
    print("Rewards per Episode: ", rewards_per_episode)
    print("Losses per Episode: ", losses_per_episode)

    # Evaluating the final policy
    state = env.reset()
    done = False
    total_reward = 0
    order_behavior = []

    while not done:
        action = [0] * env.n_agents
        action[0] = dqn_agent.act(state[0])  # DQN agent for the first agent
        rnd_action = env.action_space.sample()
        for i in range(1, env.n_agents):
            action[i] = rnd_action[i]

        order_behavior.append(action[0])  # Record the DQN agent's action
        next_state, reward, done_list, _ = env.step(action)
        done = all(done_list)
        total_reward += reward[0]
        state = next_state

    print(f"Evaluation finished. Total Reward: {total_reward}")
    print("Order Behavior: ", order_behavior)

def run_ppo():
    # Initialize environment and PPO agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = BeerGame()
    ppo_agent = PPOAgent(state_dim=30, action_dim=5, device=device)  # Adjust state_dim as per observation space

    episodes = 100
    rewards_per_episode = []

    # Training the PPO agent
    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = [0] * env.n_agents
            log_probs = [0] * env.n_agents
            action[0], log_probs[0] = ppo_agent.act(state[0])  # PPO agent for the first agent
            rnd_action = env.action_space.sample()
            for i in range(1, env.n_agents):
                action[i] = rnd_action[i]

            next_state, reward, done_list, _ = env.step(action)
            done = all(done_list)
            total_reward += reward[0]

            ppo_agent.remember(state[0], action[0], reward[0], next_state[0], done, log_probs[0])
            state = next_state
            
            if done:
                ppo_agent.update()
        
        rewards_per_episode.append(total_reward)
        print(f"Episode {e+1}/{episodes} finished. Total Reward: {total_reward}")

    print("Training finished.")
    print("Rewards per Episode: ", rewards_per_episode)

    # Evaluating the final policy
    state = env.reset()
    done = False
    total_reward = 0
    order_behavior = []

    while not done:
        action = [0] * env.n_agents
        log_probs = [0] * env.n_agents
        action[0], log_probs[0] = ppo_agent.act(state[0])  # PPO agent for the first agent
        rnd_action = env.action_space.sample()
        for i in range(1, env.n_agents):
            action[i] = rnd_action[i]

        order_behavior.append(action[0])  # Record the PPO agent's action
        next_state, reward, done_list, _ = env.step(action)
        done = all(done_list)
        total_reward += reward[0]
        state = next_state

    print(f"Evaluation finished. Total Reward: {total_reward}")
    print("Order Behavior: ", order_behavior)
    
if __name__ == '__main__':
    run_dqn()
    run_ppo()
