import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class PPOPolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PPOPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

class PPOValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(PPOValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PPOAgent:
    def __init__(self, state_dim, action_dim, device, lr=1e-3, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.memory = []
        self.policy_net = PPOPolicyNetwork(state_dim, action_dim).to(self.device)
        self.value_net = PPOValueNetwork(state_dim).to(self.device)
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=lr)

    def remember(self, state, action, reward, next_state, done, log_prob):
        self.memory.append((state, action, reward, next_state, done, log_prob))
    
    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action_probs = self.policy_net(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def evaluate(self, state, action):
        action_probs = self.policy_net(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.value_net(state)
        return action_logprobs, torch.squeeze(state_values), dist_entropy

    def update(self):
        if len(self.memory) == 0:
            return

        states, actions, rewards, next_states, dones, log_probs = zip(*self.memory)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        log_probs = torch.FloatTensor(log_probs).to(self.device)

        # Calculate returns and advantages
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        returns = torch.FloatTensor(returns).to(self.device)

        advantages = returns - self.value_net(states).detach()

        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            action_logprobs, state_values, dist_entropy = self.evaluate(states, actions)
            ratios = torch.exp(action_logprobs - log_probs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values, returns) - 0.01 * dist_entropy

            self.optimizer_policy.zero_grad()
            self.optimizer_value.zero_grad()
            loss.mean().backward()
            self.optimizer_policy.step()
            self.optimizer_value.step()

        self.memory = []
