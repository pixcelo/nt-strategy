import torch
import random
from qnetwork import QNetwork

class Agent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0):
        self.qnetwork = QNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.qnetwork.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_dim = action_dim

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return torch.argmax(self.qnetwork(state)).item()

    def train(self, batch_data):
        states, actions, rewards, next_states = zip(*batch_data)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        
        q_values = self.qnetwork(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.qnetwork(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values
        
        loss = torch.nn.MSELoss()(q_values, target_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
