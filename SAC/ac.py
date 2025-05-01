import torch
from torch import nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, state_embedding):
        return self.net(state_embedding)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state_embedding, action_embedding):
        x = torch.cat([state_embedding, action_embedding], dim=-1)
        return self.net(x)