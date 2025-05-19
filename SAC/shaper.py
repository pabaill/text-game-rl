import torch
from torch import nn

class RewardShaper(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, next_state):
        x = torch.cat([state, next_state], dim=-1)
        return self.net(x)