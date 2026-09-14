# Feed-forward actor and critic: the FFN baseline.
from __future__ import annotations

import torch
import torch.nn as nn

LOG_STD_MIN, LOG_STD_MAX = -5.0, 0.2


class FFNActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.state_dim, self.action_dim = state_dim, action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, state: torch.Tensor):
        if state.dim() == 3:
            state = state[:, -1, :]
        x = self.net(state)
        return self.mean(x), self.log_std(x).clamp(LOG_STD_MIN, LOG_STD_MAX)

    @torch.no_grad()
    def act_deterministic(self, state: torch.Tensor) -> torch.Tensor:
        # tanh(mean): what the validation scripts and MATLAB export use.
        mean, _ = self.forward(state)
        return torch.tanh(mean)


class FFNCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.dim() == 3:
            state = state[:, -1, :]
        if action.dim() == 3:
            action = action[:, -1, :]
        return self.net(torch.cat([state, action], dim=-1))
