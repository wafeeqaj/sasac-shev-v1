# GRU actor and critic (paper Sec. III-A, Fig. 2a).
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

LOG_STD_MIN, LOG_STD_MAX = -5.0, 0.2


class GRUActor(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128, action_dim: int = 2, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.state_dim, self.action_dim, self.hidden, self.n_layers = state_dim, action_dim, hidden, n_layers
        self.gru = nn.GRU(state_dim, hidden, n_layers, dropout=dropout if n_layers > 1 else 0.0, batch_first=True)
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)
        self.layer_norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() == 2:  # [T, D] -> [1, T, D]
            x = x.unsqueeze(0)
        out, h = self.gru(x, h)
        out = self.layer_norm(out)
        return self.mean(out), self.log_std(out).clamp(LOG_STD_MIN, LOG_STD_MAX), h

    @torch.no_grad()
    def act_deterministic(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        # tanh(mean) of the last time step: used by validation and the MATLAB export.
        mean, _, _ = self.forward(x, h)
        return torch.tanh(mean[:, -1, :])


class GRUCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(state_dim + action_dim, hidden, n_layers, dropout=dropout if n_layers > 1 else 0.0, batch_first=True)
        self.q_value = nn.Linear(hidden, 1)
        self.layer_norm = nn.LayerNorm(hidden)

    def forward(self, state: torch.Tensor, action: torch.Tensor, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = torch.cat([state, action], dim=2)  # [B, T, D]
        out, _ = self.gru(x, h)
        out = self.layer_norm(out)
        return self.q_value(out)  # [B, T, 1]
