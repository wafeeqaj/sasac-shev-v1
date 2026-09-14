# Decision-Transformer actor and critic (paper Sec. III-A, Fig. 2b).
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    import transformers
except ImportError:  # pragma: no cover - only the DT models need it; FFN/GRU must keep working without it
    transformers = None

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


@dataclass
class DTConfig:
    hidden_size: int = 128
    num_heads: int = 4
    num_layers: int = 1
    context: int = 100                 # k, the sequence length seen by the transformer
    max_ep_len: Optional[int] = None   # size of the timestep embedding table; default 4 * context
    n_ctx: Optional[int] = None        # GPT-2 positional table; default 3 * context
    dropout: float = 0.1

    def resolved(self) -> "DTConfig":
        return DTConfig(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            context=self.context,
            max_ep_len=self.max_ep_len or 4 * self.context,
            n_ctx=self.n_ctx or 3 * self.context,
            dropout=self.dropout,
        )


def _gpt2(cfg: DTConfig) -> "transformers.GPT2Model":
    if transformers is None:
        raise ImportError("The DT actor/critic need the `transformers` package (pip install transformers)")
    config = transformers.GPT2Config(
        vocab_size=1,  # unused: we feed embeddings directly
        n_embd=cfg.hidden_size,
        n_head=cfg.num_heads,
        n_layer=cfg.num_layers,
        n_positions=cfg.n_ctx,
        n_ctx=cfg.n_ctx,
        resid_pdrop=cfg.dropout,
        embd_pdrop=cfg.dropout,
        attn_pdrop=cfg.dropout,
    )
    return transformers.GPT2Model(config)


class _DTBackbone(nn.Module):
    # Causal GPT-2 over (R, s, a) tokens with a learned timestep embedding.

    def __init__(self, state_dim: int, act_dim: int, cfg: DTConfig):
        super().__init__()
        cfg = cfg.resolved()
        self.cfg = cfg
        self.state_dim, self.act_dim, self.hidden_size = state_dim, act_dim, cfg.hidden_size
        self.max_length = None  # kept for parity with the reference implementation
        self.transformer = _gpt2(cfg)
        self.embed_timestep = nn.Embedding(cfg.max_ep_len, cfg.hidden_size)
        self.embed_return = nn.Linear(1, cfg.hidden_size)
        self.embed_state = nn.Linear(state_dim, cfg.hidden_size)
        self.embed_action = nn.Linear(act_dim, cfg.hidden_size)
        self.embed_ln = nn.LayerNorm(cfg.hidden_size)

    def trunk(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        device = states.device
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=device)
        if returns_to_go.dim() == 2:
            returns_to_go = returns_to_go.unsqueeze(-1)
        if timesteps.dim() == 3:
            timesteps = timesteps.squeeze(-1)
        timesteps = timesteps.long() % self.embed_timestep.num_embeddings

        state_emb = self.embed_state(states) + self.embed_timestep(timesteps)
        action_emb = self.embed_action(actions) + self.embed_timestep(timesteps)
        return_emb = self.embed_return(returns_to_go) + self.embed_timestep(timesteps)

        stacked = (
            torch.stack((return_emb, state_emb, action_emb), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        stacked = self.embed_ln(stacked)
        stacked_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )
        out = self.transformer(inputs_embeds=stacked, attention_mask=stacked_mask)["last_hidden_state"]
        return out.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)


class DTActor(_DTBackbone):
    def __init__(self, state_dim: int, act_dim: int, cfg: Optional[DTConfig] = None,
                 log_std_max: float = LOG_STD_MAX):
        super().__init__(state_dim, act_dim, cfg or DTConfig())
        self.predict_action_mean = nn.Linear(self.hidden_size, act_dim)
        self.predict_action_log_std = nn.Linear(self.hidden_size, act_dim)
        # wide when std is scaled by alpha, the FFN clamp otherwise
        self.log_std_max = float(log_std_max)

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(states, actions, returns_to_go, timesteps, attention_mask)
        h = x[:, 1]  # state tokens
        mean = self.predict_action_mean(h)
        log_std = self.predict_action_log_std(h).clamp(LOG_STD_MIN, self.log_std_max)
        return mean, log_std  # [B, T, A] each

    @torch.no_grad()
    def act_deterministic(self, states, actions, returns_to_go, timesteps) -> torch.Tensor:
        # tanh(mean) at the last time step (validation + MATLAB export).
        mean, _ = self.forward(states, actions, returns_to_go, timesteps)
        return torch.tanh(mean[:, -1, :])


class DTCritic(_DTBackbone):
    # DT backbone with a scalar head. Used by the DT-DT ablation only.

    def __init__(self, state_dim: int, act_dim: int, cfg: Optional[DTConfig] = None):
        cfg = cfg or DTConfig(num_heads=8, num_layers=6, n_ctx=512)
        super().__init__(state_dim, act_dim, cfg)
        self.q_value_head = nn.Linear(self.hidden_size, 1)

    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None) -> torch.Tensor:
        x = self.trunk(states, actions, returns_to_go, timesteps, attention_mask)
        return self.q_value_head(x[:, 2])  # action tokens: Q must depend on a_t. [B, T, 1]
