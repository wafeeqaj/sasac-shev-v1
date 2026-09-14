# SAC for every actor x critic combination in the paper (Algorithm 1).
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from .buffers import SequenceBatch
from .config import AgentConfig
from .models import DTActor, DTConfig, DTCritic, FFNActor, FFNCritic, GRUActor, GRUCritic

EPS = 1e-6


def colored_noise(beta: float, n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    # (n, dim) unit-variance noise with a 1/f^beta spectrum (Timmer and Koenig 1995).
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    scale = f[:, None] ** (-beta / 2.0)
    w = scale * (rng.normal(size=(len(f), dim)) + 1j * rng.normal(size=(len(f), dim)))
    w[0] = w[0].real
    if n % 2 == 0:
        w[-1] = w[-1].real
    x = np.fft.irfft(w, n=n, axis=0)
    return x / x.std(axis=0, keepdims=True)


def one_step_on(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    # The same window one step later, for conditioning the critic target.
    return None if x is None else torch.cat((x[:, 1:], x[:, -1:]), dim=1)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@dataclass
class ActionResult:
    scaled: np.ndarray      # action in engine units [omega (rad/s), T (Nm)]
    normalized: np.ndarray  # tanh output in [-1, 1]  (what is stored in the replay buffer)
    state_norm: np.ndarray  # normalised current state (what is stored in the replay buffer)


class SACAgent:
    def __init__(self, cfg: AgentConfig, state_dim: int, action_dim: int, action_min, action_max):
        self.cfg = cfg
        self.state_dim, self.action_dim = state_dim, action_dim
        self.action_min = np.asarray(action_min, dtype=np.float32)
        self.action_max = np.asarray(action_max, dtype=np.float32)
        self.device = resolve_device(cfg.device)
        self.uses_sequences = not (cfg.actor == "ffn" and cfg.critic == "ffn")

        # ---- networks
        self.actor = self._build_actor().to(self.device)
        self.critic_1 = self._build_critic().to(self.device)
        self.critic_2 = self._build_critic().to(self.device)
        self.critic_1_target = self._build_critic().to(self.device)
        self.critic_2_target = self._build_critic().to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_1_target.eval()   # no dropout in a regression target
        self.critic_2_target.eval()

        # ---- optimisers
        Opt = optim.AdamW if cfg.optimizer.lower() == "adamw" else optim.Adam
        self.actor_optimizer = Opt(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic_1_optimizer = Opt(self.critic_1.parameters(), lr=cfg.lr_critic)
        self.critic_2_optimizer = Opt(self.critic_2.parameters(), lr=cfg.lr_critic)

        # temperature (auto-tuned, Table II); log_alpha seeded from log(alpha_init)
        self.log_alpha = torch.tensor([float(np.log(cfg.alpha_init))], requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = Opt([self.log_alpha], lr=cfg.lr_alpha, weight_decay=cfg.alpha_weight_decay)
        self.target_entropy = -float(action_dim) if cfg.target_entropy == "neg_action_dim" else float(cfg.target_entropy)
        # The floor binds log_alpha itself, or the tuner descends under a clamped alpha
        floors = [f for f in (cfg.log_alpha_min,
                              None if cfg.alpha_min is None else float(np.log(cfg.alpha_min)))
                  if f is not None]
        self.log_alpha_floor = max(floors) if floors else None

        # ---- state normalisation bounds (updated by the training loop every episode)
        self.state_min = np.zeros(state_dim, dtype=np.float32)
        self.state_max = np.ones(state_dim, dtype=np.float32)

        self.update_actor = True
        self.last_losses: Dict[str, torch.Tensor] = {}
        self._update_count = 0
        self._step = 0
        self._noise: Optional[np.ndarray] = None
        self._noise_rng = np.random.default_rng(np.random.randint(2 ** 31))


    # ------------------------------------------------------------------ builders
    def _dt_cfg(self, arch, context) -> DTConfig:
        return DTConfig(hidden_size=arch.hidden_size, num_heads=arch.num_heads, num_layers=arch.num_layers,
                        context=context, max_ep_len=arch.max_ep_len, n_ctx=arch.n_ctx, dropout=arch.dropout)

    def _build_actor(self) -> nn.Module:
        c = self.cfg
        if c.actor == "ffn":
            return FFNActor(self.state_dim, self.action_dim, c.hidden_actor, c.ffn_dropout)
        if c.actor == "gru":
            return GRUActor(self.state_dim, c.hidden_actor, self.action_dim, c.gru_layers, c.gru_dropout)
        if c.actor == "dt":
            return DTActor(self.state_dim, self.action_dim, self._dt_cfg(c.dt, c.context),
                           log_std_max=2.0 if c.dt_std_scaled_by_alpha else 0.2)
        raise ValueError(f"unknown actor type {c.actor!r}")

    def _build_critic(self) -> nn.Module:
        c = self.cfg
        if c.critic == "ffn":
            return FFNCritic(self.state_dim, self.action_dim, c.hidden_critic)
        if c.critic == "gru":
            return GRUCritic(self.state_dim, self.action_dim, c.hidden_critic, c.gru_layers, c.gru_dropout)
        if c.critic == "dt":
            return DTCritic(self.state_dim, self.action_dim, self._dt_cfg(c.dt_critic, c.context))
        raise ValueError(f"unknown critic type {c.critic!r}")

    # ------------------------------------------------------------------ helpers
    def describe(self) -> dict:
        # The resolved settings that decide how this agent explores, logged at the start of a run.
        c = self.cfg
        d = {
            "actor": c.actor, "critic": c.critic,
            "target_entropy": self.target_entropy,
            "alpha_init": c.alpha_init,
        }
        if c.actor != "ffn":
            d["context"] = c.context
        if c.actor == "dt":
            d["std"] = "exp(log_std) * alpha" if c.dt_std_scaled_by_alpha else "exp(log_std)"
        return d

    def normalize_state(self, state):
        return (np.asarray(state, dtype=np.float32) - self.state_min) / (self.state_max - self.state_min) * 2.0 - 1.0

    def scale_action(self, a_norm: np.ndarray) -> np.ndarray:
        return self.action_min + (a_norm + 1.0) * (self.action_max - self.action_min) / 2.0

    def _t(self, x, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)

    def _q(self, critic: nn.Module, states, actions, rtg=None, timesteps=None) -> torch.Tensor:
        if isinstance(critic, DTCritic):
            return critic(states, actions, rtg, timesteps)
        return critic(states, actions)

    # ------------------------------------------------------------------ policy sampling
    def reset_noise(self, n_steps: int) -> None:
        # Draw this episode's roll-out noise sequence, if it is coloured.
        if self.cfg.exploration_noise == "pink":
            self._noise = colored_noise(1.0, n_steps, self.action_dim, self._noise_rng)
            self._noise_i = 0

    def _next_noise(self) -> Optional[torch.Tensor]:
        if self.cfg.exploration_noise != "pink" or self._noise is None:
            return None
        e = self._noise[self._noise_i % len(self._noise)]
        self._noise_i += 1
        return self._t(e)

    def _sample(self, states, actions=None, rtg=None, timesteps=None, eps=None):
        # Reparameterised tanh-squashed Gaussian sample and its log-prob.
        c = self.cfg
        if c.actor == "ffn":
            mean, log_std = self.actor(states)
            normal = Normal(mean, log_std.exp())
            x_t = normal.rsample() if eps is None else mean + log_std.exp() * eps
            action = torch.tanh(x_t).clamp(-1 + EPS, 1 - EPS)
            log_prob = normal.log_prob(x_t).sum(-1, keepdim=True) - torch.log(1 - action.pow(2) + EPS).sum(-1, keepdim=True)
            return action, log_prob
        if c.actor == "gru":
            mean, log_std, _ = self.actor(states)
            std = log_std.exp()
        else:  # dt
            mean, log_std = self.actor(states, actions, rtg, timesteps)
            std = log_std.exp() * (self.alpha if c.dt_std_scaled_by_alpha else 1.0)
        normal = Normal(mean, std)
        x_t = normal.rsample() if eps is None else mean + std * eps
        action = torch.tanh(x_t).clamp(-1 + EPS, 1 - EPS)
        log_prob = (normal.log_prob(x_t) - torch.log(1 - action.pow(2) + EPS)).sum(2, keepdim=True)
        return action, log_prob

    @torch.no_grad()
    def select_action(self, state_seq, action_seq=None, rtg_seq=None, timesteps=None) -> ActionResult:
        # Sample an action (Algorithm 1, lines 3-19).
        c = self.cfg
        self.actor.eval()   # no dropout at roll-out
        state_norm = self.normalize_state(state_seq)
        return self._select_action(c, state_norm, action_seq, rtg_seq, timesteps)

    def _select_action(self, c, state_norm, action_seq, rtg_seq, timesteps) -> ActionResult:
        eps = self._next_noise()
        if c.actor == "ffn":
            s = self._t(state_norm.reshape(1, -1))
            a, _ = self._sample(s, eps=eps)
            a_norm = a[0].float().cpu().numpy()
            return ActionResult(self.scale_action(a_norm), a_norm, state_norm.reshape(-1))
        s = self._t(state_norm).reshape(1, -1, self.state_dim)
        if c.actor == "gru":
            a, _ = self._sample(s, eps=eps)
        else:
            a, _ = self._sample(
                s,
                self._t(action_seq).reshape(1, -1, self.action_dim),
                self._t(rtg_seq).reshape(1, -1, 1),
                self._t(timesteps, torch.long).reshape(1, -1),
                eps=eps,
            )
        a_norm = a[0, -1].float().cpu().numpy()
        return ActionResult(self.scale_action(a_norm), a_norm, state_norm[-1])

    # ------------------------------------------------------------------ learning
    def update(self, batch) -> Dict[str, torch.Tensor]:
        # One SAC update (Algorithm 1, lines 21-39). Returns detached 0-dim tensors.
        c = self.cfg
        self.actor.train()
        if isinstance(batch, SequenceBatch):
            states, actions, next_states = self._t(batch.states), self._t(batch.actions), self._t(batch.next_states)
            reward, done = self._t(batch.rewards), self._t(batch.dones)
            rtg = self._t(batch.returns_to_go) if batch.returns_to_go is not None else None
            timesteps = self._t(batch.timesteps, torch.long)
        else:
            s, a, ns, r, d = batch
            states, actions, next_states, reward, done = map(self._t, (s, a, ns, r, d))
            rtg = timesteps = None

        # ---- critic target (line 25-26)
        with torch.no_grad():
            if c.actor == "dt" or c.critic == "dt":
                # Positions are window-local, so only the contents shift.
                hist, rtg_n, steps_n = one_step_on(actions), one_step_on(rtg), timesteps
            else:
                hist, rtg_n, steps_n = actions, rtg, timesteps
            if c.actor == "dt":
                next_action, next_log_pi = self._sample(next_states, hist, rtg_n, steps_n)
            else:
                next_action, next_log_pi = self._sample(next_states)
            tq1 = self._q(self.critic_1_target, next_states, next_action, rtg_n, steps_n)
            tq2 = self._q(self.critic_2_target, next_states, next_action, rtg_n, steps_n)
            if tq1.dim() == 2 and next_log_pi.dim() == 3:  # FFN critic fed with sequences: last step
                next_log_pi, reward, done = next_log_pi[:, -1], reward[:, -1], done[:, -1]
            target_q = torch.min(tq1, tq2) - self.alpha * next_log_pi
            if c.q_target_clamp is not None:
                target_q = target_q.clamp(*c.q_target_clamp)
            if c.critic == "dt" and c.dt_critic_loss == "rtg_regression":
                target_q = rtg  # regress directly onto the observed return-to-go (paper, Sec. III-A)
            else:
                target_q = reward + (1 - done) * c.gamma * target_q

        # ---- critic losses (line 27 / 30)
        q1 = self._q(self.critic_1, states, actions, rtg, timesteps)
        q2 = self._q(self.critic_2, states, actions, rtg, timesteps)
        critic_loss_1 = nn.functional.mse_loss(q1, target_q)
        critic_loss_2 = nn.functional.mse_loss(q2, target_q)
        # Critics share no parameters, so one backward over the summed loss gives identical gradients.
        self.critic_1_optimizer.zero_grad(set_to_none=True)
        self.critic_2_optimizer.zero_grad(set_to_none=True)
        (critic_loss_1 + critic_loss_2).backward()
        if c.critic_grad_clip:
            nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=c.critic_grad_clip)
            nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=c.critic_grad_clip)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
        # Detached 0-dim tensors; float() here would sync the GPU every update.
        losses = {"critic_1": critic_loss_1.detach(), "critic_2": critic_loss_2.detach()}

        # re-score the same states with shuffled actions; a critic that ignores the action reads 0
        self._step += 1
        if c.probe_every and self._step % c.probe_every == 0:
            with torch.no_grad():
                perm = torch.randperm(actions.shape[0], device=actions.device)
                q_real = self._q(self.critic_1, states, actions, rtg, timesteps)
                q_wrong = self._q(self.critic_1, states, actions[perm], rtg, timesteps)
                losses["q_action"] = (q_real - q_wrong).abs().mean()

        # ---- actor and temperature (lines 33-38)
        if self.update_actor:
            if c.actor == "dt":
                pi, log_pi = self._sample(states, actions, rtg, timesteps)
            else:
                pi, log_pi = self._sample(states)
            q_pi = torch.min(self._q(self.critic_1, states, pi, rtg, timesteps), self._q(self.critic_2, states, pi, rtg, timesteps))
            if q_pi.dim() == 2 and log_pi.dim() == 3:
                log_pi = log_pi[:, -1]
            elif c.actor_q_reduction == "last" and q_pi.dim() == 3:
                q_pi, log_pi = q_pi[:, -1:], log_pi[:, -1:]
            actor_loss = (self.alpha * log_pi - q_pi).mean()
            alpha_loss = -(self.log_alpha * (log_pi.detach() + self.target_entropy)).mean()
            self._update_count += 1
            # NaN guard is sampled every nan_check_every updates; isnan() syncs the GPU.
            every = c.nan_check_every
            if every and self._update_count % every == 0:
                if torch.isnan(log_pi).any() or torch.isnan(actor_loss):
                    raise RuntimeError(
                        f"NaN in log_pi or actor_loss at update {self._update_count}")

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            if c.actor_grad_clip:
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=c.actor_grad_clip)
            self.actor_optimizer.step()

            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                if self.log_alpha_floor is not None:
                    self.log_alpha.clamp_(min=self.log_alpha_floor)
                self.alpha = self.log_alpha.exp().detach()
            losses.update({"actor": actor_loss.detach(), "alpha": alpha_loss.detach()})

        # ---- soft target update (line 39)
        for net, tgt in ((self.critic_1, self.critic_1_target), (self.critic_2, self.critic_2_target)):
            for p, tp in zip(net.parameters(), tgt.parameters()):
                tp.data.copy_(c.tau * p.data + (1 - c.tau) * tp.data)
        self.last_losses = losses
        return losses

    # ------------------------------------------------------------------ persistence
    def state_dicts(self) -> Dict[str, dict]:
        return {
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_1_target": self.critic_1_target.state_dict(),
            "critic_2_target": self.critic_2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "state_min": torch.as_tensor(self.state_min),
            "state_max": torch.as_tensor(self.state_max),
        }

    def save(self, prefix: str) -> None:
        # Write ``<prefix>.pt``.
        Path(prefix).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(self.cfg),
                    "state_dim": self.state_dim, "action_dim": self.action_dim,
                    "action_min": self.action_min, "action_max": self.action_max,
                    **self.state_dicts()}, f"{prefix}.pt")

    def load(self, prefix: str, strict: bool = True) -> None:
        # Load ``<prefix>.pt``.
        path = prefix if prefix.endswith(".pt") else f"{prefix}.pt"
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"], strict=strict)
        for k in ("critic_1", "critic_2", "critic_1_target", "critic_2_target"):
            if k in ck:
                getattr(self, k).load_state_dict(ck[k], strict=strict)
        if "log_alpha" in ck:
            with torch.no_grad():
                self.log_alpha.copy_(ck["log_alpha"].to(self.device))
                self.alpha = self.log_alpha.exp().detach()
        if "state_min" in ck:
            self.state_min = ck["state_min"].cpu().numpy()
            self.state_max = ck["state_max"].cpu().numpy()
