# Probes for a trained agent: does the critic see the action, does the actor see the state.
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..train import act_with_history, pad_front


def _window(env, agent, cfg, n_steps: int):
    # Drive the agent for n_steps and return its last window, plus the commands.
    k = cfg.agent.context
    rtg_init = float(env.max_steps) if cfg.rtg_init == "max_steps" else float(cfg.rtg_init)
    state = env.reset()
    agent.reset_noise(env.max_steps)
    sh = [np.asarray(state, np.float32)]
    ah = [np.zeros(env.action_dim, np.float32)]
    rh = [rtg_init]
    speeds, torques = [], []
    for t in range(n_steps):
        res = act_with_history(agent, cfg.agent.actor, cfg, state, sh, ah, rh, k, rtg_init)
        state, r, _ = env.step(res.scaled, t, env.max_steps)
        if cfg.agent.actor == "dt":
            step = r / env.max_steps if cfg.rtg_update.endswith("normalized") else r
            rh.append(rh[-1] + step if cfg.rtg_update == "add_normalized" else rh[-1] - step)
            ah.append(res.normalized)
        sh.append(np.asarray(state, np.float32))
        speeds.append(env.previous_speed)
        torques.append(env.previous_torq)
    s = agent.normalize_state(pad_front(np.array(sh[-k:]), k, cfg.pad_mode))
    a = pad_front(np.array(ah[-k:], np.float32), k, cfg.pad_mode)
    return s, a, np.array(rh[-k:], np.float32), np.mean(speeds), np.mean(torques)


def critic_signal(env, agent, cfg, points: Sequence[int] = (1500, 3000, 5000, 7000)) -> dict:
    # Q spread when the action moves, against Q spread when the state moves.
    import torch

    k = cfg.agent.context

    def b(x, dt=torch.float32):
        return torch.as_tensor(np.asarray(x), dtype=dt, device=agent.device).unsqueeze(0)

    steps = b(np.arange(k), torch.long)
    s, a, rtg, _, _ = _window(env, agent, cfg, max(k + 1, points[0]))
    s_t, rtg_t = b(s), b(rtg.reshape(-1, 1))
    grid = np.linspace(-1.0, 1.0, 21)

    with torch.no_grad():
        last, whole = [], []
        for v in grid:
            aa = a.copy()
            aa[-1] = v
            last.append(float(agent._q(agent.critic_1, s_t, b(aa), rtg_t, steps).reshape(-1)[-1]))
            whole.append(float(agent._q(agent.critic_1, s_t, b(np.full_like(a, v)), rtg_t, steps).reshape(-1)[-1]))

        by_state, means, stds = [], [], []
        for n in points:
            s2, a2, r2, _, _ = _window(env, agent, cfg, n)
            by_state.append(float(agent._q(agent.critic_1, b(s2), b(a), b(r2.reshape(-1, 1)), steps).reshape(-1)[-1]))
            mu, sd = _actor_out(agent, cfg, b(s2), b(a2), b(r2.reshape(-1, 1)), steps)
            means.append(mu)
            stds.append(sd)

    m = np.asarray(means)

    def span(v):
        return float(max(v) - min(v))

    return {
        "q_span_newest_action": span(last),
        "q_span_whole_window": span(whole),
        "q_span_across_states": span(by_state),
        "actor_mean_span": np.round(m.max(0) - m.min(0), 4).tolist(),
        "actor_std": round(float(np.mean(stds)), 4),
        "alpha": float(agent.alpha),
    }


def _actor_out(agent, cfg, s, a, rtg, steps):
    # (tanh(mean), std) at the newest position, in the units _sample uses.
    import torch

    if cfg.agent.actor == "dt":
        mean, log_std = agent.actor(s, a, rtg, steps)
        std = log_std.exp() * (agent.alpha if cfg.agent.dt_std_scaled_by_alpha else 1.0)
    elif cfg.agent.actor == "gru":
        mean, log_std, _ = agent.actor(s)
        std = log_std.exp()
    else:
        mean, log_std = agent.actor(s)
        std = log_std.exp()
    d = agent.action_dim
    return (torch.tanh(mean.reshape(-1, d)[-1]).cpu().numpy(),
            float(std.reshape(-1, d)[-1].mean()))


def control_fraction(env, agent, cfg, std: float, q_grad: float = 0.0, seed: int = 0) -> dict:
    # Switch counts of a fixed-mean policy at this width, the floor to read a run against.
    lo, hi = env.action_min, env.action_max
    _, _, _, speed, torque = _window(env, agent, cfg, min(600, env.max_steps))
    rng = np.random.default_rng(seed)
    env.reset()
    mu = np.arctanh(np.clip(2 * (np.array([speed, torque]) - lo) / (hi - lo) - 1, -0.999, 0.999))
    n = env.max_steps
    for t in range(n):
        cmd = lo + (np.tanh(mu + rng.normal(0.0, std, 2)) + 1) / 2 * (hi - lo)
        env.step(cmd, t, n)
    # the entropy term pulls the mean to the box centre with force 2 alpha a
    pos = (np.array([speed, torque]) - (lo + hi) / 2) / ((hi - lo) / 2)
    fence = min(q_grad / (2 * float(agent.alpha)), 1.0)
    return {"fixed_mean_torq_switch": round(100 * env.torq_change / n, 2),
            "fixed_mean_speed_switch": round(100 * env.speed_change / n, 2),
            "at_std": std, "about_the_mean": [round(speed, 1), round(torque, 1)],
            "mean_position": np.round(pos, 3).tolist(),
            "entropy_fence": round(fence, 3)}


def report(env, agent, cfg) -> dict:
    # Both probes, as one dict. The switch-count floor uses the actor's own width.
    out = critic_signal(env, agent, cfg)
    out.update(control_fraction(env, agent, cfg, out["actor_std"], out["q_span_newest_action"] / 2))
    return out
