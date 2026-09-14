# Training loop (paper Algorithm 1). run_training(load_config("configs/paper/dt_gru_k100.yaml"))
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .agent import SACAgent
from .buffers import EpisodicSequenceBuffer, TransitionReplayBuffer
from .config import TrainConfig, save_config
from .envs import check_reward_shape, make_env
from .utils import TrainingHistory, set_seed

EpisodeCallback = Callable[[TrainingHistory, dict], None]


def training_summary(cfg: TrainConfig, env) -> dict:
    # Gradient steps and episode counts implied by a config.
    steps = env.max_steps
    critic = steps // cfg.update_every
    actor = sum(1 for s in range(1, steps + 1)
                if s % cfg.update_every == 0 and s % cfg.actor_update_every == 0)
    warmup = (cfg.warmup_episodes if cfg.agent.actor != "ffn"
              else max(1, cfg.warmup_transitions // steps))
    return {
        "critic_updates_per_episode": critic,
        "actor_updates_per_episode": actor,
        "warmup_episodes": warmup,
        "critic_updates_over_the_run": critic * max(0, cfg.num_episodes - warmup),
        "q_action_sens_samples_per_episode": critic // cfg.agent.probe_every if cfg.agent.probe_every else 0,
        "note": f"losses are nan until episode {warmup}",
    }


def build(cfg: TrainConfig):
    # Instantiate ``(env, agent, replay_buffer)`` from a config.
    if cfg.load_from:
        prefix = Path(cfg.load_from)
        if not prefix.with_suffix(".pt").exists():
            prev = prefix.name
            raise FileNotFoundError(
                f"{cfg.name} continues from '{prev}' but {prefix}.pt does not exist; "
                f"run {prev} first, or set load_from: null")
    if cfg.agent.actor == "dt" and cfg.agent.dt_history_actions != "normalized":
        raise ValueError("dt_history_actions must be 'normalized': the buffer stores tanh output")
    if cfg.agent.exploration_noise not in ("white", "pink"):
        raise ValueError(f"unknown exploration_noise {cfg.agent.exploration_noise!r}")
    if cfg.rtg_update not in ("add_normalized", "subtract_normalized", "subtract"):
        raise ValueError(f"unknown rtg_update {cfg.rtg_update!r}")
    if cfg.rtg_update == "subtract" and cfg.rtg_init != "max_steps":
        raise ValueError(f"rtg_update 'subtract' needs rtg_init 'max_steps', not {cfg.rtg_init!r}")
    set_seed(cfg.seed)
    env_overrides = dict(cfg.env_overrides)
    env_overrides.setdefault("seed", cfg.seed)
    if cfg.max_steps_per_episode is not None:
        env_overrides["max_steps"] = cfg.max_steps_per_episode
    env = make_env(cfg.env, **env_overrides)
    env.reset()
    if cfg.check_reward_shape:
        check_reward_shape(env)
    agent = SACAgent(cfg.agent, env.state_dim, env.action_dim, env.action_min, env.action_max)
    if agent.uses_sequences:
        buffer = EpisodicSequenceBuffer(cfg.buffer_capacity, cfg.agent.context, store_rtg=(cfg.agent.actor == "dt"), seed=cfg.seed)
    else:
        buffer = TransitionReplayBuffer(cfg.buffer_capacity, seed=cfg.seed, sampling=cfg.sampling)
    if cfg.load_from:
        agent.load(str(Path(cfg.load_from)))
    return env, agent, buffer


def _mean(values) -> float:
    # Mean of a list of 0-dim tensors, synced once.
    if not values:
        return float("nan")
    if hasattr(values[0], "detach"):
        import torch
        return float(torch.stack(values).mean())
    return float(np.mean(values))


def pad_front(seq: np.ndarray, length: int, mode: str, constant=0.0) -> np.ndarray:
    # Left-pad ``seq`` to ``length`` rows, keeping the newest observation in the last slot.
    if len(seq) >= length:
        return seq[-length:]
    pad = length - len(seq)
    if seq.ndim == 1:
        return np.pad(seq, (pad, 0), mode=mode, **({} if mode == "edge" else {"constant_values": constant}))
    return np.pad(seq, ((pad, 0), (0, 0)), mode=mode, **({} if mode == "edge" else {"constant_values": constant}))


def act_with_history(agent, actor_type, cfg, state, state_history, action_history,
                     rtg_history, k, rtg_init):
    # One action, with whatever history this actor type needs; shared with sasac.utils.probe.
    if actor_type == "ffn":
        return agent.select_action(state)
    seq = pad_front(np.array(state_history[-k:]), k, cfg.pad_mode)
    if actor_type == "gru":
        # The window starts from a zero hidden state, as the update and validate.py do.
        return agent.select_action(seq)
    # token j is (R, s, a) of one step; the newest action slot is a placeholder behind the causal mask
    past = action_history[-(k - 1):] if k > 1 else []
    a_seq = pad_front(np.array([*past, np.zeros_like(action_history[0])]), k, cfg.pad_mode)
    r_seq = pad_front(np.array(rtg_history[-k:]), k, cfg.pad_mode, constant=rtg_init)
    t_seq = np.arange(k)
    return agent.select_action(seq, a_seq, r_seq, t_seq)


def _checkpoint(agent, history, cfg, run_dir):
    agent.save(str(run_dir / cfg.name))
    history.to_mat(run_dir / f"{cfg.name}.mat", cfg.name)
    history.to_csv(run_dir / f"{cfg.name}.csv")


def run_training(
    cfg: TrainConfig,
    on_episode_end: Optional[EpisodeCallback] = None,
    verbose: bool = True,
    env=None,
    agent: Optional[SACAgent] = None,
    buffer=None,
) -> TrainingHistory:
    if env is None or agent is None or buffer is None:
        env, agent, buffer = build(cfg)
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    history = TrainingHistory()
    best_reward = -np.inf
    k = cfg.agent.context
    actor_type = cfg.agent.actor
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    for episode in range(cfg.num_episodes):
        t0 = time.time()
        state = env.reset()
        max_steps = env.max_steps
        agent.state_min = np.array([0.0, 0.0, env.power_min], dtype=np.float32)
        agent.state_max = np.array([1.0, 1.0, env.power_max], dtype=np.float32)
        rtg_init = float(max_steps) if cfg.rtg_init == "max_steps" else float(cfg.rtg_init)
        agent.reset_noise(max_steps)

        state_history = [np.asarray(state, dtype=np.float32)]
        action_history = [np.zeros(env.action_dim, dtype=np.float32)]
        rtg_history = [rtg_init]
        done = False
        total_steps = 0
        ep_reward = 0.0
        speeds, torques, fuels, alphas = [], [], [], []
        q1s, q2s, actor_ls, alpha_ls, q_act = [], [], [], [], []

        while total_steps < max_steps and not done:
            # -------- act (Algorithm 1, lines 3-17)
            res = act_with_history(agent, actor_type, cfg, state, state_history,
                                   action_history, rtg_history, k, rtg_init)
            next_state, reward, done = env.step(res.scaled, total_steps, max_steps)
            next_state_norm = agent.normalize_state(next_state)

            if actor_type == "dt":
                rtg = rtg_history[-1]      # the value the actor was conditioned on at this state
                step = reward / max_steps if cfg.rtg_update.endswith("normalized") else reward
                rtg_history.append(rtg + step if cfg.rtg_update == "add_normalized" else rtg - step)
                action_history.append(res.normalized if cfg.agent.dt_history_actions == "normalized" else res.scaled)
                buffer.add(res.state_norm, res.normalized, next_state_norm, reward, done, rtg)
            else:
                buffer.add(res.state_norm, res.normalized, next_state_norm, reward, done)
            state_history.append(np.asarray(next_state, dtype=np.float32))
            state = next_state
            ep_reward += reward
            total_steps += 1

            speeds.append(env.previous_speed)
            torques.append(env.previous_torq)
            fuels.append(env.fuel_consumption)
            alphas.append(float(agent.alpha))

            # -------- learn (lines 21-39)
            ready = (len(buffer) > cfg.warmup_episodes) if agent.uses_sequences else (len(buffer) > cfg.warmup_transitions)
            if ready and total_steps % cfg.update_every == 0:
                agent.update_actor = (total_steps % cfg.actor_update_every == 0)
                losses = agent.update(buffer.sample(cfg.batch_size))
                # reduced once per episode; float() here would sync the GPU every update
                q1s.append(losses["critic_1"])
                q2s.append(losses["critic_2"])
                if "actor" in losses:
                    actor_ls.append(losses["actor"])
                    alpha_ls.append(losses["alpha"])
                if "q_action" in losses:
                    q_act.append(losses["q_action"])
            if cfg.log_every_steps and total_steps % cfg.log_every_steps == 0:
                log(f"  step {total_steps}/{max_steps}")

        # -------- episode summary
        epi_reward = 100.0 * ep_reward / max_steps
        stats = dict(
            episode=episode, rewards=epi_reward, avg_alpha=float(np.mean(alphas)),
            avg_speeds=float(np.mean(speeds)), avg_torques=float(np.mean(torques)), socs=float(env.soc),
            initial_soc_list=float(env.soc_start), avg_fuel_list=float(np.mean(fuels)),
            pow_demand_list=float(env.power_change), duration_mult_list=float(env.power_multiply),
            torq_switch_counts=100.0 * env.torq_change / max_steps, speed_switch_counts=100.0 * env.speed_change / max_steps,
            time_list=(time.time() - t0) / 60.0,
            q1_loss_list=_mean(q1s), q2_loss_list=_mean(q2s),
            actor_loss_list=_mean(actor_ls), alpha_loss_list=_mean(alpha_ls),
            q_action_sens=_mean(q_act), steps_list=total_steps,
        )
        history.append(**stats)
        log(f"Episode {episode}, Reward: {ep_reward:.3f} (norm {epi_reward:.2f}), "
            f"final SOC {env.soc:.3f}, init SOC {env.soc_start}, steps {total_steps}, "
            f"{stats['time_list']:.2f} min"
            + ("" if q1s else "   [warmup: no updates, losses are nan]"))
        if on_episode_end is not None:
            on_episode_end(history, stats)

        if episode % cfg.save_every == 0:
            if epi_reward > best_reward:
                best_reward = epi_reward
                agent.save(str(run_dir / f"{cfg.name}_best"))
            _checkpoint(agent, history, cfg, run_dir)

    _checkpoint(agent, history, cfg, run_dir)
    log("Training complete.")
    return history
