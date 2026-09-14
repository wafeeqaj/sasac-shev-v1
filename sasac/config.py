# Experiment configuration (dataclasses + YAML); nested keys take dotted names.
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, get_type_hints

import yaml


@dataclass
class DTArchConfig:
    hidden_size: int = 128
    num_heads: int = 4
    num_layers: int = 1
    n_ctx: Optional[int] = None       # default 3 * context
    max_ep_len: Optional[int] = None  # default 4 * context
    dropout: float = 0.1


@dataclass
class AgentConfig:
    # Everything that defines the SAC agent (paper Table II + notebook specifics).

    actor: str = "dt"      # "ffn" | "gru" | "dt"
    critic: str = "gru"    # "ffn" | "gru" | "dt"
    context: int = 100     # k: sequence length (ignored for ffn-ffn)

    # network sizes
    hidden_actor: int = 128
    hidden_critic: int = 128
    gru_layers: int = 2
    ffn_dropout: float = 0.0
    gru_dropout: float = 0.1
    dt: DTArchConfig = field(default_factory=DTArchConfig)
    dt_critic: DTArchConfig = field(default_factory=lambda: DTArchConfig(num_heads=8, num_layers=6, n_ctx=512))

    # optimisation (Table II defaults)
    lr_actor: float = 1e-4
    lr_critic: float = 1e-4
    lr_alpha: float = 1e-4
    optimizer: str = "adam"          # "adam" | "adamw"
    alpha_weight_decay: float = 0.0  # the FFN notebooks used 1e-3 on the alpha optimiser
    gamma: float = 0.99
    tau: float = 0.005
    alpha_init: float = 1.0
    target_entropy: Union[str, float] = "neg_action_dim"  # "neg_action_dim" (= -dim(A)) or a number
    critic_grad_clip: Optional[float] = 0.25
    actor_grad_clip: Optional[float] = None
    q_target_clamp: Optional[Tuple[float, float]] = None  # e.g. (-1, 1) in the GRU notebooks
    log_alpha_min: Optional[float] = None                 # e.g. -10
    alpha_min: Optional[float] = None                     # e.g. 0.05
    actor_q_reduction: str = "all"    # "all": mean over the window, "last": last step only
    dt_std_scaled_by_alpha: bool = True   # DT notebooks: std = exp(log_std) * alpha
    dt_critic_loss: str = "bellman"       # "bellman" | "rtg_regression" (see models/dt.py)
    dt_history_actions: str = "normalized"  # what the DT sees as past actions at rollout: "normalized" | "scaled"
    exploration_noise: str = "white"      # roll-out only: "white" (SAC) | "pink" (Eberhard et al. 2023), losses unchanged
    device: str = "auto"

    # How often to measure whether the critic's Q depends on the action. 0 = off.
    probe_every: int = 100

    # The NaN guard syncs the GPU. 1 = every update, 0 = off.
    nan_check_every: int = 100


@dataclass
class TrainConfig:
    name: str = "dt_gru_k100"
    description: str = ""
    env: str = "vary_soc_duration"                 # preset name, see sasac.envs.ENV_PRESETS
    env_overrides: Dict[str, Any] = field(default_factory=dict)
    agent: AgentConfig = field(default_factory=AgentConfig)

    seed: int = 1
    num_episodes: int = 500
    batch_size: int = 64
    buffer_capacity: int = 500          # episodes (sequence buffers) or transitions (ffn)
    sampling: str = "random"            # ffn buffer only: "random" | "sequential" (ablation study 1)
    warmup_episodes: int = 10           # sequence buffers: start updating after this many episodes
    warmup_transitions: int = 70_000    # ffn buffer: start updating after this many transitions
    update_every: int = 25              # env steps between critic updates
    actor_update_every: int = 1         # actor/alpha update when (step % update_every == 0) and (step % actor_update_every == 0)
    pad_mode: str = "constant"          # how short histories are padded at rollout: "constant" | "edge"
    rtg_init: Union[str, float] = 0.95  # DT return-to-go seed: number or "max_steps"
    rtg_update: str = "subtract_normalized"  # R -= r / max_steps (return-to-go) | "add_normalized" | "subtract"
    save_every: int = 2                 # episodes between checkpoint / .mat log writes
    out_dir: str = "results/runs"
    load_from: Optional[str] = None     # checkpoint prefix to continue training from
    log_every_steps: int = 1000
    max_steps_per_episode: Optional[int] = None  # truncate episodes (quick checks only)
    check_reward_shape: bool = True     # refuse to start on a reward with no usable gradient

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.name


# ----------------------------------------------------------------------------------


def _from_dict(cls, d: Dict[str, Any]):
    # Recursively build a dataclass from a dict (unknown keys raise).
    if d is None:
        return cls()
    kwargs = {}
    names = {f.name: f for f in fields(cls)}
    hints = get_type_hints(cls)  # resolves the string annotations from `from __future__ import annotations`
    for k, v in d.items():
        if k not in names:
            raise KeyError(f"unknown config key {k!r} for {cls.__name__}; valid: {sorted(names)}")
        sub = _dataclass_type(hints.get(k))
        if sub is not None and isinstance(v, dict):
            v = _from_dict(sub, v)
        elif isinstance(v, list) and k in ("q_target_clamp",):
            v = tuple(v)
        kwargs[k] = v
    return cls(**kwargs)


def _dataclass_type(tp):
    if tp is not None and is_dataclass(tp):
        return tp
    return None


def to_dict(cfg) -> Dict[str, Any]:
    return asdict(cfg)


def apply_overrides(d: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    d = copy.deepcopy(d)
    for key, value in (overrides or {}).items():
        node = d
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return d


def load_config(path: Union[str, Path], overrides: Optional[Dict[str, Any]] = None) -> TrainConfig:
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    raw = apply_overrides(raw, overrides)
    return _from_dict(TrainConfig, raw)


def save_config(cfg: TrainConfig, path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(to_dict(cfg), fh, sort_keys=False)

