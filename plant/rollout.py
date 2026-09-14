# Roll a trained actor out on the high-fidelity plant (paper Sec. IV-B).
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import yaml

from .simulator import HEVSimulator, fuel_economy_mpg, stack_outputs

ENGINE_OFF_SPEED = 85.0


def normalize(value, lo, hi):
    return 2.0 * (value - lo) / (hi - lo) - 1.0


@dataclass
class RolloutSettings:
    agent: str                                  # "ffn" | "gru" | "dt"
    context: int = 1
    initial_obs: Tuple[float, float, float] = (1.0, -1.0, -1.0)
    soc_range: Tuple[float, float] = (0.0, 1.0)
    action_max: Tuple[float, float] = (267.0, 1523.0)
    action_min: Dict[str, Tuple[float, float]] = field(default_factory=dict)   # per cycle
    pm_range: Dict[str, Tuple[float, float]] = field(default_factory=dict)     # per cycle
    initial_soc: float = 0.85
    rtg_init: float = 0.95
    seed: int = 2

    @classmethod
    def from_yaml(cls, path) -> "RolloutSettings":
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        # PyYAML reads "-69.56e3" as a string; coerce every numeric field explicitly.
        raw["action_min"] = {k: tuple(float(x) for x in v) for k, v in raw.get("action_min", {}).items()}
        raw["pm_range"] = {k: tuple(float(x) for x in v) for k, v in raw.get("pm_range", {}).items()}
        for k in ("initial_obs", "soc_range", "action_max"):
            if k in raw:
                raw[k] = tuple(float(x) for x in raw[k])
        for k in ("initial_soc", "rtg_init"):
            if k in raw:
                raw[k] = float(raw[k])
        return cls(**raw)

    def ranges_for(self, cycle: str):
        key = cycle if cycle in self.action_min else _family(cycle)
        if key not in self.action_min or key not in self.pm_range:
            raise KeyError(f"no validation ranges for cycle {cycle!r} (have {list(self.action_min)})")
        return np.asarray(self.action_min[key], float), np.asarray(self.action_max, float), self.pm_range[key]


def _family(cycle: str) -> str:
    c = cycle.upper()
    if c.startswith("FHDS"):
        return "FHDS"
    if c.startswith("US06"):
        return "US06"
    if c.startswith("HHD"):
        return "HHDDT"
    return cycle


class _ShiftBuffer:
    def __init__(self, length: int, dim: int, fill=0.0):
        self.buf = np.zeros((length, dim), dtype=np.float32)
        self.buf[:] = np.asarray(fill, dtype=np.float32).reshape(-1)

    def push(self, x) -> np.ndarray:
        self.buf[:-1] = self.buf[1:]
        self.buf[-1] = np.asarray(x, dtype=np.float32).reshape(-1)
        return self.buf


def _training_reward(cycle: str, initial_soc: float, n_steps: int):
    # The reward the DT was conditioned on in training, so return-to-go falls the same way.
    from sasac.envs import make_env

    env = make_env("fixed_hfet_10", cycle=cycle, max_steps=n_steps)
    env.reset()
    env.soc_start = initial_soc
    env.distance = 0.0
    fuel_max = env.vehicle.fuel_max

    def reward(soc, m_dot_fuel):
        env.distance += env.total_distance / n_steps
        env.target_soc = 0.85 - env.delta_soc * env.distance
        return env.reward_for(soc, m_dot_fuel / fuel_max)
    return reward


def rollout(actor, settings: RolloutSettings, cycle: str, sim: Optional[HEVSimulator] = None,
            device=None, verbose: bool = True) -> dict:
    # Run one cycle and return the logged trajectory.
    import torch

    device = device or torch.device("cpu")
    sim = sim or HEVSimulator(drive_file=cycle)
    a_min, a_max, (pm_min, pm_max) = settings.ranges_for(cycle)
    soc_lo, soc_hi = settings.soc_range
    N = sim.N
    torch.manual_seed(settings.seed)

    obs = np.asarray(settings.initial_obs, dtype=np.float32)
    soc = float(settings.initial_soc)
    out_list, soc_list = [], []
    kind = settings.agent
    if kind in ("gru", "dt"):
        # Pre-roll slots repeat the first observation, matching pad_mode "edge" in training.
        s_buf = _ShiftBuffer(settings.context, 3, obs)
    if kind == "dt":
        a_buf = _ShiftBuffer(settings.context, 2)
        r_buf = _ShiftBuffer(settings.context, 1, settings.rtg_init)
        steps = np.arange(settings.context, dtype=np.float32).reshape(-1, 1)
        rtg = float(settings.rtg_init)
        placeholder = np.zeros(2, dtype=np.float32)
        reward_of = _training_reward(cycle, soc, N)

    actor.eval()
    with torch.no_grad():
        for k in range(N):
            if kind == "ffn":
                a_norm = actor.act_deterministic(torch.as_tensor(obs, device=device).reshape(1, -1))[0].cpu().numpy()
            elif kind == "gru":
                seq = torch.as_tensor(s_buf.push(obs), device=device).unsqueeze(0)
                a_norm = actor.act_deterministic(seq)[0].cpu().numpy()
            else:
                s = torch.as_tensor(s_buf.push(obs), device=device).unsqueeze(0)
                # The newest action slot is a placeholder until this step's action exists.
                a = torch.as_tensor(a_buf.push(placeholder), device=device).unsqueeze(0)
                r = torch.as_tensor(r_buf.push([rtg]), device=device).unsqueeze(0)
                t = torch.as_tensor(steps, device=device).unsqueeze(0).long()
                s[torch.isnan(s)] = -1
                a[torch.isnan(a)] = -1
                a_norm = actor.act_deterministic(s, a, r, t)[0].cpu().numpy()
                a_norm = np.nan_to_num(a_norm, nan=0.0)
                a_buf.buf[-1] = a_norm

            action = (a_norm + 1) / 2 * (a_max - a_min) + a_min
            if action[0] < ENGINE_OFF_SPEED:
                action[:] = 0.0
            next_state, m_dot_fuel, P_m, _, _, _, _, _, out, _, _ = sim.evaluate_step(k, soc, float(action[0]), float(action[1]))
            soc = float(next_state[0])
            soc_list.append(soc)
            out_list.append(out)
            if kind == "dt":
                rtg -= reward_of(soc, 0.0 if np.isnan(m_dot_fuel) else m_dot_fuel) / N
            obs = np.array([normalize(soc, soc_lo, soc_hi), normalize(k / N, 0, 1), normalize(next_state[2], pm_min, pm_max)], dtype=np.float32)
            if verbose and k % 1000 == 0:
                print(f"  {k}/{N}  SOC={soc:.4f}", flush=True)

    res = stack_outputs(out_list)
    res["soc"] = np.asarray(soc_list)
    res["final_soc"] = soc_list[-1]
    res["mpg"] = fuel_economy_mpg(res["m_dot_fuel"], sim.v)
    return res


# Fields written to results/validation/.
ROLLOUT_FIELDS = ("m_dot_fuel", "next_state", "P_motor_effective", "T_motor", "wg")


def save_rollout_mat(res: dict, path, agent_tag: str, cycle: str) -> None:
    from scipy.io import savemat

    key = f"SAC_{agent_tag}_{cycle}_test"
    keep = {k: res[k] for k in ROLLOUT_FIELDS}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    savemat(str(path), {key: keep}, do_compression=True)
