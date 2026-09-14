# Series hybrid electric vehicle training environment (paper Sec. II-A).
from __future__ import annotations

from typing import Optional

from plant.simulator import P_AUX_W
from plant.training import get_vehicle

from .diagnostics import describe, untrained_final_soc
from .episode import sample_episode
from .presets import ENV_PRESETS, EnvConfig, make_env  # noqa: F401
from .reward import check_reward_shape, reward_for  # noqa: F401

import numpy as np

try:  # gymnasium is optional; the env only uses it to advertise spaces.
    import gymnasium as gym
    from gymnasium import spaces

    _GymBase = gym.Env
except Exception:  # pragma: no cover - fallback when gymnasium is not installed
    gym = None
    spaces = None
    _GymBase = object


class SeriesHEVEnv(_GymBase):
    # Series-HEV training environment: reset() / step().

    metadata = {"render_modes": []}
    state_dim, action_dim = 3, 2

    def __init__(self, cfg: Optional[EnvConfig] = None):
        self.cfg = cfg or EnvConfig()
        self.preset: Optional[str] = None          # set by make_env, for describe()
        self.vehicle = get_vehicle(self.cfg.cycle)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.action_min = np.array(self.cfg.action_min, dtype=float)
        self.action_max = np.array(self.cfg.action_max, dtype=float)
        if spaces is not None:
            self.observation_space = spaces.Box(-np.inf, np.inf, (self.state_dim,), np.float32)
            self.action_space = spaces.Box(self.action_min.astype(np.float32),
                                           self.action_max.astype(np.float32), dtype=np.float32)
        self.previous_torq = self.previous_speed = 0.0    # read by the training log
        self.reset()

    def seed(self, seed: Optional[int]) -> None:
        self.rng = np.random.default_rng(seed)

    reward_for = reward_for
    untrained_final_soc = untrained_final_soc
    describe = describe

    @property
    def max_steps(self) -> int:
        return len(self.power_demand)

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.seed(seed)
        self.power_demand, self.power_multiply, self.power_change = sample_episode(
            self.cfg, self.rng, self.vehicle.power_demand)
        self.power_min, self.power_max = float(self.power_demand.min()), float(self.power_demand.max())
        self.total_distance = self.vehicle.distance_km * self.max_steps / self.vehicle.sim.N

        self.soc = self.soc_start = float(self.rng.choice(self.cfg.initial_soc_choices))
        self.delta_soc = 0.70 / self.total_distance
        self.distance = self.fuel_consumption = 0.0
        self.target_soc = 0.85 * 0.9
        self.torq_change = self.speed_change = 0
        self.band = tuple(self.cfg.soc_band)

        self.state = np.array([self.soc, 0.0, self.power_demand[0]])
        return self.state

    def step(self, action, t: int, max_timestep: int):
        c = self.cfg
        speed, torque = float(action[0]), float(action[1])
        if speed < c.engine_off_speed:
            speed = torque = 0.0

        self.fuel_consumption, pgen_w = self.vehicle.engine(speed, torque)
        engine_power = c.engine_power_factor * pgen_w / 1000.0                  # kW at the bus
        if self.power_demand[t] < -1e-5:                                       # regeneration, as the plant
            engine_power = 0.0
        self.soc = self.vehicle.soc_step(self.soc, (self.power_demand[t] - engine_power) * 1000.0 + P_AUX_W)

        self.distance += self.total_distance / max_timestep
        self.target_soc = 0.85 - self.delta_soc * self.distance
        self.torq_change += abs(torque - self.previous_torq) > 100
        self.speed_change += abs(speed - self.previous_speed) > 30
        self.previous_torq, self.previous_speed = torque, speed

        reward = self.reward_for(self.soc, self.fuel_consumption)
        done = self.distance > self.total_distance - self.total_distance / max_timestep
        self.state = np.array([self.soc, self.distance / self.total_distance,
                               self.power_demand[t + 1] if t + 1 < max_timestep else 0.0])
        return self.state, float(reward), bool(done)
