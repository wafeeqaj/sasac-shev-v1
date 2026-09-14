# Setup-time diagnostics: what this preset looks like before any training.
from __future__ import annotations

from typing import Dict

import numpy as np

_UNTRAINED_CACHE: Dict[tuple, float] = {}


def untrained_final_soc(env) -> float:
    # Final SOC of a uniform-random policy: the level an early curve should be read against.
    key = (env.cfg.cycle, env.cfg.max_steps, env.soc_start, env.power_multiply, env.power_change)
    if key not in _UNTRAINED_CACHE:
        probe = type(env)(env.cfg)
        probe.reset()
        for k in ("power_demand", "power_multiply", "power_change", "power_min", "power_max",
                  "total_distance", "delta_soc"):      # the same episode, not a fresh draw
            setattr(probe, k, getattr(env, k))
        probe.soc = probe.soc_start = env.soc_start
        rng = np.random.default_rng(0)
        n, span = probe.max_steps, probe.action_max - probe.action_min
        for t in range(n):
            probe.step(probe.action_min + rng.random(2) * span, t, n)
        _UNTRAINED_CACHE[key] = float(probe.soc)
    return _UNTRAINED_CACHE[key]


def describe(env) -> dict:
    from .shev_env import ENV_PRESETS
    c = env.cfg
    d = {"preset": env.preset or next((k for k, v in ENV_PRESETS.items() if v == c), None),
         "steps_per_episode": env.max_steps,
         "initial_soc": env.soc_start,
         "duration_multiplier": env.power_multiply,
         "power_scale": env.power_change,
         "trip_distance_km": round(env.total_distance, 1),
         "untrained_final_soc_pct": round(100 * untrained_final_soc(env), 1),
         "soc_band_pct": tuple(round(100 * b, 1) for b in env.band),
         "band_bonus": c.w_soc_good,
         "cd_tracking": c.w_soc_track}
    return d
