# Reward (Eq. 4) and its preflight check.
from __future__ import annotations

import numpy as np


def normalize(reward: float, lo: float, hi: float) -> float:
    return 2.0 * (reward - lo) / (hi - lo) - 1.0


def reward_for(env, soc: float, fuel_consumption: float) -> float:
    # Reward at a given SOC and normalised fuel rate.
    c = env.cfg
    fuel_w = c.w_fuel * env.soc_start ** 2
    p = 100.0 * soc
    lo, hi, top = 100.0 * env.band[0], 100.0 * env.band[1], 100.0 * c.soc_high_limit
    reward = (-fuel_w * fuel_consumption
              + c.w_soc_low * min(p - lo, 0.0)
              + c.w_soc_good * (lo <= p <= hi)
              - c.w_soc_high * max(p - top, 0.0)
              - c.w_soc_track * abs(p - 100.0 * env.target_soc))
    # analytic worst case, so a failing episode normalises to -1 rather than past it
    env.min_reward = -(fuel_w + c.w_soc_low * lo + c.w_soc_track * 70.0)
    env.max_reward = c.w_soc_good
    reward = normalize(reward, env.min_reward, env.max_reward)
    return reward if np.isfinite(reward) else 0.0


def check_reward_shape(env) -> None:
    # Raise if the reward has a region an agent cannot climb out of.
    fuel, band = 0.12, env.band
    saved = env.target_soc
    try:
        env.target_soc = 0.5 * (band[0] + band[1])          # end of the trip
        socs = [0.02, 0.05, 0.08, 0.11, 0.13, 0.145]
        flat = [(a, b) for a, b in zip(socs, socs[1:])
                if env.reward_for(b, fuel) - env.reward_for(a, fuel) < 1e-6]
        if flat:
            raise ValueError(f"reward is flat below the band, between {flat}")
        margin = 100.0 * (env.reward_for(0.5 * (band[0] + band[1]), fuel) - env.reward_for(0.20, fuel))
        if margin <= 2.0:
            raise ValueError(f"in-band margin is only {margin:.2f} reward points")
        env.target_soc = 0.50                               # mid-trip
        slope = 100.0 * (env.reward_for(0.50, fuel) - env.reward_for(0.70, fuel)) / 20.0
        if slope <= 0.2:
            raise ValueError(f"descent slope is only {slope:.3f} points per SOC point")
    finally:
        env.target_soc = saved
