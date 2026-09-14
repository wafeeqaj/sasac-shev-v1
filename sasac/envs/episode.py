# Per-episode randomisation: how much of the cycle to run, and at what power scale.
from __future__ import annotations

import numpy as np


def sample_episode(cfg, rng, base):
    # Return ``(power_demand, duration_multiplier, power_scale)`` for one episode.
    pd = base if cfg.trace_slice is None else base[cfg.trace_slice[0]:cfg.trace_slice[1]]

    multiplier = 1.0
    if cfg.duration_multipliers is not None:
        multiplier = float(rng.choice(cfg.duration_multipliers))
        full, rem = divmod(int(round(len(pd) * multiplier)), len(pd))
        pd = np.concatenate([np.tile(pd, full), pd[:rem]])

    scale = 1.0
    if cfg.power_scale_choices is not None:
        scale = float(rng.choice(cfg.power_scale_choices))
        pd = pd * scale

    if cfg.max_steps is not None:
        pd = pd[:cfg.max_steps]
    return np.asarray(pd, dtype=float), multiplier, scale

