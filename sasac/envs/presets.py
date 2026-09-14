# Environment configuration and the preset combinations used in the paper.
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple


@dataclass
class EnvConfig:
    # Everything that varies between the ablation studies.

    trace_slice: Optional[Tuple[int, int]] = None
    initial_soc_choices: Tuple[float, ...] = (0.85, 0.75, 0.65, 0.55, 0.45)
    duration_multipliers: Optional[Tuple[float, ...]] = tuple(round(i * 0.1, 1) for i in range(1, 11))
    power_scale_choices: Optional[Tuple[float, ...]] = None
    engine_power_factor: float = 1.0
    max_steps: Optional[int] = None
    seed: Optional[int] = None

    # Fuel map stores infeasible cells as 0; the plant clips torque first so none is reachable.
    cycle: str = "FHDS_10"           # drive cycle in data/cycles; the plant derives the rest
    engine_off_speed: float = 85.0   # rad/s
    action_min: Tuple[float, float] = (80.0, 0.0)
    action_max: Tuple[float, float] = (267.56, 1523.0)

    # reward (Eq. 4). w_soc_good 50 and w_soc_track 3 differ from the printed 2.5 and 0, deliberately.
    w_fuel: float = 5.0        # -w_fuel * mdot * soc_init^2
    w_soc_low: float = 15.0    # per SOC point below the band
    w_soc_good: float = 50.0   # band bonus (Eq. 4 prints 2.5)
    w_soc_high: float = 10.0   # per SOC point above soc_high_limit
    w_soc_track: float = 3.0   # per SOC point from the charge-depleting reference (Eq. 4 prints 0)
    soc_band: Tuple[float, float] = (0.15, 0.18)
    soc_high_limit: float = 0.85

    def to_dict(self) -> dict:
        return asdict(self)


ENV_PRESETS = {
    # Ten HFET cycles, fixed 0.85 initial SOC (studies 1-3).
    "fixed_hfet_10": EnvConfig(
        initial_soc_choices=(0.85,),
        duration_multipliers=None,
        power_scale_choices=None,
        engine_power_factor=0.91,
    ),
    # Single HFET cycle: the first 779 s of the ten-cycle trace (study 1).
    "fixed_hfet_1": EnvConfig(
        trace_slice=(0, 779),
        initial_soc_choices=(0.85,),
        duration_multipliers=None,
        power_scale_choices=None,
        engine_power_factor=0.91,
    ),
    # Random initial SOC and random episode duration (study 5).
    "vary_soc_duration": EnvConfig(),
    # The same, plus random EM power scaling (study 6).
    "vary_all": EnvConfig(
        power_scale_choices=tuple(round(0.5 + 0.1 * i, 1) for i in range(11)),
    ),
    # Random initial SOC only (study 4).
    "vary_soc": EnvConfig(duration_multipliers=None),
}


def make_env(preset_or_cfg="vary_soc_duration", **overrides) -> "SeriesHEVEnv":
    # Build an environment from a preset name or an EnvConfig; overrides apply on top of the preset.
    from .shev_env import SeriesHEVEnv

    if isinstance(preset_or_cfg, str):
        if preset_or_cfg not in ENV_PRESETS:
            raise KeyError(f"unknown env preset {preset_or_cfg!r}; choose from {list(ENV_PRESETS)}")
        cfg = ENV_PRESETS[preset_or_cfg]
    else:
        cfg = preset_or_cfg
    preset = preset_or_cfg if isinstance(preset_or_cfg, str) else None
    if overrides:
        d = cfg.to_dict()
        d.update(overrides)
        cfg = EnvConfig(**d)
    env = SeriesHEVEnv(cfg)
    env.preset = preset          # so describe() can name it even with overrides applied
    return env
