# The plant wrapped for RL training: one cycle, its power demand precomputed.
from __future__ import annotations

from typing import Dict

import numpy as np

from .engine_operating_point import find_best_engine_operating_point_vectorized
from .simulator import BATTERY_EFF, SOC_MAX, HEVSimulator


class Vehicle:
    def __init__(self, cycle: str):
        self.sim = HEVSimulator(drive_file=cycle)
        self._engine_point = find_best_engine_operating_point_vectorized
        self.battery_eff = BATTERY_EFF
        self.power_demand = self.sim.motor_power_demand() / 1000.0     # kW
        self.distance_km = float(self.sim.drive.distance_miles()[-1]) * 1.609344
        # Fuel map fills with 0 off its speed grid, which would read as free fuel.
        mf = np.asarray(self.sim.ENG.mf, float)
        self.fuel_max = float(mf.max())
        we = np.asarray(self.sim.ENG.speed_range, float).ravel()
        self.speed_range = (float(we.min()), float(we.max()))

    def engine(self, speed: float, torque: float):
        # (fuel normalised to [0, 1], generator output W) for one engine command.
        if speed > 0.0:
            speed = min(max(speed, self.speed_range[0]), self.speed_range[1])
        mf, _, _, pgen, _, _ = self._engine_point(self.sim.ENG, self.sim.GEN, speed, torque)
        return mf / self.fuel_max, pgen

    def soc_step(self, soc: float, p_batt_w: float) -> float:
        s = self.sim
        v_oc = s.v_oc(soc)
        r = float(np.interp(soc, s._soc_grid, s._r_dis if p_batt_w >= 0 else s._r_chg))
        ib = self.battery_eff * (v_oc - np.sqrt(max(v_oc ** 2 - 4 * r * p_batt_w, 0.0))) / (2 * r)
        ib = min(max(ib, s.Ib_min), s.Ib_max)
        return min(max(soc - ib / (s.cap_Ah * 3600.0) * s.Ts, 0.0), SOC_MAX)


_CACHE: Dict[str, Vehicle] = {}


def get_vehicle(cycle: str) -> Vehicle:
    # One Vehicle per cycle; building one costs a few seconds.
    if cycle not in _CACHE:
        _CACHE[cycle] = Vehicle(cycle)
    return _CACHE[cycle]
