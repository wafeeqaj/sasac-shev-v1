# High-fidelity Class-8 series-HEV plant used for validation (paper Sec. IV).
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator, interp1d

from .engine_operating_point import find_best_engine_operating_point_vectorized

P_AUX_W = 5000.0        # constant accessory load [W]
SOC_MAX = 1.0 - 1e-6    # v_oc is singular at 1
BATTERY_EFF = 0.9       # Coulombic efficiency applied to the current
BATTERY_TEMP_C = 25.0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_params_file() -> Path:
    return _repo_root() / "data" / "params" / "Range_Extender_Class_8_Parameters.mat"


def cycle_file(name_or_path) -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p
    cand = _repo_root() / "data" / "cycles" / f"{name_or_path}.mat"
    if cand.exists():
        return cand
    raise FileNotFoundError(f"drive cycle {name_or_path!r} not found (looked in data/cycles)")


class DriveCycle:
    def __init__(self, path):
        d = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
        self.v = np.asarray(d["speed_vector"], float).squeeze()
        self.a = np.asarray(d["acceleration_vector"], float).squeeze()
        self.grad = np.asarray(d["elevation_vector"], float).squeeze()
        self.gear = np.asarray(d["gearnumber_vectorRUL"], float).squeeze()
        self.N = len(self.v)
        self.path = Path(path)

    def distance_miles(self) -> np.ndarray:
        t = np.arange(1, self.N + 1, dtype=float)
        dx = np.diff(t)
        avg = (self.v[:-1] + self.v[1:]) / 2
        return np.concatenate(([0.0], np.cumsum(dx * avg))) / 1609.34


class HEVSimulator:
    def __init__(self, param_file=None, drive_file=None, battery: str = "BAT_LFP"):
        self.params = sio.loadmat(str(param_file or default_params_file()), squeeze_me=True, struct_as_record=False)
        self.drive = DriveCycle(cycle_file(drive_file))
        self.v, self.a, self.grad, self.gear, self.N = self.drive.v, self.drive.a, self.drive.grad, self.drive.gear, self.drive.N
        p = self.params
        self.BAT, self.ENG, self.GEN, self.MOT, self.TRAN, self.ENV, self.VEH = (
            p[battery], p["ENG"], p["GEN"], p["MOT"], p["TRAN"], p["ENV"], p["VEH"])
        self.ENG.mf = np.asarray(self.ENG.mf, dtype=float)
        self.T_K = BATTERY_TEMP_C + 273.15
        self.Ts = 1.0

        # vehicle constants
        self.r, self.rho, self.Af, self.Cd, self.Cr = map(float, (self.VEH.r, self.ENV.rho, self.VEH.Af, self.VEH.Cd, self.VEH.Cr))
        self.M, self.Mv, self.g = map(float, (self.VEH.M, self.VEH.Mv, self.ENV.g))
        self.r_gear = np.asarray(self.TRAN.r_gear, float).ravel()
        self.gb_eff = float(self.TRAN.gb_effi)

        # motor maps
        self.Tm_list = np.asarray(self.MOT.Tm_list, float).ravel()
        self.wm_list = np.asarray(self.MOT.wm_list, float).ravel()
        self.max_wm = float(self.wm_list[-1])
        self.interp_Tmmax = interp1d(np.asarray(self.MOT.max_wm_list, float).ravel(), np.asarray(self.MOT.Tmmax, float).ravel(), kind="linear", fill_value="extrapolate")
        self.interp_Tmmin = interp1d(np.asarray(self.MOT.min_wm_list, float).ravel(), np.asarray(self.MOT.Tmmin, float).ravel(), kind="linear", fill_value="extrapolate")
        self.eff_interp = RegularGridInterpolator((self.wm_list, self.Tm_list), np.asarray(self.MOT.etam, float), bounds_error=False, fill_value=1.0)

        # battery tables
        temp_list = np.asarray(self.BAT.temp_list, float).ravel()
        soc_list = np.asarray(self.BAT.soc_list, float).ravel()
        self.r_dis_interp = RegularGridInterpolator((temp_list, soc_list), np.asarray(self.BAT.battery_soc_Rdis, float).T, method="linear", bounds_error=False)
        self.r_chg_interp = RegularGridInterpolator((temp_list, soc_list), np.asarray(self.BAT.battery_soc_Rchg, float).T, method="linear", bounds_error=False)
        # Pack temperature is fixed, so the 2-D tables collapse to one curve each over SOC.
        self._soc_grid = np.asarray(soc_list, float).ravel()
        self._r_dis = np.array([float(self.r_dis_interp([[BATTERY_TEMP_C, s]]).item()) for s in self._soc_grid])
        self._r_chg = np.array([float(self.r_chg_interp([[BATTERY_TEMP_C, s]]).item()) for s in self._soc_grid])
        self.v0, self.alpha0, self.beta, self.gamma_b, self.zeta_b, self.epsilon_b, self.Ns, self.Np = map(
            float, (self.BAT.v0, self.BAT.alpha0, self.BAT.beta, self.BAT.gamma, self.BAT.zeta, self.BAT.epsilon, self.BAT.Ns, self.BAT.Np))
        self.cap_Ah = self.Np * float(self.BAT.Bat_cap)
        self.Ib_min, self.Ib_max = float(self.BAT.Ib_min), float(self.BAT.Ib_max)

    # ------------------------------------------------------------------ physics
    def v_oc(self, soc: float) -> float:
        return self.Ns * (self.v0 + self.alpha0 * (1 - np.exp(-self.beta * soc)) + self.gamma_b * soc
                          + self.zeta_b * (1 - np.exp(-self.epsilon_b / (1 - soc))))

    def motor_power_demand(self) -> np.ndarray:
        # Motor power demand (W) for the whole cycle.
        out = np.empty(self.N)
        for k in range(self.N):
            v, grad, gear = self.v[k], self.grad[k], self.gear[k]
            wg = self.r_gear[int(gear) - 1] * v / self.r
            Tv = (self.Cr * self.M * self.g * np.cos(grad)
                  + 0.5 * self.rho * self.Cd * self.Af * v ** 2
                  + self.Mv * self.a[k] + self.M * self.g * np.sin(grad)) * self.r
            T = Tv * (v / self.r) / wg if wg > 0.0 else 0.0
            T = float(np.clip(T, float(self.interp_Tmmin(wg)), float(self.interp_Tmmax(wg))))
            e = float(self.eff_interp([[np.clip(wg, self.wm_list.min(), self.wm_list.max()),
                                        np.clip(abs(T), self.Tm_list.min(), self.Tm_list.max())]])[0])
            out[k] = (T * wg) / e if T >= 0 else (T * wg) * e
        return out

    def evaluate_step(self, k: int, SOC_k: float, we: float, Te: float):
        # Advance the plant by Ts = 1 s.
        v, a, grad, gear = self.v[k], self.a[k], self.grad[k], self.gear[k]
        rg = self.r_gear[int(gear) - 1]

        wv = v / self.r
        wg = rg * wv  # motor shaft speed
        Tv = (self.Cr * self.M * self.g * np.cos(grad) + 0.5 * self.rho * self.Cd * self.Af * v ** 2
              + self.Mv * a + self.M * self.g * np.sin(grad)) * self.r
        power_req = Tv * wv

        Tm_max, Tm_min = float(self.interp_Tmmax(wg)), float(self.interp_Tmmin(wg))
        T_motor = power_req / wg if wg > 0.0 else 0.0
        T_motor = float(np.clip(T_motor, Tm_min, Tm_max))
        Tm_clipped = np.clip(abs(T_motor), self.Tm_list.min(), self.Tm_list.max())
        wg_clipped = np.clip(wg, self.wm_list.min(), self.wm_list.max())
        e_m = float(self.eff_interp([[wg_clipped, Tm_clipped]])[0])
        P_motor_effective = (T_motor * wg) / e_m if T_motor >= 0 else (T_motor * wg) * e_m

        mf_out, Peng, gen_eff, Pgen_out, we, Te = find_best_engine_operating_point_vectorized(self.ENG, self.GEN, we, Te)
        P_engine, Pgen = Peng, Pgen_out
        if P_motor_effective >= -0.01:
            P_batt = P_motor_effective - Pgen
        else:  # regeneration: engine off, everything goes to the battery
            P_engine, Pgen = 0.0, 0.0
            P_batt = P_motor_effective
        P_batt = P_batt + P_AUX_W

        V_oc = self.v_oc(SOC_k)
        r_bat = float(np.interp(SOC_k, self._soc_grid, self._r_dis if P_batt >= 0 else self._r_chg))
        disc = max(V_oc ** 2 - 4 * r_bat * P_batt, 0.0)
        Ib = BATTERY_EFF * (V_oc - np.sqrt(disc)) / (2 * r_bat)
        Ib = min(max(Ib, self.Ib_min), self.Ib_max)
        SOC_next = min(max(SOC_k - Ib / (self.cap_Ah * 3600.0) * self.Ts, 0.0), SOC_MAX)

        next_state = np.array([SOC_next, k, P_motor_effective])
        out = SimpleNamespace(m_dot_fuel=mf_out, next_state=next_state, P_motor_effective=P_motor_effective,
                              P_batt=P_batt, T_motor=T_motor, wg=wg, P_engine=P_engine, Pgen=Pgen, Te=Te, we=we,
                              Ib=Ib, V_oc=V_oc, SOC=SOC_next)
        return next_state, mf_out, P_motor_effective, P_batt, T_motor, wg, P_engine, Pgen, out, we, Te


def fuel_economy_mpg(m_dot_fuel_g_per_s, speed_m_per_s, density_kg_per_l: float = 0.84245) -> float:
    # MPG the way the MATLAB/notebook post-processing computes it (trapezoidal integration).
    mf = np.asarray(m_dot_fuel_g_per_s, float)
    v = np.asarray(speed_m_per_s, float)
    t = np.arange(1, len(v) + 1, dtype=float)
    dist_miles = np.concatenate(([0.0], np.cumsum(np.diff(t) * (v[:-1] + v[1:]) / 2))) / 1609.34
    gal_per_s = (mf * 1e-3 / density_kg_per_l) / 3.78541
    gal = float(np.sum(np.diff(t) * (gal_per_s[:-1] + gal_per_s[1:]) / 2))
    return float(dist_miles[-1] / gal) if gal > 0 else float("inf")


def stack_outputs(out_list) -> dict:
    keys = vars(out_list[0]).keys()
    return {k: np.array([getattr(o, k) for o in out_list]) for k in keys}
