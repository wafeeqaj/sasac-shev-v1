# Engine + generator feasibility and fuel lookup for the validation plant.
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from ._interp import bilinear


def _interps(ENG, GEN):
    cache = getattr(ENG, "_sasac_interp_cache", None)
    if cache is not None:
        return cache
    we_grid = np.asarray(ENG.speed_range, dtype=float).ravel()
    Te_grid = np.asarray(ENG.Torq_range, dtype=float).ravel()
    Tmax_engine = interp1d(np.asarray(ENG.maxTrq_w, float).ravel(), np.asarray(ENG.max_torque_v, float).ravel(),
                           kind="linear", fill_value="extrapolate")(we_grid)
    Tmax_gen = interp1d(np.asarray(GEN.maxTrqCrv_w, float).ravel(), np.asarray(GEN.maxTrqCrv_trq_peak, float).ravel(),
                        kind="linear", fill_value="extrapolate")(we_grid)
    cache = {
        "maxTrq_w": np.asarray(ENG.maxTrq_w, float).ravel(),
        "Tmax_both": np.minimum(Tmax_engine, Tmax_gen),
        "we": we_grid, "Te": Te_grid, "mf": np.asarray(ENG.mf, float),
        "gw": np.asarray(GEN.effMapPos_w, float).ravel(),
        "gt": np.asarray(GEN.effMapPos_trq, float).ravel(),
        "ge": np.asarray(GEN.effMapPos_eff, float),
    }
    try:
        ENG._sasac_interp_cache = cache
    except Exception:  # pragma: no cover - read-only struct
        pass
    return cache


def find_best_engine_operating_point_vectorized(ENG, GEN, speed, torq):
    # Return ``(mf, Peng, gen_eff, Pgen_output, speed, torq)`` for one engine command.
    c = _interps(ENG, GEN)
    speed_idx = int(np.abs(c["maxTrq_w"] - speed).argmin())
    max_torq = c["Tmax_both"][speed_idx]
    if torq > max_torq:
        torq = max_torq
    mf = float(bilinear(c["we"], c["Te"], c["mf"], speed, torq, 0.0))
    Peng = speed * torq
    eff = float(bilinear(c["gw"], c["gt"], c["ge"], speed, torq, np.nan))
    if np.isnan(eff) or eff <= 0 or eff > 1.5:
        gen_eff, Pgen_output = 0.0, 0.0
    else:
        gen_eff, Pgen_output = eff, eff * Peng
    return mf, Peng, gen_eff, Pgen_output, speed, torq
