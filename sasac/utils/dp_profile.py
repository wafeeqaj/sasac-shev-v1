# Convert the solved DP workspaces to plain arrays for Fig. 4 and Table IV.
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

# profile field -> name the figure and table code expects
FIELDS = {"SOC": "SOC", "Ib": "Ib", "Pow_em": "Pow_em", "mfdot": "m_dot_fuel"}
CYCLES = {"FHDS-10_DP.mat": "FHDS_10", "US06_DP.mat": "US06_21", "HHDT_5_DP.mat": "HHDT_5"}


def _flatten(struct_array, field) -> np.ndarray:
    # A (1, N) struct-array field -> a plain (N,) float array.
    out = np.empty(struct_array.size, dtype=float)
    for i, cell in enumerate(struct_array[field].ravel()):
        a = np.asarray(cell).ravel()
        out[i] = a[0] if a.size else np.nan
    return out


def export(mat_path, cycle: str, out_dir) -> Path:
    m = loadmat(str(mat_path))
    if "profile" not in m:
        raise KeyError(f"{Path(mat_path).name} has no 'profile' struct array")
    data = {out: _flatten(m["profile"], src) for src, out in FIELDS.items()}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"DP_{cycle}.mat"
    savemat(str(out_path), data, do_compression=True)
    print(f"{out_path.name}  {data['SOC'].size} steps, final SOC {100 * data['SOC'][-1]:.2f} %")
    return out_path


def export_all(dp_dir="results/dp"):
    # Export every workspace present. They are not shipped, so missing ones are skipped.
    dp_dir = Path(dp_dir)
    return [export(dp_dir / n, cyc, dp_dir / "plain")
            for n, cyc in CYCLES.items() if (dp_dir / n).exists()]
