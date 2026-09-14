# Per-episode training log, written as ``.mat`` and ``.csv``.
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.io import loadmat, savemat

# Order matches the savemat dict of the original runs.
FIELDS: List[str] = [
    "episode", "rewards", "avg_alpha", "avg_speeds", "avg_torques", "socs", "initial_soc_list",
    "avg_fuel_list", "pow_demand_list", "duration_mult_list", "torq_switch_counts", "speed_switch_counts",
    "time_list", "q1_loss_list", "q2_loss_list", "actor_loss_list", "alpha_loss_list", "steps_list",
    "q_action_sens",
]


class TrainingHistory:
    def __init__(self):
        self.data: Dict[str, list] = {k: [] for k in FIELDS}

    def append(self, **kw) -> None:
        for k in FIELDS:
            self.data[k].append(kw.get(k, np.nan))

    def __getitem__(self, k: str) -> list:
        return self.data[k]

    def __len__(self) -> int:
        return len(self.data["episode"])

    def to_mat(self, path, struct_name: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        savemat(str(path), {struct_name: {k: np.asarray(v, dtype=float) for k, v in self.data.items()}})

    def to_csv(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(FIELDS)
            for i in range(len(self)):
                w.writerow([self.data[k][i] for k in FIELDS])

    @classmethod
    def from_mat(cls, path) -> "TrainingHistory":
        m = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        key = [k for k in m if not k.startswith("__")][0]
        s = m[key]
        h = cls()
        for k in FIELDS:
            if hasattr(s, k):
                h.data[k] = list(np.atleast_1d(getattr(s, k)))
        return h
