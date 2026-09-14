# Load a trained actor and roll it through the plant (paper Sec. IV-B).
from __future__ import annotations

from pathlib import Path

import torch

from sasac.utils.mat_actors import build_actor

from .rollout import RolloutSettings, rollout, save_rollout_mat

CYCLES = ["FHDS_10", "US06_21", "HHDT_5"]
TAGS = {"ffn": "DNN", "gru": "GRU", "dt": "DT"}


def load_actor(agent: str, checkpoint, context: int = 100):
    # Accepts a training-run checkpoint or one rebuilt from a ``.mat`` export.
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "arch" in ck:
        arch = ck["arch"]
    else:
        cfg, dt = ck.get("config", {}), ck.get("config", {}).get("dt", {})
        arch = dict(actor=agent,
                    state_dim=ck.get("state_dim", 3), action_dim=ck.get("action_dim", 2),
                    hidden=cfg.get("hidden_actor", 128), gru_layers=cfg.get("gru_layers", 2),
                    context=cfg.get("context", context), **dt)
    actor = build_actor(arch)
    actor.load_state_dict(ck["actor"], strict=False)
    actor.eval()
    return actor


def validate(agent: str, checkpoint, cycles=None, settings=None, out_dir="results/validation",
             tag=None, verbose=True):
    # Roll one actor through every cycle, writing ``SAC_<tag>_<cycle>_test.mat``.
    settings = RolloutSettings.from_yaml(settings or f"configs/validation/{agent}.yaml")
    actor = load_actor(agent, checkpoint, settings.context)
    tag, out_dir = tag or TAGS[agent], Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for cyc in cycles or CYCLES:
        res = rollout(actor, settings, cyc, verbose=False)
        save_rollout_mat(res, out_dir / f"SAC_{tag}_{cyc}_test.mat", tag, cyc)
        results[cyc] = res
        if verbose:
            print(f"{tag:4s} {cyc:8s}  final SOC {100 * res['final_soc']:6.2f} %"
                  f"   MPG {res['mpg']:6.2f}")
    return results
