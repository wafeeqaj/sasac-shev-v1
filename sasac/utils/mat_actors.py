# Rebuild PyTorch actors from the MATLAB ``actor_struct`` exports.
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..models import DTActor, DTConfig, FFNActor, GRUActor
from .checkpoints import infer_architecture, load_state_dict_from_mat, read_actor_struct


def build_actor(arch: dict):
    # Construct an actor from an ``arch`` record (see :func:`infer_architecture`).
    kind, s, a = arch["actor"], arch["state_dim"], arch["action_dim"]
    if kind == "dt":
        return DTActor(s, a, DTConfig(hidden_size=arch.get("hidden_size", 128),
                                      num_heads=arch.get("num_heads", 4),
                                      num_layers=arch.get("num_layers", 1),
                                      context=arch.get("context", 100),
                                      n_ctx=arch.get("n_ctx"),
                                      max_ep_len=arch.get("max_ep_len")))
    if kind == "gru":
        return GRUActor(s, arch.get("hidden", 128), a, arch.get("gru_layers", 2))
    return FFNActor(s, a, arch.get("hidden", 128))


def _forward_is_finite(actor, arch: dict) -> bool:
    # One deterministic forward pass, to catch a mis-shaped load.
    torch.manual_seed(0)
    kind, s, a = arch["actor"], arch["state_dim"], arch["action_dim"]
    if kind == "ffn":
        out = actor.act_deterministic(torch.rand(1, s) * 2 - 1)
    else:
        k = arch.get("context", 10) if kind == "dt" else 10
        seq = torch.rand(1, k, s) * 2 - 1
        out = (actor.act_deterministic(seq) if kind == "gru" else
               actor.act_deterministic(seq, torch.zeros(1, k, a), torch.ones(1, k, 1),
                                       torch.arange(k).reshape(1, k)))
    return bool(np.all(np.isfinite(out.numpy())))


def convert(mat_path: Path, out_dir: Path, num_heads: int = 4) -> Path:
    # ``<name>_actor.mat`` -> ``<out_dir>/<name>_actor.pt``. ``num_heads`` is Table II's 4.
    mat_path, out_dir = Path(mat_path), Path(out_dir)
    arch = infer_architecture(read_actor_struct(mat_path))
    if arch["actor"] == "dt":
        arch["num_heads"] = int(num_heads)
    actor = build_actor(arch)
    load_state_dict_from_mat(actor, mat_path)
    actor.eval()
    if not _forward_is_finite(actor, arch):
        raise RuntimeError(f"non-finite forward pass after loading {mat_path.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{mat_path.stem}.pt"
    torch.save({"actor": actor.state_dict(), "arch": arch, "source_mat": mat_path.name}, out_path)
    print(f"{mat_path.name} -> {out_path.name}  ({arch['actor']})")
    return out_path


def convert_all(mat_dir="results/checkpoints/mat", out_dir="results/checkpoints"):
    # Convert every ``*_actor.mat`` in ``mat_dir``.
    return [convert(f, out_dir) for f in sorted(Path(mat_dir).glob("*_actor.mat"))]
