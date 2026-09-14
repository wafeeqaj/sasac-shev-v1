# Weight conversion between PyTorch and the MATLAB ``actor_struct`` format.
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.io import loadmat


def torch_key_to_mat(key: str) -> str:
    k = key.replace(".", "_")
    if "transformer" in k:
        k = k.replace("transformer", "trans")
    return k


def read_actor_struct(path) -> Dict[str, np.ndarray]:
    m = loadmat(str(path))
    if "actor_struct" not in m:
        raise KeyError(f"{path} has no 'actor_struct' variable (keys: {[k for k in m if not k.startswith('__')]})")
    s = m["actor_struct"]
    return {name: np.asarray(s[name][0, 0]) for name in s.dtype.names}


def match_keys(model_keys: List[str], mat_keys: List[str]) -> Tuple[Dict[str, str], List[str], List[str]]:
    # Return ``(torch_key -> mat_key, missing_torch_keys, unused_mat_keys)``.
    mat_set = set(mat_keys)
    mapping, missing = {}, []
    for k in model_keys:
        mk = torch_key_to_mat(k)
        if mk in mat_set:
            mapping[k] = mk
        else:
            missing.append(k)
    unused = sorted(mat_set - set(mapping.values()))
    return mapping, missing, unused


def load_state_dict_from_mat(module, path, strict_params: bool = True):
    # Load a ``.mat`` ``actor_struct`` into ``module`` in place; returns ``(missing, unused)``.
    import torch

    struct = read_actor_struct(path)
    sd = module.state_dict()
    mapping, missing, unused = match_keys(list(sd.keys()), list(struct.keys()))
    param_names = {n for n, _ in module.named_parameters()}
    missing_params = [k for k in missing if k in param_names]
    if strict_params and missing_params:
        raise KeyError(f"parameters missing from {path}: {missing_params}")
    new_sd = {}
    for tk, mk in mapping.items():
        arr = np.asarray(struct[mk], dtype=np.float32)
        tgt = sd[tk]
        if arr.shape != tuple(tgt.shape):
            # MATLAB drops trailing singleton dims / stores vectors as (1, n)
            arr = arr.reshape(tuple(tgt.shape))
        new_sd[tk] = torch.as_tensor(arr, dtype=tgt.dtype)
    module.load_state_dict(new_sd, strict=False)
    return missing, unused


def infer_architecture(struct: Dict[str, np.ndarray]) -> Dict[str, object]:
    # Guess actor type / sizes from the key names and shapes of an ``actor_struct``.
    keys = set(struct)
    if any(k.startswith("trans_") for k in keys):
        hidden = int(struct["embed_state_weight"].shape[0])
        n_layers = 1 + max(int(k.split("_")[2]) for k in keys if k.startswith("trans_h_"))
        n_ctx = int(struct["trans_wpe_weight"].shape[0])
        max_ep_len = int(struct["embed_timestep_weight"].shape[0])
        return {"actor": "dt", "hidden_size": hidden, "num_layers": n_layers, "n_ctx": n_ctx,
                "max_ep_len": max_ep_len, "context": n_ctx // 3,
                "state_dim": int(struct["embed_state_weight"].shape[1]),
                "action_dim": int(struct["embed_action_weight"].shape[1])}
    if any(k.startswith("gru_") for k in keys):
        hidden = int(struct["gru_weight_hh_l0"].shape[1])
        n_layers = 1 + max(int(k[-1]) for k in keys if k.startswith("gru_weight_hh_l"))
        return {"actor": "gru", "hidden": hidden, "gru_layers": n_layers,
                "state_dim": int(struct["gru_weight_ih_l0"].shape[1]),
                "action_dim": int(struct["mean_weight"].shape[0])}
    if "net_0_weight" in keys:
        return {"actor": "ffn", "hidden": int(struct["net_0_weight"].shape[0]),
                "state_dim": int(struct["net_0_weight"].shape[1]),
                "action_dim": int(struct["mean_weight"].shape[0])}
    raise ValueError("unrecognised actor_struct layout")
