# Transition replay (FFN) and episodic sequence replay (GRU/DT). numpy only.
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


class TransitionReplayBuffer:
    # Uniform replay over (s, a, s', r, done).

    def __init__(self, max_size: int = 10_000_000, seed: Optional[int] = None, sampling: str = "random"):
        self.storage: list = []
        self.max_size = int(max_size)
        self.ptr = 0
        self.sample_ptr = 0
        if sampling not in ("random", "sequential"):
            raise ValueError("sampling must be 'random' or 'sequential'")
        self.sampling = sampling
        self.rng = np.random.default_rng(seed)

    def add(self, state, action, next_state, reward, done) -> None:
        item = (np.asarray(state, dtype=np.float32), np.asarray(action, dtype=np.float32),
                np.asarray(next_state, dtype=np.float32), np.float32(reward), np.float32(done))
        if len(self.storage) < self.max_size:
            self.storage.append(item)
        else:
            self.storage[self.ptr] = item
        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        n = len(self.storage)
        if self.sampling == "sequential":
            if n < batch_size:
                raise ValueError("Not enough elements in buffer to sample the batch")
            end = self.sample_ptr + batch_size
            batch = self.storage[self.sample_ptr:end] if end <= n else self.storage[self.sample_ptr:] + self.storage[: end % n]
            self.sample_ptr = (self.sample_ptr + batch_size) % n
        else:
            ind = self.rng.integers(0, n, size=batch_size)
            batch = [self.storage[i] for i in ind]
        s, a, ns, r, d = map(np.stack, zip(*batch))
        return s, a, ns, r[:, None], d[:, None]

    def __len__(self) -> int:
        return len(self.storage)


@dataclass
class SequenceBatch:
    states: np.ndarray        # [B, T, S]
    actions: np.ndarray       # [B, T, A]
    next_states: np.ndarray   # [B, T, S]
    rewards: np.ndarray       # [B, T, 1]
    dones: np.ndarray         # [B, T, 1]
    timesteps: np.ndarray     # [B, T]  position within the window, 0..T-1
    returns_to_go: Optional[np.ndarray] = None  # [B, T, 1]

    def __len__(self) -> int:
        return int(self.states.shape[0])


class _Episode(list):
    # One stored episode, with a cache of its stacked arrays.

    __slots__ = ("arrays", "n_cached")

    def __init__(self):
        super().__init__()
        self.arrays = None
        self.n_cached = -1


class EpisodicSequenceBuffer:
    def __init__(self, capacity_episodes: int, context: int, store_rtg: bool = False, seed: Optional[int] = None):
        self.max_size = int(capacity_episodes)
        self.context = int(context)
        self.store_rtg = store_rtg
        self.buffer: deque = deque(maxlen=self.max_size)
        self.rng = np.random.default_rng(seed)

    def add(self, state, action, next_state, reward, done, return_to_go: Optional[float] = None) -> None:
        if len(self.buffer) == 0 or self.buffer[-1][-1][4]:  # last stored transition was terminal
            self.buffer.append(_Episode())
        item = [np.asarray(state, dtype=np.float32), np.asarray(action, dtype=np.float32),
                np.asarray(next_state, dtype=np.float32), np.float32(reward), bool(done)]
        if self.store_rtg:
            item.append(np.float32(0.0 if return_to_go is None else return_to_go))
        self.buffer[-1].append(tuple(item))

    def _stacked(self, episode: "_Episode"):
        # Stack this episode's transitions once and cache them.
        if episode.arrays is None or episode.n_cached != len(episode):
            parts = list(zip(*episode))
            episode.arrays = (
                np.stack(parts[0]), np.stack(parts[1]), np.stack(parts[2]),
                np.asarray(parts[3], dtype=np.float32), np.asarray(parts[4], dtype=np.float32),
                np.asarray(parts[5], dtype=np.float32) if self.store_rtg else None,
            )
            episode.n_cached = len(episode)
        return episode.arrays

    def sample(self, batch_size: int) -> SequenceBatch:
        # Draws from every episode except the newest, which is usually still being collected.
        n_avail = max(len(self.buffer) - 1, 1)
        sampled_episodes = self.rng.choice(n_avail, batch_size)
        cols = {k: [] for k in ("s", "a", "ns", "r", "d", "rtg")}
        for episode_idx in sampled_episodes:
            episode = self.buffer[int(episode_idx)]
            if len(episode) < self.context:
                continue
            start = int(self.rng.choice(len(episode) - self.context + 1))
            s, a, ns, r, d, rtg = self._stacked(episode)
            stop = start + self.context
            cols["s"].append(s[start:stop])
            cols["a"].append(a[start:stop])
            cols["ns"].append(ns[start:stop])
            cols["r"].append(r[start:stop])
            cols["d"].append(d[start:stop])
            if self.store_rtg:
                cols["rtg"].append(rtg[start:stop])
        if not cols["s"]:
            longest = max((len(e) for e in self.buffer), default=0)
            raise RuntimeError(
                f"no complete episode of at least context = {self.context} steps "
                f"(longest {longest})")
        return SequenceBatch(
            states=np.stack(cols["s"]),
            actions=np.stack(cols["a"]),
            next_states=np.stack(cols["ns"]),
            rewards=np.stack(cols["r"])[..., None],
            dones=np.stack(cols["d"])[..., None],
            # Window-local, so a given position is the same row for every sample.
            timesteps=np.tile(np.arange(self.context), (len(cols["s"]), 1)),
            returns_to_go=np.stack(cols["rtg"])[..., None] if self.store_rtg else None,
        )

    def __len__(self) -> int:
        return len(self.buffer)
