"""Cube sampling from general BLASTNet cases, for super-resolution training pairs.

    ds = CubeDataset([open_case(p) for p in paths], size=128, ratio=8, mode="tile")
    item = ds[0]   # {"hr": [C, s, s, s], "lr": [C, s/r, ...], "dx", "case", "snapshot", "origin"}

LR cubes are made from the physical-unit HR cube with `blastnet.filters.make_lr`; `normalize`
(mean, std) is applied to both afterwards.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from blastnet import filters
from blastnet.data.dat import VARS
from blastnet.data.info import Case, open_case

__all__ = ["CubeDataset"]

MODES = ("tile", "random")


class CubeDataset(Dataset):
    """Size^3 cubes tiled or randomly drawn from BLASTNet cases.

    mode "tile": every cube of a (size-stride) lattice that fits, per case and snapshot (stride
    defaults to size, i.e. non-overlapping). mode "random": `n_random` cubes in total, each with a
    seeded random case, snapshot and origin (the same cubes on every run for a given seed).
    `snapshots` is None (all) or an iterable of snapshot indices applied to every case.
    `normalize` is None or (mean, std), scalars or per-channel sequences, applied to hr and lr.
    With `ratio`, `variables` must be the four momentum variables (the (rho, u, v, w) channels
    of `filters.make_lr`), in that order for lr_kind="favre_box".
    "dx" is float32[3] in metres, NaN on a non-uniform axis.
    """

    def __init__(self, cases, variables=VARS, size: int = 128, ratio: int | None = None,
                 mode: str = "tile", stride: int | None = None, n_random: int | None = None,
                 snapshots=None, seed: int = 0, lr_kind: str = "favre_box", normalize=None):
        if isinstance(cases, (Case, str, Path)):
            cases = [cases]
        self.cases = [c if isinstance(c, Case) else open_case(c) for c in cases]
        self.variables = tuple(variables)
        self.size, self.ratio, self.mode, self.lr_kind = size, ratio, mode, lr_kind
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if ratio is not None:
            if size % ratio:
                raise ValueError(f"size {size} is not a multiple of ratio {ratio}")
            if lr_kind not in filters.KINDS:
                raise ValueError(f"lr_kind must be one of {filters.KINDS}, got {lr_kind!r}")
            if len(self.variables) != 4:
                raise ValueError("ratio needs the 4 momentum variables (rho, u, v, w), got "
                                 f"{self.variables}")
            if lr_kind == "favre_box" and self.variables != VARS:
                raise ValueError(f"lr_kind='favre_box' needs variables == {VARS} in that order, "
                                 f"got {self.variables}")
        if normalize is None:
            self.normalize = None
        else:
            mean, std = (torch.as_tensor(np.asarray(a, dtype=np.float32)).reshape(-1, 1, 1, 1)
                         for a in normalize)
            self.normalize = (mean, std)
        for c in self.cases:
            missing = [v for v in self.variables if v not in c.variables]
            if missing:
                raise ValueError(f"case {c.id!r} has no variables {missing}")
            if any(n < size for n in c.shape):
                raise ValueError(f"case {c.id!r} shape {c.shape} is smaller than size {size}")
        self._dx = [np.array([np.nan if d is None else d for d in c.spacing()], dtype=np.float32)
                    for c in self.cases]

        def snaps(c: Case) -> list[int]:
            if snapshots is None:
                return list(range(c.n_snapshots))
            return [int(s) % c.n_snapshots for s in snapshots]

        items: list[tuple[int, int, tuple[int, int, int]]] = []
        if mode == "tile":
            step = size if stride is None else int(stride)
            if step < 1:
                raise ValueError(f"stride must be >= 1, got {stride}")
            for ci, c in enumerate(self.cases):
                origins = list(itertools.product(
                    *(range(0, n - size + 1, step) for n in c.shape)))
                items += [(ci, s, o) for s in snaps(c) for o in origins]
        else:
            if n_random is None:
                raise ValueError("mode='random' needs n_random")
            rng = np.random.default_rng(seed)
            pool = [(ci, s) for ci, c in enumerate(self.cases) for s in snaps(c)]
            for _ in range(n_random):
                ci, s = pool[int(rng.integers(len(pool)))]
                o = tuple(int(rng.integers(n - size + 1)) for n in self.cases[ci].shape)
                items.append((ci, s, o))
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        ci, snap, origin = self.items[i]
        case = self.cases[ci]
        region = tuple(slice(o, o + self.size) for o in origin)
        hr = torch.from_numpy(case.fields(self.variables, snap, region))
        out: dict = {}
        lr = None
        if self.ratio is not None:
            lr = filters.make_lr(hr[None], self.ratio, kind=self.lr_kind)[0]
        if self.normalize is not None:
            mean, std = self.normalize
            hr = (hr - mean) / std
            if lr is not None:
                lr = (lr - mean) / std
        out["hr"] = hr
        if lr is not None:
            out["lr"] = lr
        out.update(dx=torch.from_numpy(self._dx[ci].copy()), case=case.id,
                   snapshot=snap, origin=origin)
        return out
