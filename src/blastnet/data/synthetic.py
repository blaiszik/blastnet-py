"""Synthetic Momentum128-like cubes and an on-disk tree in the shipped layout, for tests.

Velocity is a sum of transverse plane waves (each mode is curl of a vector potential, so the
continuum field is exactly divergence-free) with non-integer wavenumbers (non-periodic on the
grid), rescaled to the upstream train statistics (mean 28, std 48 m/s per component). Density is
a smooth positive field with mean 0.24 and std 0.068 kg/m^3. LR is the Favre (density-weighted)
block average of u, v, w and the block average of rho, as upstream my_favre_filter/my_box_filter.
"""

from __future__ import annotations

import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from blastnet.data.dat import COLUMNS, SPLITS, VARS, _parse_meta, _res_dir

__all__ = ["cubes", "write_tree"]

_CASES = {  # (kaggle_data_id, case_description, dx [m]) shaped like the shipped CSVs
    "default": [
        ("free-propagating-h2-vit-air-li-case-24", "(Kau = 8.9, Uin/SL = 3.67)",
         "4.0707236621528864e-05"),
        ("free-propagating-h2-vit-air-li-case-19", "(Kau = 19, Uin/SL = 5.51)",
         "3.3400810934836045e-05"),
        ("free-propagating-h2-vit-air-li-case-4", "(Kau = 13, Uin /SL = 2.45)",
         "2.0353618310764432e-05"),
    ],
    "forcedhit": [("free-propagating-ch4-ffcm", "Reacting Forced HIT", "2.6210155920119997e-06")],
}
_ANISO = ("round-jet-premixed-bfer", "Reacting Jet Flow (6 Species)",
          "1.8623321011546068e-05", "1.5846149835851975e-05")


def _box(x: np.ndarray, r: int) -> np.ndarray:
    # block mean over r^3 cells; equals uniform_filter(size=r)[r//2::r, ...] (upstream my_box_filter)
    n = x.shape[-1] // r
    return x.reshape(*x.shape[:-3], n, r, n, r, n, r).mean(axis=(-5, -3, -1))


def _favre(phi: np.ndarray, rho: np.ndarray, r: int) -> np.ndarray:
    return _box(rho * phi, r) / _box(rho, r)


def _lr(hr: np.ndarray, r: int) -> np.ndarray:
    # hr [..., 4, n, n, n] float32 -> LR float32, computed in float64
    h = hr.astype(np.float64)
    rho = h[..., :1, :, :, :]
    return np.concatenate([_box(rho, r), _favre(h[..., 1:, :, :, :], rho, r)], axis=-4).astype(
        np.float32)


def _field(rng: np.random.Generator, size: int, modes: int = 12) -> np.ndarray:
    # [4, n, n, n] float64: rho, then divergence-free velocity from random transverse waves
    grid = np.stack(np.meshgrid(*[np.arange(size, dtype=np.float64)] * 3, indexing="ij"))
    out = np.zeros((4, size, size, size))
    for _ in range(modes):
        d = rng.normal(size=3)
        k = d / np.linalg.norm(d) * rng.uniform(0.5, 2.0) * 2 * np.pi / size
        phase = np.tensordot(k, grid, axes=1) + rng.uniform(0, 2 * np.pi)
        a = rng.normal(size=3)
        out[1:] += np.cross(k, a)[:, None, None, None] * np.cos(phase)
        out[0] += rng.normal() * np.sin(phase + rng.uniform(0, 2 * np.pi))
    out[0] = 0.24 + 0.068 * (out[0] - out[0].mean()) / out[0].std()
    np.clip(out[0], 0.02, None, out=out[0])
    out[1:] = 28.0 + 48.0 * (out[1:] - out[1:].mean()) / out[1:].std()
    return out


def _meta(rng: np.random.Generator, n: int, split: str) -> pd.DataFrame:
    cases = _CASES.get(split, _CASES["default"])
    rows = []
    for i in range(n):
        h = str(int(rng.integers(-(2**63), 2**63 - 1, dtype=np.int64)))
        kid, desc, dx = cases[i % len(cases)]
        dy = dz = dx
        if i == 1 and split in ("train", "synthetic"):  # dx != dy = dz, like the round jets
            kid, desc, dx, dy = _ANISO
            dz = dy
        elif i == 1 and split == "val":  # empty dy/dz (the reader fills them with dx)
            dy = dz = ""
        rows.append((h, kid, desc, str(int(rng.integers(0, 20))), dx, dy, dz))
    return pd.DataFrame(rows, columns=list(COLUMNS), dtype=str)


def _generate(n: int, size: int, ratios: tuple[int, ...], seed: int, split: str):
    for r in ratios:
        if size % r:
            raise ValueError(f"size {size} is not a multiple of ratio {r}")
    rng = np.random.default_rng([seed, zlib.crc32(split.encode())])
    raw = _meta(rng, n, split)
    hr = np.stack([_field(rng, size) for _ in range(n)]).astype(np.float32)
    return raw, hr, {r: _lr(hr, r) for r in ratios}


def cubes(n: int, size: int = 16, ratio: int = 8, seed: int = 0) -> dict:
    """Deterministic synthetic cubes: dict(lr=[n,4,s/r,..], hr=[n,4,s,s,s] float32, meta=frame)."""
    raw, hr, lr = _generate(n, size, (ratio,), seed, "synthetic")
    return {"lr": torch.from_numpy(lr[ratio]), "hr": torch.from_numpy(hr),
            "meta": _parse_meta(raw, "synthetic")}


def write_tree(root: str | os.PathLike, n_per_split: int = 2, size: int = 16,
               ratios: tuple[int, ...] = (8,), splits: tuple[str, ...] = SPLITS,
               seed: int = 0) -> Path:
    """Write synthetic cubes in the shipped on-disk layout (HR edge `size`) under root."""
    root = Path(root)
    (root / "best_model_weights").mkdir(parents=True, exist_ok=True)
    frames = []
    for split in splits:
        raw, hr, lrs = _generate(n_per_split, size, tuple(ratios), seed, split)
        for r, arr in ((None, hr), *lrs.items()):
            d = root / "dataset" / _res_dir(r) / split
            d.mkdir(parents=True, exist_ok=True)
            for h, cube in zip(raw["hash"], arr):
                for v, ch in zip(VARS, cube):
                    np.ascontiguousarray(ch, dtype=np.float32).tofile(d / f"{v}_id{h}.dat")
        raw.to_csv(root / f"{split}_data_summary.csv", index=False)
        frames.append(raw)
    pd.concat(frames).to_csv(root / "all_data_summary.csv", index=False)
    return root
