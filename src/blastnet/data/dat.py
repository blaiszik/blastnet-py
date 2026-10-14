"""The shipped BLASTNet Momentum128 on-disk layout: raw .dat cubes, metadata CSVs, constants.

    <root>/dataset/HR/<split>/<VAR>_id<hash>.dat             float32, n^3 (n = 128), C order (x, y, z)
    <root>/dataset/LR_{r}x/<split>/<VAR>_id<hash>.dat        float32, (n/r)^3
    <root>/<split>_data_summary.csv, <root>/all_data_summary.csv
    <root>/best_model_weights/...

Channel order is (rho, u, v, w) everywhere, as upstream.
"""

from __future__ import annotations

import os
import shutil
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

__all__ = [
    "CHANNELS",
    "COLUMNS",
    "DX_MIN",
    "RATIOS",
    "SPLITS",
    "VARS",
    "VECTOR_CHANNELS",
    "cube_path",
    "load_meta",
    "norm_stats",
    "read_cube",
    "read_dat",
    "subset",
]

CHANNELS = ("rho", "u", "v", "w")
VECTOR_CHANNELS = (1, 2, 3)
VARS = ("RHO_kgm-3", "UX_ms-1", "UY_ms-1", "UZ_ms-1")
SPLITS = ("train", "val", "test", "forcedhit", "paramvar")
RATIOS = (8, 16, 32)
COLUMNS = ("hash", "kaggle_data_id", "case_description", "kmeans_cluster_idx",
           "dx [m]", "dy [m]", "dz [m]")
DX_MIN = 3.906250185536919e-06
"""Upstream divides dx, dy, dz by this inside its dataset; blastnet returns metres instead."""

# Upstream common/data.py get_mean_std{,_test,_paramvar,_forcedhit}, (rho, u, v, w).
_STATS = {
    "train": ([0.24, 28.0, 28.0, 28.0], [0.068, 48.0, 48.0, 48.0]),
    "test": ([0.24, 29.0, 29.0, 29.0], [0.068, 48.0, 48.0, 48.0]),
    "paramvar": ([0.23, 34.0, 34.0, 34.0], [0.059, 55.0, 55.0, 55.0]),
    "forcedhit": ([11, -0.051, -0.051, -0.051], [4.6, 1.4, 1.4, 1.4]),
}
_STATS["val"] = _STATS["train"]


def norm_stats(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream (mean[4], std[4]) float32 for a split: train/val, test, paramvar, forcedhit."""
    if split not in _STATS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    mean, std = _STATS[split]
    return torch.tensor(mean), torch.tensor(std)


def read_dat(path: str | os.PathLike, shape: tuple[int, ...]) -> np.memmap:
    """Read-only float32 memmap of a raw .dat file, checked against `shape`."""
    expected = int(np.prod(shape)) * 4
    size = os.path.getsize(path)
    if size != expected:
        raise ValueError(f"{path}: {size} bytes, expected {expected} for float32 {shape}")
    return np.memmap(path, dtype=np.float32, mode="r", shape=tuple(shape))


def _res_dir(ratio: int | None) -> str:
    return "HR" if ratio in (None, 1) else f"LR_{ratio}x"


def cube_path(root: str | os.PathLike, split: str, hash_: str, var: str,
              ratio: int | None = None) -> Path:
    """Path of one variable of one cube; ratio None (or 1) is HR, else LR_{ratio}x."""
    return Path(root) / "dataset" / _res_dir(ratio) / split / f"{var}_id{hash_}.dat"


def read_cube(root: str | os.PathLike, split: str, hash_: str, ratio: int | None = None,
              hr_size: int = 128) -> list[np.memmap]:
    """The four (rho, u, v, w) memmaps of one cube at HR (ratio None) or LR_{ratio}x."""
    n = hr_size if ratio in (None, 1) else hr_size // ratio
    return [read_dat(cube_path(root, split, hash_, v, ratio), (n, n, n)) for v in VARS]


def _read_csv_raw(path: str | os.PathLike) -> pd.DataFrame:
    # every cell as the exact string on disk (hash stays a signed string, empty stays "")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_meta(root: str | os.PathLike, split: str) -> pd.DataFrame:
    """<split>_data_summary.csv: hash as str, float dx/dy/dz (empty dy/dz = dx), split column."""
    return _parse_meta(_read_csv_raw(Path(root) / f"{split}_data_summary.csv"), split)


def _parse_meta(raw: pd.DataFrame, split: str) -> pd.DataFrame:
    df = raw.copy()
    df["kmeans_cluster_idx"] = pd.to_numeric(df["kmeans_cluster_idx"]).astype("int64")
    for c in ("dx [m]", "dy [m]", "dz [m]"):
        df[c] = pd.to_numeric(df[c].replace("", np.nan)).astype("float64")
    df["dx"] = df["dx [m]"]
    df["dy"] = df["dy [m]"].fillna(df["dx"])
    df["dz"] = df["dz [m]"].fillna(df["dx"])
    df["split"] = split
    return df


def subset(root: str | os.PathLike, out: str | os.PathLike, n_per_split: int = 2,
           ratios: tuple[int, ...] | None = None, splits: tuple[str, ...] = SPLITS,
           seed: int | None = None) -> Path:
    """Copy n cubes per split (HR + every LR ratio) and their CSV rows into a new tree at `out`.

    seed None takes the first n rows of each split CSV; an int takes a seeded random sample of n
    rows (kept in CSV order). ratios None copies every LR_{r}x directory present under root.
    """
    root, out = Path(root), Path(out)
    if ratios is None:
        ratios = tuple(sorted(int(p.name[3:-1]) for p in (root / "dataset").glob("LR_*x")))
    keep: set[str] = set()
    for split in splits:
        raw = _read_csv_raw(root / f"{split}_data_summary.csv")
        n = min(n_per_split, len(raw))
        if seed is None:
            rows = raw.iloc[:n]
        else:
            rng = np.random.default_rng([seed, zlib.crc32(split.encode())])
            idx = np.sort(rng.choice(len(raw), n, replace=False))
            rows = raw.iloc[idx]
        for r in (None, *ratios):
            d = out / "dataset" / _res_dir(r) / split
            d.mkdir(parents=True, exist_ok=True)
            for h in rows["hash"]:
                for v in VARS:
                    shutil.copyfile(cube_path(root, split, h, v, r), d / f"{v}_id{h}.dat")
        rows.to_csv(out / f"{split}_data_summary.csv", index=False)
        keep.update(rows["hash"])
    all_csv = root / "all_data_summary.csv"
    if all_csv.exists():
        raw = _read_csv_raw(all_csv)
        raw[raw["hash"].isin(keep)].to_csv(out / "all_data_summary.csv", index=False)
    return out
