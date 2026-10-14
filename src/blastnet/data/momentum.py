"""PyTorch datasets over BLASTNet Momentum128: the shipped .dat tree and its Zarr conversion.

Each item is {"lr": [4, c/r, c/r, c/r], "hr": [4, c, c, c], "dx": float32[3], "index": i,
"hash": str} with channels (rho, u, v, w) and c the crop edge (hr_size when crop is None).

Conventions (match upstream training unless noted):
- dx is the HR grid spacing (dx, dy, dz) in metres, empty dy/dz filled with dx. Upstream divides by
  DX_MIN = 3.906250185536919e-06 inside its dataset; here that scaling is left to the caller.
- normalize applies (x - mean) / std with norm_stats(split), before augmentation (upstream order:
  ScaleTransform, then rot90/flip on the normalised tensors, so flipped velocity is -normalised).
- crop is an HR edge (multiple of ratio). The LR offset is drawn uniformly on the LR grid and the
  HR offset is ratio times it, so lr is exactly the LR cells covering hr.
- augment draws one octahedral element g (uniform over all 48, as upstream's rot90 + flips
  generate O_h) and applies it to lr and hr jointly; dx is permuted with g.perm.
- Randomness (crop offset, then g) comes from a generator seeded by (seed, epoch, index), so
  items are reproducible; call set_epoch(e) each epoch (before building the loader iterator when
  workers are not persistent).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from blastnet import octahedral
from blastnet.data.dat import VECTOR_CHANNELS, _parse_meta, load_meta, norm_stats, read_cube

__all__ = ["Momentum128", "Momentum128Zarr"]


class Momentum128(Dataset):
    """Momentum128 cubes from the shipped .dat tree, memmapped per item (or preloaded to RAM)."""

    def __init__(self, root: str | os.PathLike, split: str = "train", ratio: int = 8,
                 crop: int | None = None, normalize: bool = True, augment: bool = False,
                 seed: int = 0, hr_size: int = 128, preload: bool = False):
        self.root = Path(root)
        self.split = split
        self.ratio = int(ratio)
        self.normalize = normalize
        self.augment = augment
        self.seed = int(seed)
        self.epoch = 0
        self.meta = self._load_meta()
        self.hr_size = self._hr_size(hr_size)
        if self.hr_size % self.ratio:
            raise ValueError(f"hr_size {self.hr_size} is not a multiple of ratio {self.ratio}")
        if crop is not None and (crop % self.ratio or not 0 < crop <= self.hr_size):
            raise ValueError(f"crop {crop} must be a multiple of ratio {self.ratio} "
                             f"in (0, {self.hr_size}]")
        self.crop = crop
        self.mean, self.std = norm_stats(split)
        self._dx = torch.tensor(self.meta[["dx", "dy", "dz"]].to_numpy(), dtype=torch.float32)
        self._hr = self._lr = None
        if preload:
            self._preload()

    # --- storage backend (overridden by Momentum128Zarr) ---
    def _load_meta(self):
        return load_meta(self.root, self.split)

    def _hr_size(self, hr_size: int) -> int:
        return int(hr_size)

    def _read_raw(self, i: int, hr_sl: tuple, lr_sl: tuple) -> tuple[np.ndarray, np.ndarray]:
        h = self.meta["hash"].iat[i]
        lr = np.stack([m[lr_sl] for m in read_cube(self.root, self.split, h, self.ratio,
                                                   self.hr_size)])
        hr = np.stack([m[hr_sl] for m in read_cube(self.root, self.split, h, None,
                                                   self.hr_size)])
        return lr, hr

    def _preload(self) -> None:
        n, nl, full = self.hr_size, self.hr_size // self.ratio, (slice(None),) * 3
        self._lr = np.empty((len(self), 4, nl, nl, nl), dtype=np.float32)
        self._hr = np.empty((len(self), 4, n, n, n), dtype=np.float32)
        for i in range(len(self)):
            self._lr[i], self._hr[i] = self._read_raw(i, full, full)

    # --- public API ---
    def __len__(self) -> int:
        return len(self.meta)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch mixed into the per-item seed (crop offset and augmentation element)."""
        self.epoch = int(epoch)

    def sample_params(self, i: int) -> tuple[tuple[int, int, int] | None,
                                             octahedral.Element | None]:
        """(HR crop offset or None, octahedral element or None) that __getitem__(i) will use."""
        state = np.random.SeedSequence([self.seed, self.epoch, int(i)]).generate_state(2)
        gen = torch.Generator().manual_seed(int(state[0]) << 32 | int(state[1]))
        offset = None
        if self.crop is not None:
            span = (self.hr_size - self.crop) // self.ratio + 1
            offset = tuple(int(o) * self.ratio for o in torch.randint(span, (3,), generator=gen))
        g = None
        if self.augment:  # same draw as octahedral.random
            g = octahedral.elements()[int(torch.randint(48, (1,), generator=gen).item())]
        return offset, g

    def __getitem__(self, i: int) -> dict:
        """The contract item dict for cube i (crop, normalize, augment as configured)."""
        if not 0 <= i < len(self):
            raise IndexError(i)
        offset, g = self.sample_params(i)
        if offset is None:
            hr_sl = lr_sl = (slice(None),) * 3
        else:
            c, r = self.crop, self.ratio
            hr_sl = tuple(slice(o, o + c) for o in offset)
            lr_sl = tuple(slice(o // r, (o + c) // r) for o in offset)
        if self._hr is not None:
            lr, hr = self._lr[(i, slice(None), *lr_sl)], self._hr[(i, slice(None), *hr_sl)]
        else:
            lr, hr = self._read_raw(i, hr_sl, lr_sl)
        lr, hr = torch.from_numpy(np.array(lr)), torch.from_numpy(np.array(hr))
        if self.normalize:
            mean, std = self.mean[:, None, None, None], self.std[:, None, None, None]
            lr, hr = (lr - mean) / std, (hr - mean) / std
        dx = self._dx[i].clone()
        if g is not None:
            lr = octahedral.apply(lr, g, VECTOR_CHANNELS)
            hr = octahedral.apply(hr, g, VECTOR_CHANNELS)
            dx = dx[list(g.perm)]
        return {"lr": lr, "hr": hr, "dx": dx, "index": int(i), "hash": str(self.meta["hash"].iat[i])}


class Momentum128Zarr(Momentum128):
    """Momentum128 read from a zarr_convert store; same items and options as Momentum128."""

    def __init__(self, store: str | os.PathLike, split: str = "train", ratio: int = 8,
                 crop: int | None = None, normalize: bool = True, augment: bool = False,
                 seed: int = 0, preload: bool = False):
        self._arrays = None
        super().__init__(store, split, ratio, crop, normalize, augment, seed, preload=preload)

    def _group(self):
        import zarr

        if self._arrays is None:
            g = zarr.open_group(self.root, mode="r")[self.split]
            self._arrays = (g, g["hr"], g[f"lr_{self.ratio}x"])
        return self._arrays

    def _load_meta(self):
        import pandas as pd

        attrs = self._group()[0].attrs
        return _parse_meta(pd.DataFrame(attrs["rows"], columns=attrs["columns"], dtype=str),
                           self.split)

    def _hr_size(self, hr_size: int) -> int:
        return int(self._group()[1].shape[-1])

    def _read_raw(self, i: int, hr_sl: tuple, lr_sl: tuple) -> tuple[np.ndarray, np.ndarray]:
        _, hr, lr = self._group()
        return lr[(i, slice(None), *lr_sl)], hr[(i, slice(None), *hr_sl)]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arrays"] = None  # reopen in each DataLoader worker
        return state
