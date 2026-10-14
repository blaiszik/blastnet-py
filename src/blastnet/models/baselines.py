"""Interpolation baselines: nearest, trilinear, tricubic upsampling as `nn.Module`s.

`Interp(ratio, mode)` maps [B, C, n, n, n] -> [B, C, n*r, n*r, n*r] with no parameters. Every
mode is a linear map whose weights sum to 1 along each axis, so it is affine-invariant per
channel (`interp(a*x + b) == a*interp(x) + b`) and acts on normalised tensors directly.

    nearest    each voxel repeated r times per axis (`repeat_interleave`)
    trilinear  `F.interpolate(mode="trilinear", align_corners=False)` (cell-centred, matching
               block-average LR)
    tricubic   the upstream BLASTNet 2.0 cubic baseline (`create_cubic_files.ipynb`):
               `scipy.ndimage.zoom(arr, r, order=3, mode="nearest")`, i.e. cubic B-spline
               interpolation with corner-aligned grids (output i samples input i*(n-1)/(n*r-1)),
               a an edge-replicated 12-sample pad before a mirror-boundary spline prefilter. Implemented
               here as separable per-axis matrices in torch (no scipy).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

_PAD = 12  # scipy pads by 12 before spline filtering with mode="nearest"
MODES = ("nearest", "trilinear", "tricubic")


def _bspline3(t: torch.Tensor) -> torch.Tensor:
    t = t.abs()
    inner = 2 / 3 - t**2 + t**3 / 2
    outer = (2 - t) ** 3 / 6
    return torch.where(t < 1, inner, torch.where(t < 2, outer, torch.zeros_like(t)))


def tricubic_matrix(n: int, ratio: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """[n*ratio, n] 1-D operator equal to `scipy.ndimage.zoom(x, ratio, order=3, mode='nearest')`.

    Like scipy, the input is edge-replicated by `_PAD` samples, spline-filtered with mirror
    boundaries (solve the 1/6, 4/6, 1/6 collocation system), and sampled at corner-aligned
    positions i*(n-1)/(n*ratio-1) shifted by `_PAD`.
    """
    N = n * ratio
    if n == 1:
        return torch.ones(N, 1, dtype=dtype)
    m = n + 2 * _PAD
    A = torch.zeros(m, m, dtype=torch.float64)
    idx = torch.arange(m)
    A[idx, idx] = 4 / 6
    A[idx[:-1], idx[:-1] + 1] = 1 / 6
    A[idx[1:], idx[1:] - 1] = 1 / 6
    A[0, 1] = 2 / 6
    A[m - 1, m - 2] = 2 / 6
    P = torch.zeros(m, n, dtype=torch.float64)  # edge padding
    P[torch.arange(m), (torch.arange(m) - _PAD).clamp(0, n - 1)] = 1.0
    pos = torch.arange(N, dtype=torch.float64) * (n - 1) / (N - 1) + _PAD
    base = pos.floor().long()
    W = torch.zeros(N, m, dtype=torch.float64)
    for k in range(-1, 3):
        j = base + k
        w = _bspline3(pos - j.to(torch.float64))
        W.index_put_((torch.arange(N), j), w, accumulate=True)
    return (W @ torch.linalg.solve(A, P)).to(dtype)


class Interp(nn.Module):
    """Parameter-free upsampler: `mode` in nearest | trilinear | tricubic."""

    def __init__(self, ratio: int = 8, mode: str = "trilinear"):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown interpolation mode {mode!r}; expected one of {MODES}")
        self.ratio = int(ratio)
        self.mode = mode
        self._cache: dict[tuple, torch.Tensor] = {}

    def _matrix(self, n: int, x: torch.Tensor) -> torch.Tensor:
        key = (n, x.dtype, x.device)
        if key not in self._cache:
            self._cache[key] = tricubic_matrix(n, self.ratio).to(device=x.device, dtype=x.dtype)
        return self._cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.ratio
        if self.mode == "nearest":
            for d in (2, 3, 4):
                x = x.repeat_interleave(r, dim=d)
            return x
        if self.mode == "trilinear":
            return F.interpolate(x, scale_factor=r, mode="trilinear", align_corners=False)
        mx, my, mz = (self._matrix(n, x) for n in x.shape[2:])
        return torch.einsum("xi,yj,zk,bcijk->bcxyz", mx, my, mz, x)
