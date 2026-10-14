"""Filters and LR generation on [B, C, X, Y, Z] grids (torch only, differentiable).

    from blastnet import filters
    lr = filters.make_lr(hr, 8)                  # [B, 4, n, n, n] (rho, u, v, w) -> [B, 4, n/8, ...]
    rb = filters.box(rho, 8); uf = filters.favre(rho, u, 8)

`box` reproduces upstream `common/data.py::my_box_filter` (scipy `uniform_filter(size=r)` sampled
at r//2::r) bit for bit on float32: scipy filters one axis at a time (x, then y, then z),
accumulating in double and storing float32 between axes; `box` does the same (float64 block mean
per axis, rounded to the input dtype after each axis). The sampled outputs never touch scipy's
reflected boundary, so `box` is the plain r^3 block mean. `favre` = box(rho * phi) / box(rho) in
the input dtype, as upstream `my_favre_filter`.

Note: `upstream.torch_filter` (used by `upstream.sgs` and the test metrics) is a different
implementation of the same block mean (float32 conv3d) and differs from `box` by rounding.
"""

from __future__ import annotations

import math

import torch

__all__ = ["KINDS", "box", "favre", "gaussian", "make_lr", "spectral"]

KINDS = ("favre_box", "box", "subsample")
"""`make_lr` kinds. The default is provisional until the main loop checks the shipped LR files."""

_DIMS = (-3, -2, -1)


def _check(x: torch.Tensor, r: int) -> None:
    if x.dim() < 3:
        raise ValueError(f"expected [..., X, Y, Z], got shape {tuple(x.shape)}")
    if r < 1 or any(s % r for s in x.shape[-3:]):
        raise ValueError(f"spatial shape {tuple(x.shape[-3:])} is not a multiple of ratio {r}")


def box(x: torch.Tensor, r: int) -> torch.Tensor:
    """Block (top-hat) average over r^3 cells: [..., X, Y, Z] -> [..., X/r, Y/r, Z/r]."""
    _check(x, r)
    acc = torch.float64 if x.dtype in (torch.float16, torch.bfloat16, torch.float32) else x.dtype
    y = x
    for d in _DIMS:
        d = d % y.dim()
        y = y.to(acc).unflatten(d, (y.shape[d] // r, r)).mean(d + 1).to(x.dtype)
    return y


def favre(rho: torch.Tensor, phi: torch.Tensor, r: int) -> torch.Tensor:
    """Density-weighted (Favre) filter box(rho * phi) / box(rho); rho broadcasts over channels."""
    return box(rho * phi, r) / box(rho, r)


def _gauss_matrix(n: int, r: int, sigma: float, dtype, device) -> torch.Tensor:
    # [n/r, n] weights centred on block centres k*r + (r-1)/2, truncated at the edges and
    # renormalised (no periodic wrap, no reflection), truncated at 4 sigma
    centres = torch.arange(n // r, dtype=torch.float64, device=device) * r + (r - 1) / 2
    pos = torch.arange(n, dtype=torch.float64, device=device)
    d = pos[None, :] - centres[:, None]
    w = torch.exp(-0.5 * (d / sigma) ** 2) * (d.abs() <= 4 * sigma + 0.5 * r)
    return (w / w.sum(1, keepdim=True)).to(dtype)


def gaussian(x: torch.Tensor, r: int, sigma: float | None = None) -> torch.Tensor:
    """Gaussian filter sampled at block centres: [..., X, Y, Z] -> [..., X/r, Y/r, Z/r].

    sigma is in grid cells; the default r / sqrt(12) matches the second moment of the r-cell box.
    Non-periodic: kernels are truncated at the domain edge and renormalised to unit sum.
    """
    _check(x, r)
    sigma = r / math.sqrt(12.0) if sigma is None else float(sigma)
    y = x
    for d in _DIMS:
        d = d % y.dim()
        w = _gauss_matrix(y.shape[d], r, sigma, y.dtype, y.device)
        y = torch.movedim(torch.movedim(y, d, -1) @ w.T, -1, d)
    return y


def spectral(x: torch.Tensor, r: int) -> torch.Tensor:
    """Sharp spectral cutoff to the coarse grid, sampled at block centres (assumes periodicity).

    Keeps Fourier modes |k_i| < n_i / (2 r) per axis (the coarse Nyquist mode is dropped so the
    result is real) and evaluates the band-limited field at positions k*r + (r-1)/2.
    Non-periodic inputs ring at the edges (Gibbs); use `box` or `gaussian` for crops.
    """
    _check(x, r)
    X = torch.fft.fftn(x, dim=_DIMS)
    for d in _DIMS:
        d = d % X.dim()
        n = X.shape[d]
        m = n // r
        f = torch.fft.fftfreq(n, d=1.0 / n, dtype=torch.float64, device=x.device)
        # integer frequencies |f| < m/2 (for even m this excludes the coarse Nyquist f = m/2)
        pos, neg = (m + 1) // 2, (m - 1) // 2
        src = torch.cat([torch.arange(0, pos), torch.arange(n - neg, n)]).to(x.device)
        dst = torch.cat([torch.arange(0, pos), torch.arange(m - neg, m)]).to(x.device)
        # phase shift so the coarse inverse FFT samples at block centres j*r + (r-1)/2
        shift = torch.exp(2j * math.pi * f * ((r - 1) / 2) / n).to(X.dtype)
        shape = [1] * X.dim()
        shape[d] = n
        X = (X * shift.reshape(shape)).index_select(d, src)
        coarse = torch.zeros(*X.shape[:d], m, *X.shape[d + 1:], dtype=X.dtype, device=x.device)
        X = coarse.index_copy(d, dst, X)
    y = torch.fft.ifftn(X, dim=_DIMS).real / (r**3)
    return y.to(x.dtype)


def make_lr(hr: torch.Tensor, r: int, kind: str = "favre_box") -> torch.Tensor:
    """LR cube from HR (rho, u, v, w): [B, 4, n, n, n] -> [B, 4, n/r, n/r, n/r].

    kind="favre_box": box(rho) and Favre-filtered u, v, w (upstream my_box_filter /
    my_favre_filter); "box": box on every channel; "subsample": hr[..., r//2::r, r//2::r, r//2::r]
    (the same sample points upstream's box path uses). Provisional default; see KINDS.
    """
    if hr.dim() < 4 or hr.shape[-4] != 4:
        raise ValueError(f"expected [..., 4, X, Y, Z] (rho, u, v, w), got {tuple(hr.shape)}")
    _check(hr, r)
    if kind == "favre_box":
        rho = hr[..., 0:1, :, :, :]
        return torch.cat([box(rho, r), favre(rho, hr[..., 1:4, :, :, :], r)], dim=-4)
    if kind == "box":
        return box(hr, r)
    if kind == "subsample":
        s = r // 2
        return hr[..., s::r, s::r, s::r].contiguous()
    raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
