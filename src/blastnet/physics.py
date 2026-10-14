"""Flow physics on [B, 3, X, Y, Z] velocity grids (torch only, differentiable, [B, ...] outputs).

    from blastnet import physics
    u = hr[:, 1:4]                                   # velocity channels of (rho, u, v, w)
    A = physics.velocity_gradient(u, dx)             # [B, 3, 3, X, Y, Z], A[:, i, j] = du_i/dx_j
    P, Q, R = physics.strain_invariants(u, dx).unbind(1)

Spacing `dx` (metres or any unit) is a float, 3 values (dx, dy, dz), or a [B, 3] tensor.
Boundary handling `bc`:
- "nonperiodic" (default; real cubes are crops of non-periodic DNS): second-order central
  differences in the interior and second-order one-sided differences on the boundary planes,
  (-3 f0 + 4 f1 - f2) / (2h), as torch.gradient(edge_order=2). Exact for quadratics everywhere.
  (Upstream's torch_dx uses edge_order=1; see blastnet.upstream.)
- "periodic": central differences with wrap-around.

Conventions: A_ij = du_i/dx_j; S = (A + A^T)/2, W = (A - A^T)/2, |M|^2 = M_ij M_ij;
P = -tr A, Q = (P^2 - tr A^2)/2, R = -det A, so the eigenvalues of A solve
lambda^3 + P lambda^2 + Q lambda + R = 0 (Chong, Perry & Cantwell 1990).
q_criterion is Hunt's Q = (|W|^2 - |S|^2)/2 = -tr(A^2)/2, which equals the invariant Q only when
tr A = 0. enstrophy = |omega|^2 / 2 = |W|^2 with omega = curl u. dissipation = 2 nu S_ij S_ij
(nu kinematic; no dilatational correction).
"""

from __future__ import annotations

import math

import torch

from blastnet import upstream
from blastnet.upstream import ssim3d

__all__ = [
    "BCS",
    "dissipation",
    "energy_spectrum",
    "enstrophy",
    "metrics",
    "q_criterion",
    "sgs_stress",
    "spectrum_error",
    "ssim3d",
    "strain_invariants",
    "velocity_gradient",
]

BCS = ("nonperiodic", "periodic")


def _spacing(dx, batch: int, like: torch.Tensor) -> torch.Tensor:
    # -> [B, 3] in like's dtype/device
    d = torch.as_tensor(dx, dtype=like.dtype, device=like.device)
    if d.dim() == 0:
        d = d.expand(3)
    if d.dim() == 1 and d.shape[0] == 3:
        d = d.expand(batch, 3)
    if tuple(d.shape) != (batch, 3):
        raise ValueError(f"dx must be a float, 3 values, or [B, 3]; got shape {tuple(d.shape)}")
    return d


def _check_u(u: torch.Tensor) -> None:
    if u.dim() != 5 or u.shape[1] != 3:
        raise ValueError(f"velocity must be [B, 3, X, Y, Z], got {tuple(u.shape)}")


def _diff(f: torch.Tensor, h: torch.Tensor, dim: int, bc: str) -> torch.Tensor:
    # d f / d x along `dim` of a [B, ...] tensor; h is [B]
    h = h.reshape(-1, *([1] * (f.dim() - 1)))
    if bc == "periodic":
        return (f.roll(-1, dim) - f.roll(1, dim)) / (2 * h)
    if bc != "nonperiodic":
        raise ValueError(f"bc must be one of {BCS}, got {bc!r}")
    n = f.shape[dim]
    if n < 3:
        raise ValueError(f"non-periodic differences need >= 3 points along dim {dim}, got {n}")
    s = f.narrow
    first = -1.5 * s(dim, 0, 1) + 2 * s(dim, 1, 1) - 0.5 * s(dim, 2, 1)
    inner = 0.5 * (s(dim, 2, n - 2) - s(dim, 0, n - 2))
    last = 1.5 * s(dim, n - 1, 1) - 2 * s(dim, n - 2, 1) + 0.5 * s(dim, n - 3, 1)
    return torch.cat([first, inner, last], dim) / h


def velocity_gradient(u: torch.Tensor, dx=1.0, bc: str = "nonperiodic") -> torch.Tensor:
    """A[:, i, j] = du_i/dx_j: [B, 3, X, Y, Z] -> [B, 3, 3, X, Y, Z]."""
    _check_u(u)
    h = _spacing(dx, u.shape[0], u)
    return torch.stack([_diff(u, h[:, j], 2 + j, bc) for j in range(3)], dim=2)


def _invariants(A: torch.Tensor) -> torch.Tensor:
    tr = A[:, 0, 0] + A[:, 1, 1] + A[:, 2, 2]
    tr2 = torch.einsum("bij...,bji...->b...", A, A)
    (a00, a01, a02), (a10, a11, a12), (a20, a21, a22) = (A[:, i].unbind(1) for i in range(3))
    det =(a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20)
           + a02 * (a10 * a21 - a11 * a20))
    P = -tr
    return torch.stack([P, 0.5 * (P * P - tr2), -det], dim=1)


def strain_invariants(u: torch.Tensor, dx=1.0, bc: str = "nonperiodic") -> torch.Tensor:
    """(P, Q, R) = (-tr A, (P^2 - tr A^2)/2, -det A) of the velocity gradient: [B, 3, X, Y, Z]."""
    return _invariants(velocity_gradient(u, dx, bc))


def _sym_antisym_sq(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    At = A.transpose(1, 2)
    S = 0.5 * (A + At)
    W = 0.5 * (A - At)
    return (S * S).sum((1, 2)), (W * W).sum((1, 2))


def q_criterion(u: torch.Tensor, dx=1.0, bc: str = "nonperiodic") -> torch.Tensor:
    """Hunt's Q = (|W|^2 - |S|^2) / 2: [B, 3, X, Y, Z] -> [B, X, Y, Z]."""
    ss, ww = _sym_antisym_sq(velocity_gradient(u, dx, bc))
    return 0.5 * (ww - ss)


def enstrophy(u: torch.Tensor, dx=1.0, bc: str = "nonperiodic") -> torch.Tensor:
    """|curl u|^2 / 2 (= |W|^2): [B, 3, X, Y, Z] -> [B, X, Y, Z]."""
    A = velocity_gradient(u, dx, bc)
    wx = A[:, 2, 1] - A[:, 1, 2]
    wy = A[:, 0, 2] - A[:, 2, 0]
    wz = A[:, 1, 0] - A[:, 0, 1]
    return 0.5 * (wx * wx + wy * wy + wz * wz)


def dissipation(u: torch.Tensor, dx=1.0, nu=1.0, bc: str = "nonperiodic") -> torch.Tensor:
    """2 nu S_ij S_ij (nu kinematic; float, [B], or broadcastable): -> [B, X, Y, Z]."""
    ss, _ = _sym_antisym_sq(velocity_gradient(u, dx, bc))
    nu = torch.as_tensor(nu, dtype=ss.dtype, device=ss.device)
    if nu.dim() == 1:
        nu = nu.reshape(-1, 1, 1, 1)
    return 2 * nu * ss


def _window(n: int, kind: str | None, like: torch.Tensor) -> torch.Tensor:
    if kind is None or kind == "none":
        return torch.ones(n, dtype=like.dtype, device=like.device)
    if kind == "hann":
        return torch.hann_window(n, periodic=False, dtype=like.dtype, device=like.device)
    raise ValueError(f"window must be 'hann' or None, got {kind!r}")


def energy_spectrum(u: torch.Tensor, dx=1.0, window: str | None = "hann"
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Shell-binned kinetic-energy spectrum of [B, 3, X, Y, Z] -> (k [B, K], E [B, K]).

    Normalisation: u' = u - <u> per sample and component; u' is multiplied by the separable
    window w (symmetric Hann by default, for non-periodic crops; None for periodic fields) and
    the result divided by <w^2>, so that sum over ALL modes of E(k) dk = <|w u'|^2> / (2 <w^2>),
    which is <|u'|^2> / 2 (kinetic energy per unit mass) exactly when window is None.
    Wavevectors are physical, k_a = 2 pi m / (N_a dx_a); shell width dk = 2 pi / max_a(N_a dx_a);
    shell n holds |k| / dk in [n - 1/2, n + 1/2); E(n dk) = (energy in shell) / dk. Returned
    shells are n = 1..K, K = floor(k_nyq_min / dk) with k_nyq_min = pi / max_a dx_a (min over
    the batch, so that the batch stacks; K = N/2 for an N^3 cube at any spacing). Shell 0 (the
    removed mean, plus window leakage) and the corner modes beyond K are not returned.
    """
    _check_u(u)
    B = u.shape[0]
    n = u.shape[-3:]
    h = _spacing(dx, B, u).to(torch.float64)
    up = u - u.mean(dim=(-3, -2, -1), keepdim=True)
    w1 = [_window(m, window, u) for m in n]
    w = w1[0][:, None, None] * w1[1][None, :, None] * w1[2][None, None, :]
    w2 = (w * w).mean()
    U = torch.fft.fftn(up * w, dim=(-3, -2, -1))
    N3 = n[0] * n[1] * n[2]
    e = 0.5 * (U.real**2 + U.imag**2).sum(1) / (N3 * N3) / w2  # [B, X, Y, Z], sums to the energy
    L = h * torch.tensor(n, dtype=torch.float64, device=u.device)  # [B, 3]
    dk = 2 * math.pi / L.max(1).values  # [B]
    m = [torch.fft.fftfreq(s, d=1.0 / s, dtype=torch.float64, device=u.device) for s in n]
    kk = torch.zeros(B, *n, dtype=torch.float64, device=u.device)
    for a in range(3):
        shape = [1, 1, 1]
        shape[a] = n[a]
        ka = (2 * math.pi / L[:, a]).reshape(B, 1, 1, 1) * m[a].reshape(shape)
        kk = kk + ka * ka
    # floor(x + 0.5), not torch.round (half-to-even), so exact midpoints go up as documented
    shell = torch.floor(kk.sqrt() / dk.reshape(B, 1, 1, 1) + 0.5).long()  # [B, X, Y, Z]
    K = int(torch.floor((math.pi / h.max(1).values) / dk + 1e-6).min().item())
    nb = int(shell.max().item()) + 1
    flat = e.reshape(B, -1)
    idx = shell.reshape(B, -1) + nb * torch.arange(B, device=u.device)[:, None]
    E = torch.zeros(B * nb, dtype=e.dtype, device=u.device).index_add(0, idx.reshape(-1),
                                                                      flat.reshape(-1))
    E = E.reshape(B, nb)[:, 1:K + 1] / dk[:, None].to(e.dtype)
    k = dk[:, None] * torch.arange(1, K + 1, dtype=torch.float64, device=u.device)
    return k.to(e.dtype), E


def spectrum_error(u_hat: torch.Tensor, u: torch.Tensor, dx=1.0, window: str | None = "hann"
                   ) -> torch.Tensor:
    """Mean over shells of |ln E_hat(k) - ln E(k)| for [B, 3, X, Y, Z] velocities -> [B]."""
    # one sample at a time: energy_spectrum truncates a batch at its smallest cutoff, which would
    # make a sample's error depend on its batch companions when spacings differ
    h = _spacing(dx, u.shape[0], u)
    out = []
    for b in range(u.shape[0]):
        _, Eh = energy_spectrum(u_hat[b:b + 1], h[b:b + 1], window)
        _, E = energy_spectrum(u[b:b + 1], h[b:b + 1], window)
        tiny = torch.finfo(E.dtype).tiny
        out.append((Eh.clamp_min(tiny).log() - E.clamp_min(tiny).log()).abs().mean(1))
    return torch.cat(out)


def sgs_stress(ruvw: torch.Tensor, r: int) -> torch.Tensor:
    """Upstream SGS stress (rho_bar times Favre stress) of (rho, u, v, w): -> [B, 6, X/r, ...].

    Channels ordered 11, 22, 33, 12, 13, 23; delegates to blastnet.upstream.sgs (float32).
    """
    return upstream.sgs(ruvw, r)


@torch.no_grad()
def metrics(y_hat: torch.Tensor, y: torch.Tensor, ratio: int, dx, mean=None, std=None
            ) -> dict[str, float]:
    """Upstream test_* metrics plus spectrum_error and dissipation_error, as batch-mean floats.

    y_hat, y: [B, 4, n, n, n] (rho, u, v, w), normalised if mean/std ([4]) are given, else in
    physical units; dx: HR spacing in metres. spectrum_error: spectrum_error on de-normalised
    velocity (Hann window). dissipation_error: |<eps_hat> - <eps>| / <eps> per sample with the
    domain-mean eps = 2 nu S_ij S_ij (nu cancels; non-periodic differences, no edges removed).
    """
    if mean is None:
        mean = torch.zeros(4)
    if std is None:
        std = torch.ones(4)
    out = upstream.test_metrics(y_hat, y, mean, std, dx, ratio)
    m = torch.as_tensor(mean, dtype=y.dtype, device=y.device)[None, :, None, None, None]
    s = torch.as_tensor(std, dtype=y.dtype, device=y.device)[None, :, None, None, None]
    uh = (y_hat * s + m)[:, 1:4]
    ut = (y * s + m)[:, 1:4]
    h = _spacing(dx, y.shape[0], y)
    out["spectrum_error"] = float(spectrum_error(uh, ut, h).mean())
    eh = dissipation(uh, h).mean((1, 2, 3))
    et = dissipation(ut, h).mean((1, 2, 3))
    # a sample with zero true dissipation has no relative error: NaN, excluded from the mean
    rel = torch.where(et > 0, (eh - et).abs() / et.clamp_min(torch.finfo(et.dtype).tiny),
                      torch.full_like(et, float("nan")))
    out["dissipation_error"] = float(rel.nanmean())
    return out
