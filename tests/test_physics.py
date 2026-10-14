import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from blastnet import filters, octahedral, physics, upstream
from blastnet.data import norm_stats, synthetic

REF = Path(__file__).resolve().parents[2] / "reference/blastnet2_sr_benchmark"


@pytest.fixture(scope="module")
def ref():
    if not (REF / "common/functions.py").is_file():
        pytest.skip("reference checkout absent")
    pytest.importorskip("scipy")
    sys.path.insert(0, str(REF))
    import common.data as data
    import common.functions as fn
    import models.metrics as mt

    return SimpleNamespace(fn=fn, mt=mt, data=data)


def _rand(*shape, seed=0, lo=-1.0, hi=1.0):
    g = torch.Generator().manual_seed(seed)
    return lo + (hi - lo) * torch.rand(*shape, generator=g, dtype=torch.float32)


def _grid(n, dx, dtype=torch.float64):
    axes = [torch.arange(m, dtype=dtype) * h for m, h in zip(n, dx)]
    return torch.meshgrid(*axes, indexing="ij")


# ---------------------------------------------------------------- upstream ports, bit for bit


def test_upstream_differences_and_filters_equal_reference(ref):
    phi = _rand(2, 1, 12, 10, 8, seed=1)
    h = _rand(2, seed=2, lo=0.5, hi=2.0)
    for ours, theirs in ((upstream.torch_dx, ref.fn.torch_dx), (upstream.torch_dy, ref.fn.torch_dy),
                         (upstream.torch_dz, ref.fn.torch_dz)):
        assert torch.equal(ours(phi, h), theirs(phi, h))
    assert torch.equal(upstream.remove_edges(phi), ref.fn.remove_edges(phi))
    x = _rand(2, 1, 16, 16, 16, seed=3)
    rho = _rand(2, 1, 16, 16, 16, seed=4, lo=0.1, hi=0.4)
    for fw in (2, 4, 8):
        assert torch.equal(upstream.torch_filter(x, fw), ref.fn.torch_filter(x, fw))
        assert torch.equal(upstream.favre_filter(rho, x, fw), ref.fn.favre_filter(rho, x, fw))
    ruvw = torch.cat([rho, 30 + 50 * _rand(2, 3, 16, 16, 16, seed=5)], 1)
    for fw in (2, 4):
        assert torch.equal(upstream.sgs(ruvw, fw), ref.fn.sgs(ruvw, fw))
    stress = _rand(2, 6, 8, 7, 6, seed=6)
    hs = [_rand(2, seed=s, lo=0.5, hi=2.0) for s in (7, 8, 9)]
    for a, b in zip(upstream.divergence_sgs_separate(stress, *hs),
                    ref.fn.divergence_sgs_separate(stress, *hs)):
        assert torch.equal(a, b)


def test_upstream_ssim_and_gradloss_equal_reference(ref):
    assert torch.equal(upstream.create_window_3D(9, 6), ref.mt.create_window_3D(9, 6))
    a, b = _rand(2, 1, 12, 12, 12, seed=1), _rand(2, 1, 12, 12, 12, seed=2)
    c, d = _rand(2, 6, 6, 6, 6, seed=3), _rand(2, 6, 6, 6, 6, seed=4)
    ours, theirs = upstream.SSIM3D(), ref.mt.SSIM3D()
    for x, y in ((a, b), (c, d), (a, b)):  # exercises the window cache both ways
        assert torch.equal(ours(x, y), theirs(x, y))
    for avg in (True, False):
        assert torch.equal(upstream.ssim3D(c, d, size_average=avg),
                           ref.mt.ssim3D(c, d, size_average=avg))
    assert upstream.ssim3D(c, d, size_average=False).shape == (2, 6)  # upstream's [B, Z] quirk
    per = upstream.ssim3d(c, d)
    assert per.shape == (2,)
    torch.testing.assert_close(per.mean(), upstream.ssim3D(c, d))
    p, t = _rand(2, 4, 10, 9, 8, seed=5), _rand(2, 4, 10, 9, 8, seed=6)
    assert torch.equal(upstream.GradLoss()(p, t), ref.mt.GradLoss()(p, t))


@pytest.mark.parametrize("ratio", [4, 8])
def test_test_metrics_equal_litmodel_test_step(ref, ratio):
    pytest.importorskip("pytorch_lightning")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from models.litmodel import LitModel

    n, B = 32, 2
    mean, std = norm_stats("test")
    y = _rand(B, 4, n, n, n, seed=10)
    y_hat = y + 0.3 * _rand(B, 4, n, n, n, seed=11)
    dx_m = torch.tensor([[4.0707236621528864e-05] * 3,
                         [1.8623321011546068e-05, 1.5846149835851975e-05, 1.5846149835851975e-05]],
                        dtype=torch.float32)

    class Fixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.upscale = ratio

        def forward(self, x):
            return y_hat

    lm = LitModel(Fixed(), mean, std)
    logged = {}
    lm.log = lambda name, value, *a, **k: logged.__setitem__(name, float(value))
    # upstream dataset: torch.tensor(np.float32(dx)) / dx_min per sample, then default_collate
    items = [(y[i, :, ::ratio, ::ratio, ::ratio], y[i],
              *(torch.tensor(np.float32(dx_m[i, j].item())) / upstream.DX_MIN for j in range(3)))
             for i in range(B)]
    lm.test_step(torch.utils.data.default_collate(items), 0)

    ours = upstream.test_metrics(y_hat, y, mean, std, dx_m, ratio)
    assert tuple(ours) == upstream.KEYS
    assert set(logged) == set(upstream.KEYS)
    assert ours == logged  # exact float equality, all 16 keys
    assert all(math.isfinite(v) for v in ours.values())


# ---------------------------------------------------------------- filters


def test_box_and_favre_equal_upstream_numpy_filters(ref):
    x = _rand(1, 1, 32, 32, 32, seed=1, lo=0.0, hi=60.0)
    rho = _rand(1, 1, 32, 32, 32, seed=2, lo=0.1, hi=0.4)
    for r in (2, 4, 8):
        expect = torch.from_numpy(ref.data.my_box_filter(x[0, 0].numpy(), r))
        assert torch.equal(filters.box(x, r)[0, 0], expect)
        fav = ref.data.my_favre_filter(x[0, 0].numpy(), rho[0, 0].numpy(), r)
        assert torch.equal(filters.favre(rho, x, r)[0, 0], torch.from_numpy(fav))


def test_make_lr_kinds():
    hr = synthetic.cubes(2, size=16, ratio=8)["hr"]
    lr = filters.make_lr(hr, 8)
    assert lr.shape == (2, 4, 2, 2, 2)
    assert torch.equal(lr[:, :1], filters.box(hr[:, :1], 8))
    assert torch.equal(lr[:, 1:], filters.favre(hr[:, :1], hr[:, 1:], 8))
    # L1's synthetic LR is the same Favre/box definition (one-shot float64 mean, so ~1 ulp off)
    torch.testing.assert_close(lr, synthetic.cubes(2, size=16, ratio=8)["lr"], rtol=1e-6, atol=0)
    assert torch.equal(filters.make_lr(hr, 4, kind="box"), filters.box(hr, 4))
    assert torch.equal(filters.make_lr(hr, 4, kind="subsample"), hr[..., 2::4, 2::4, 2::4])
    with pytest.raises(ValueError):
        filters.make_lr(hr, 4, kind="nope")
    with pytest.raises(ValueError):
        filters.box(hr, 5)


def test_gaussian_and_spectral_filters():
    x = torch.full((1, 2, 12, 12, 12), 3.5, dtype=torch.float64)
    torch.testing.assert_close(filters.gaussian(x, 4), torch.full((1, 2, 3, 3, 3), 3.5,
                                                                  dtype=torch.float64))
    y = torch.randn(1, 1, 12, 12, 12, dtype=torch.float64)
    torch.testing.assert_close(filters.gaussian(y, 3, sigma=1e-3), y[..., 1::3, 1::3, 1::3])
    # a band-limited periodic field is reproduced exactly at the block centres
    n, r = 16, 4
    gx, gy, gz = _grid((n, n, n), (1.0, 1.0, 1.0))

    def f(a, b, c):
        return torch.cos(2 * math.pi * a / n) + 0.5 * torch.sin(2 * math.pi * (b + c) / n)

    c = torch.arange(n // r, dtype=torch.float64) * r + (r - 1) / 2
    cx, cy, cz = torch.meshgrid(c, c, c, indexing="ij")
    torch.testing.assert_close(filters.spectral(f(gx, gy, gz)[None, None], r)[0, 0], f(cx, cy, cz))


# ---------------------------------------------------------------- physics: analytic cases


def _taylor_green(n):
    x = torch.arange(n, dtype=torch.float64) * 2 * math.pi / n
    X, Y, Z = torch.meshgrid(x, x, x, indexing="ij")
    sx, cx, sy, cy, sz, cz = X.sin(), X.cos(), Y.sin(), Y.cos(), Z.sin(), Z.cos()
    u = torch.stack([sx * cy * cz, -cx * sy * cz, torch.zeros_like(X)])[None]
    A = torch.zeros(1, 3, 3, n, n, n, dtype=torch.float64)
    A[0, 0, 0], A[0, 0, 1], A[0, 0, 2] = cx * cy * cz, -sx * sy * cz, -sx * cy * sz
    A[0, 1, 0], A[0, 1, 1], A[0, 1, 2] = sx * sy * cz, -cx * cy * cz, cx * sy * sz
    q = cz**2 * (sx**2 * sy**2 - cx**2 * cy**2)
    ens = 0.5 * (cx**2 * sy**2 * sz**2 + sx**2 * cy**2 * sz**2 + 4 * sx**2 * sy**2 * cz**2)
    ss = 2 * cx**2 * cy**2 * cz**2 + 0.5 * sz**2 * (sx**2 * cy**2 + cx**2 * sy**2)
    return u, A, q[None], ens[None], ss[None], 2 * math.pi / n


def test_taylor_green_periodic_closed_forms():
    u, A, q, ens, ss, h = _taylor_green(32)
    kap = math.sin(h) / h  # central differences of sin/cos at unit wavenumber: modified wavenumber
    tol = dict(rtol=0, atol=1e-12)
    torch.testing.assert_close(physics.velocity_gradient(u, h, bc="periodic"), kap * A, **tol)
    torch.testing.assert_close(physics.q_criterion(u, h, bc="periodic"), kap**2 * q, **tol)
    torch.testing.assert_close(physics.enstrophy(u, h, bc="periodic"), kap**2 * ens, **tol)
    nu = 1.5e-5
    eps = physics.dissipation(u, h, nu=nu, bc="periodic")
    torch.testing.assert_close(eps, 2 * nu * kap**2 * ss, rtol=1e-12, atol=1e-18)
    PQR = physics.strain_invariants(u, h, bc="periodic")
    torch.testing.assert_close(PQR[:, 0], torch.zeros_like(q), **tol)  # divergence-free
    torch.testing.assert_close(PQR[:, 1], kap**2 * q, **tol)  # invariant Q = Hunt Q when tr A = 0
    # and close to the continuum (kap^2 = 0.987 at 32 points per period)
    torch.testing.assert_close(physics.q_criterion(u, h, bc="periodic"), q, rtol=0, atol=0.02)


def _quadratic(n, dx):
    X, Y, Z = _grid(n, dx)
    u = torch.stack([1 + 2 * X - Y + 0.5 * X * X + X * Y - 0.3 * Z * Z,
                     -X + 0.7 * Y * Z + 0.2 * Y * Y - 2 * Z,
                     0.4 * X * Z - Y * Y + 3 * Z + 0.1 * X * X])
    A = torch.stack([torch.stack([2 + X + Y, -1 + X, -0.6 * Z]),
                     torch.stack([-torch.ones_like(X), 0.7 * Z + 0.4 * Y, 0.7 * Y - 2]),
                     torch.stack([0.4 * Z + 0.2 * X, -2 * Y, 0.4 * X + 3 + 0 * Z])])
    return u, A


def test_quadratic_nonperiodic_exact_including_edges():
    n = (9, 10, 11)
    dxs = [(0.3, 0.5, 0.7), (0.11, 0.2, 0.05)]
    us, As = zip(*(_quadratic(n, d) for d in dxs))
    u, A = torch.stack(us), torch.stack(As)
    got = physics.velocity_gradient(u, torch.tensor(dxs, dtype=torch.float64))  # [B, 3] spacing
    torch.testing.assert_close(got, A, rtol=0, atol=1e-10)
    single = physics.velocity_gradient(u[:1], dxs[0])  # 3-tuple spacing
    torch.testing.assert_close(single, A[:1], rtol=0, atol=1e-10)
    # same scheme as torch.gradient(edge_order=2) on an arbitrary field
    f = torch.randn(1, 3, 7, 6, 5, dtype=torch.float64)
    ref = torch.stack(torch.gradient(f[0], spacing=dxs[1], dim=(1, 2, 3), edge_order=2), 1)[None]
    torch.testing.assert_close(physics.velocity_gradient(f, dxs[1]), ref, rtol=1e-12, atol=1e-12)


def test_linear_in_x_with_anisotropic_spacing():
    dx = (0.5, 0.25, 2.0)
    X, _, _ = _grid((8, 8, 8), dx, dtype=torch.float32)
    u = torch.stack([3.0 * X + 1.0, -2.0 * X, torch.zeros_like(X)])[None]
    A = physics.velocity_gradient(u, dx)
    expect = torch.zeros_like(A)
    expect[:, 0, 0], expect[:, 1, 0] = 3.0, -2.0
    torch.testing.assert_close(A, expect, rtol=0, atol=1e-5)
    with pytest.raises(ValueError):
        physics.velocity_gradient(u, (1.0, 2.0))


def test_invariants_match_eigenvalues_sign_convention():
    # u = M x: A = M everywhere, and lambda^3 + P lambda^2 + Q lambda + R = 0 for eig(M)
    M = torch.tensor([[0.3, -1.2, 0.5], [0.8, -0.1, 0.4], [-0.7, 0.2, 0.6]], dtype=torch.float64)
    dx = (0.4, 0.9, 0.6)
    grid = torch.stack(_grid((5, 6, 7), dx))
    u = torch.einsum("ij,j...->i...", M, grid)[None]
    P, Q, R = physics.strain_invariants(u, dx).unbind(1)
    lam = torch.linalg.eigvals(M)
    p = -lam.sum().real
    q = (lam[0] * lam[1] + lam[0] * lam[2] + lam[1] * lam[2]).real
    r = -lam.prod().real
    for got, want in ((P, p), (Q, q), (R, r)):
        torch.testing.assert_close(got, torch.full_like(got, want.item()), rtol=0, atol=1e-10)
    assert abs(p.item() + M.trace().item()) < 1e-12  # P = -tr A
    # Hunt's Q differs from the invariant Q by P^2 / 2 when the field is not solenoidal
    torch.testing.assert_close(physics.q_criterion(u, dx), Q - 0.5 * P * P, rtol=0, atol=1e-10)


@pytest.mark.parametrize("bc", ["periodic", "nonperiodic"])
def test_q_criterion_octahedral_equivariance(bc):
    u = torch.randn(2, 3, 10, 10, 10, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    q = physics.q_criterion(u, 0.7, bc=bc)
    for g in octahedral.elements():
        lhs = physics.q_criterion(octahedral.apply(u, g, vector_channels=(0, 1, 2)), 0.7, bc=bc)
        rhs = octahedral.apply(q[:, None], g, vector_channels=None)[:, 0]
        torch.testing.assert_close(lhs, rhs, rtol=0, atol=1e-12)


# ---------------------------------------------------------------- spectra


def _mode(n, m, axis=0, comp=0):
    x = torch.arange(n, dtype=torch.float64)
    shape = [1, 1, 1]
    shape[axis] = n
    u = torch.zeros(1, 3, n, n, n, dtype=torch.float64)
    u[0, comp] = torch.cos(2 * math.pi * m * x / n).reshape(shape)
    return u


def test_energy_spectrum_single_mode():
    n = 32
    k, E = physics.energy_spectrum(_mode(n, 5), window=None)
    assert k.shape == E.shape == (1, n // 2)
    assert int(E[0].argmax()) == 4  # shell 5 (shells start at 1)
    dk = (k[0, 1] - k[0, 0]).item()
    torch.testing.assert_close((E.sum() * dk).item(), 0.25)  # <u^2>/2 with <cos^2> = 1/2
    torch.testing.assert_close(k[0, 4].item(), 2 * math.pi * 5 / n)
    # non-periodic crop of a non-integer mode, Hann-windowed: still peaks in shell 5
    _, Eh = physics.energy_spectrum(_mode(n, 5.3))
    assert int(Eh[0].argmax()) == 4
    # physical wavenumbers with dx != dy: a mode along y (dy = 2) sits at half the shell index
    dx = (1.0, 2.0, 1.0)
    ky, Ey = physics.energy_spectrum(_mode(n, 4, axis=1), dx=dx, window=None)
    _, Ex = physics.energy_spectrum(_mode(n, 4, axis=0), dx=dx, window=None)
    assert int(Ey[0].argmax()) == 3 and int(Ex[0].argmax()) == 7
    torch.testing.assert_close(ky[0, 3].item(), 2 * math.pi * 4 / (n * 2.0))


def test_spectrum_error():
    u = torch.randn(2, 3, 16, 16, 16, dtype=torch.float64)
    torch.testing.assert_close(physics.spectrum_error(u, u), torch.zeros(2, dtype=torch.float64))
    err = physics.spectrum_error(2 * u, u)
    assert err.shape == (2,)
    torch.testing.assert_close(err, torch.full((2,), math.log(4.0), dtype=torch.float64))


# ---------------------------------------------------------------- SGS, metrics, autograd


def test_sgs_stress_uniform_velocity_is_zero_and_matches_upstream():
    rho = _rand(2, 1, 16, 16, 16, seed=1, lo=0.1, hi=0.4)
    vel = torch.tensor([30.0, -12.0, 5.0]).reshape(1, 3, 1, 1, 1).expand(2, 3, 16, 16, 16)
    ruvw = torch.cat([rho, vel], 1)
    tau = physics.sgs_stress(ruvw, 4)
    assert tau.shape == (2, 6, 4, 4, 4)
    torch.testing.assert_close(tau, torch.zeros_like(tau), rtol=0, atol=1e-5 * 0.4 * 30.0**2)
    noisy = torch.cat([rho, 30 * _rand(2, 3, 16, 16, 16, seed=2)], 1)
    assert torch.equal(physics.sgs_stress(noisy, 4), upstream.sgs(noisy, 4))


def test_metrics_keys_and_identity():
    cubes = synthetic.cubes(2, size=16, ratio=4)
    mean, std = norm_stats("train")
    y = (cubes["hr"] - mean[None, :, None, None, None]) / std[None, :, None, None, None]
    dx = cubes["meta"][["dx", "dy", "dz"]].to_numpy()
    out = physics.metrics(y, y, 4, torch.tensor(dx, dtype=torch.float32), mean, std)
    assert tuple(out) == (*upstream.KEYS, "spectrum_error", "dissipation_error")
    for k, v in out.items():
        assert v == (1.0 if "ssim" in k else 0.0), k
    noisy = y + 0.1 * torch.randn_like(y)
    out2 = physics.metrics(noisy, y, 4, torch.tensor(dx, dtype=torch.float32), mean, std)
    ref = upstream.test_metrics(noisy, y, mean, std, torch.tensor(dx, dtype=torch.float32), 4)
    assert {k: out2[k] for k in upstream.KEYS} == ref
    assert out2["spectrum_error"] > 0 and out2["dissipation_error"] > 0


def test_physics_is_differentiable():
    u = torch.randn(1, 3, 4, 4, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda v: physics.q_criterion(v, (0.5, 1.0, 2.0)), (u,))
    assert torch.autograd.gradcheck(lambda v: physics.dissipation(v, 1.0, bc="periodic"), (u,))
    v = torch.randn(1, 3, 8, 8, 8, dtype=torch.float64, requires_grad=True)
    physics.energy_spectrum(v)[1].sum().backward()
    assert torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0
