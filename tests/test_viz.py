import subprocess
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from blastnet import physics, viz
from blastnet.data import synthetic

THEMES = ("light", "dark")


@pytest.fixture(scope="module")
def cubes():
    return synthetic.cubes(3, size=16, ratio=8)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _check(fig, tmp_path, name):
    assert isinstance(fig, Figure)
    paths = viz.save(fig, tmp_path / name, dpi=60)
    assert [p.suffix for p in paths] == [".png", ".svg"] and all(p.stat().st_size > 0 for p in paths)


@pytest.mark.parametrize("theme", THEMES)
def test_all_figures(cubes, tmp_path, theme):
    hr, lr = cubes["hr"], cubes["lr"]
    pred = hr + 0.1 * torch.randn_like(hr)
    _check(viz.slices({"LR": lr[0], "pred": pred[0], "truth": hr[0]}, residual=("pred", "truth"),
                      theme=theme), tmp_path, "s")
    _check(viz.slices({"a": hr[0, 1], "b": pred[0, 1]}, channel=None, shared_scale=False,
                      theme=theme), tmp_path, "s2")
    fields = {"truth": hr, "pred": pred}
    _check(viz.spectra(fields, 1.0, reference="truth", theme=theme), tmp_path, "sp")
    _check(viz.spectra_ratio(fields, 1.0, "truth", theme=theme), tmp_path, "sr")
    _check(viz.qr(fields, 1.0, bins=32, theme=theme), tmp_path, "qr")
    t6 = synthetic.cubes(2, size=16, ratio=8, seed=1)["hr"]
    tau = physics.sgs_stress(t6, 4)
    _check(viz.parity(tau + 0.05 * torch.randn_like(tau), tau, theme=theme), tmp_path, "par")
    rows = [dict(family=f, size=s, ratio=8, seed=sd, split="test", params=int(p), m=v + sd * 0.01)
            for f, v in (("edsr", 1.0), ("rcan", 2.0)) for s, p in (("0.5M", 5e5), ("0.8M", 8e5))
            for sd in (42, 43, 44)]
    _check(viz.leaderboard(rows, "m", baselines={"tricubic": 1.5}, theme=theme), tmp_path, "lb")
    _check(viz.layer_curves({"a": [1, 2, 3], "b": [2, 3, 5]}, "err", logy=True, ref=1.0,
                            bands={"a": ([0.5, 1, 2], [1.5, 3, 4])}, theme=theme), tmp_path, "lc")


def test_slices_residual_symmetric(cubes):
    hr = cubes["hr"][0]
    fig = viz.slices({"p": hr + 0.3, "t": hr}, residual=("p", "t"))
    res = next(a for a in fig.axes if a.get_label() == "residual")
    lo, hi = res.images[0].get_clim()
    assert lo == -hi and hi > 0
    clims = {a.images[0].get_clim() for a in fig.axes if a.get_label().startswith("field:")}
    assert len(clims) == 1  # shared scale


def test_slices_upsamples_lr(cubes):
    fig = viz.slices({"lr": cubes["lr"][0], "hr": cubes["hr"][0]})
    shapes = {a.images[0].get_array().shape for a in fig.axes if a.images}
    assert shapes == {(16, 16)}


def test_spectra_uses_energy_spectrum(cubes):
    u = cubes["hr"][:, 1:4]
    fig = viz.spectra({"truth": u, "other": u * 0.5}, 0.01, reference="truth")
    ax = fig.axes[0]
    k, E = physics.energy_spectrum(u, 0.01)
    want = E.double().log().mean(0).exp().numpy()
    line = next(ln for ln in ax.lines if ln.get_label() == "truth")
    np.testing.assert_allclose(line.get_ydata(), want, rtol=1e-6)
    np.testing.assert_allclose(line.get_xdata(), k.mean(0).double().numpy() * 0.01, rtol=1e-6)
    assert ax.get_yscale() == "log"
    fig2 = viz.spectra({"t": cubes["hr"]}, 0.01)  # [B, 4, ...] uses channels 1:4
    np.testing.assert_allclose(fig2.axes[0].lines[0].get_ydata(), want, rtol=1e-6)
    ratio = viz.spectra_ratio({"truth": u, "other": u * 0.5}, 0.01, "truth")
    y = next(ln for ln in ratio.axes[0].lines if ln.get_label() == "other").get_ydata()
    np.testing.assert_allclose(y, 0.25, rtol=1e-5)


def test_vieillefosse():
    R = np.linspace(-5, 5, 101)
    assert np.abs(27 / 4 * R**2 + viz.vieillefosse(R) ** 3).max() < 1e-10
    fig = viz.qr({"a": synthetic.cubes(1, size=16)["hr"]}, 1.0, bins=16)
    line = next(ln for ln in fig.axes[0].lines if ln.get_label() == "Vieillefosse line")
    r, q = line.get_xdata(), line.get_ydata()
    assert np.abs(27 / 4 * r**2 + q**3).max() < 1e-10


def test_leaderboard_aggregates():
    rows = [dict(family="edsr", size="0.5M", ratio=8, seed=s, split="test", params=100, m=v)
            for s, v in ((1, 1.0), (2, 2.0), (3, 6.0))]
    rows += [dict(family="edsr", size="0.8M", ratio=8, seed=1, split="test", params=200, m=4.0),
             dict(family="edsr", size="0.5M", ratio=8, seed=1, split="val", params=100, m=99.0)]
    agg = viz.aggregate(rows, "m")["edsr"]
    assert [g["n"] for g in agg] == [3, 1]
    assert agg[0]["mean"] == pytest.approx(3.0)
    assert agg[0]["std"] == pytest.approx(np.std([1, 2, 6], ddof=1))
    assert agg[1]["std"] == 0.0
    ax = viz.leaderboard(rows, "m").axes[0]
    assert ax.get_xscale() == "log"
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [3.0, 4.0])


def test_colour_order_stable():
    pal = viz.PALETTE["light"]
    fig = viz.layer_curves({"a": [1, 2], "b": [2, 3], "c": [1, 1]}, "y")
    assert [ln.get_color() for ln in fig.axes[0].lines] == list(pal[:3])
    m = {"a": pal[0], "b": pal[1], "c": pal[2]}
    f2 = viz.layer_curves({"c": [1, 2]}, "y", colors=m)  # same key, same colour
    assert f2.axes[0].lines[0].get_color() == pal[2]
    rows = [dict(family=f, size="0.5M", ratio=8, seed=1, split="test", params=1, m=1.0)
            for f in ("rcan", "edsr")]
    got = {ln.get_label(): ln.get_color() for ln in viz.leaderboard(rows, "m").axes[0].lines}
    assert got == {"edsr": pal[0], "rcan": pal[1]}  # by family, not by row order
    with pytest.raises(ValueError):
        viz.layer_curves({str(i): [1, 2] for i in range(9)}, "y")


def test_dark_theme_surfaces():
    fig = viz.layer_curves({"a": [1, 2]}, "y", theme="dark")
    assert matplotlib.colors.to_hex(fig.get_facecolor()) == "#1a1a19"
    with viz.style(theme="dark"):
        assert matplotlib.colors.to_hex(matplotlib.rcParams["axes.facecolor"]) == "#1a1a19"
    assert "blastnet_div" in matplotlib.colormaps and "blastnet_seq" in matplotlib.colormaps


def test_import_is_lazy():
    code = ("import sys, blastnet, blastnet.viz; "
            "assert 'matplotlib' not in sys.modules, 'matplotlib imported'")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
