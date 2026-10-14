from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from blastnet import models
from blastnet.models.baselines import Interp


def _x(n=4, seed=0):
    return torch.randn(2, 3, n, n, n, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("mode", models.baselines.MODES)
@pytest.mark.parametrize("ratio", [8, 16, 32])
def test_shapes(mode, ratio):
    n = 128 // ratio // 2  # keep the output small
    y = Interp(ratio, mode)(_x(n))
    assert y.shape == (2, 3, n * ratio, n * ratio, n * ratio)


@pytest.mark.parametrize("mode", models.baselines.MODES)
def test_affine_invariance_per_channel(mode):
    x = _x()
    a = torch.tensor([2.0, -0.5, 30.0]).view(1, 3, 1, 1, 1)
    b = torch.tensor([1.0, 7.0, -100.0]).view(1, 3, 1, 1, 1)
    m = Interp(4, mode)
    torch.testing.assert_close(m(a * x + b), a * m(x) + b, rtol=1e-4, atol=1e-4)


def test_nearest_is_repeat_interleave():
    x = _x()
    ref = x
    for d in (2, 3, 4):
        ref = ref.repeat_interleave(4, dim=d)
    assert torch.equal(Interp(4, "nearest")(x), ref)


def test_trilinear_is_interpolate():
    x = _x()
    ref = F.interpolate(x, scale_factor=4, mode="trilinear", align_corners=False)
    assert torch.equal(Interp(4, "trilinear")(x), ref)


@pytest.mark.parametrize("n,ratio", [(4, 4), (5, 8), (3, 2)])
def test_tricubic_matches_upstream_zoom(n, ratio):
    zoom = pytest.importorskip("scipy.ndimage").zoom
    x = _x(n)
    y = Interp(ratio, "tricubic")(x)
    ref = np.stack([[zoom(c.numpy(), ratio, order=3, mode="nearest") for c in b] for b in x])
    np.testing.assert_allclose(y.numpy(), ref, rtol=1e-4, atol=1e-5)


def test_tricubic_constants_and_no_scipy_in_src():
    y = Interp(4, "tricubic")(torch.full((1, 1, 4, 4, 4), 3.0))
    torch.testing.assert_close(y, torch.full_like(y, 3.0))
    assert "import scipy" not in Path(models.baselines.__file__).read_text()
    assert "from scipy" not in Path(models.baselines.__file__).read_text()


@pytest.mark.parametrize("name", models.baselines.MODES)
def test_build_and_sites(name):
    m = models.build(name, None, ratio=8)
    assert isinstance(m, Interp) and m.mode == name and m.ratio == 8
    assert models.param_count(m) == 0
    assert models.sites(m) == dict(stem=None, blocks=[], head=None)
    assert models.build(name, None, ratio=16).ratio == 16
