import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from blastnet import octahedral as O
from blastnet.models.gconv import (
    EDSR_PARAMS,
    SIZES,
    GConvSR,
    build_gconv,
    feature_action,
    gconv_sites,
)

G = O.elements()
REF = Path(__file__).resolve().parents[2] / "reference" / "blastnet2_sr_benchmark" / "models"


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def _run_with_features(model, x):
    """Forward x, returning (output, {site: feature}) for the stem and every residual block."""
    sites = gconv_sites(model)
    names = [sites["stem"], *sites["blocks"]]
    feats, handles = {}, []
    for name in names:
        mod = model.get_submodule(name)
        handles.append(mod.register_forward_hook(lambda m, i, o, n=name: feats.__setitem__(n, o)))
    try:
        y = model(x)
    finally:
        for h in handles:
            h.remove()
    return y, feats


def _exhaustive(model, x, tol):
    """Max relative error over all 48 g at the output and at every block (feature_action)."""
    with torch.no_grad():
        y0, f0 = _run_with_features(model, x)
        gx = torch.cat([O.apply(x, g) for g in G])  # all 48 transformed inputs, in 4 batches
        parts = [_run_with_features(model, chunk) for chunk in gx.split(12)]
        yg = torch.cat([p[0] for p in parts])
        fg = {n: torch.cat([p[1][n] for p in parts]) for n in f0}
    errs = {"out": max(_rel(yg[g.index : g.index + 1], O.apply(y0, g)) for g in G)}
    for name, h in f0.items():
        errs[name] = max(_rel(fg[name][g.index : g.index + 1], feature_action(h, g)) for g in G)
    bad = {k: v for k, v in errs.items() if v > tol}
    assert not bad, bad
    return f0, fg


def test_exact_equivariance_float64_all_elements_and_blocks():
    torch.manual_seed(0)
    model = build_gconv("0.5M", ratio=2).double().eval()
    x = torch.randn(1, 4, 4, 4, 4, dtype=torch.float64)
    f0, fg = _exhaustive(model, x, tol=1e-10)  # contract: 1e-5 relative; observed ~1e-15
    assert len(f0) == 1 + 32
    # control: a grid move alone (hidden channels treated as scalars) is not the right action
    h, g = f0["body.5"], G[13]
    grid_only = O.apply(h, g, vector_channels=None)
    assert _rel(fg["body.5"][13:14], grid_only) > 1e-2


def test_exact_equivariance_float32():
    torch.manual_seed(1)
    model = GConvSR(multiplicity=2, n_blocks=3, ratio=4).eval()
    _exhaustive(model, torch.randn(1, 4, 3, 3, 3), tol=1e-5)


def test_feature_action_is_a_group_action():
    h = torch.randn(1, 2 * 48, 3, 3, 3, dtype=torch.float64)
    for a in G[::5]:
        for b in G[::7]:
            assert torch.equal(feature_action(feature_action(h, b), a),
                               feature_action(h, O.compose(a, b)))
    assert torch.equal(feature_action(h, O.identity()), h)
    with pytest.raises(ValueError):
        feature_action(torch.zeros(1, 5, 2, 2, 2), G[1])


def test_shapes_sites_and_names():
    torch.manual_seed(0)
    model = build_gconv("0.5M", ratio=8).eval()
    x = torch.randn(1, 4, 2, 2, 2)
    with torch.no_grad():
        y, feats = _run_with_features(model, x)
    assert y.shape == (1, 4, 16, 16, 16)
    c, n_blocks = SIZES["0.5M"]
    sites = gconv_sites(model)
    assert sites == dict(stem="head", blocks=[f"body.{i}" for i in range(n_blocks)], head="tail")
    assert all(f.shape == (1, c * 48, 2, 2, 2) for f in feats.values())
    names = {n for n, _ in model.named_modules()}
    assert {"head", "body.0", f"body.{n_blocks}", "tail", "tail.0", "tail.1"} <= names
    with pytest.raises(ValueError):
        build_gconv("50M")
    with pytest.raises(ValueError):
        GConvSR(1, 1, ratio=6)


@pytest.mark.parametrize("size", sorted(SIZES))
@pytest.mark.parametrize("ratio", [8, 16, 32])
def test_param_count_matches_edsr(size, ratio):
    n = sum(p.numel() for p in build_gconv(size, ratio).parameters())
    assert abs(n / EDSR_PARAMS[(size, ratio)] - 1) < 0.30, n


@pytest.mark.skipif(not REF.exists(), reason="reference checkout absent")
def test_recorded_edsr_targets_match_reference():
    spec = importlib.util.spec_from_file_location(
        "_ref_sr_models", REF / "__init__.py", submodule_search_locations=[str(REF)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pkg
    dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # the reference checkout is read-only
    try:
        spec.loader.exec_module(pkg)
        edsr = importlib.import_module("_ref_sr_models.edsr")
        for (size, ratio), n in EDSR_PARAMS.items():
            assert sum(p.numel() for p in edsr.init_edsr(size, ratio).parameters()) == n
    finally:
        sys.dont_write_bytecode = dont_write
        for k in [k for k in sys.modules if k.startswith("_ref_sr_models")]:
            del sys.modules[k]


def test_gradients_flow_and_stay_tied():
    torch.manual_seed(0)
    model = GConvSR(multiplicity=1, n_blocks=2, ratio=2).double()
    x = torch.randn(1, 4, 3, 3, 3, dtype=torch.float64)
    model(x).square().mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    with torch.no_grad():
        for p in model.parameters():
            p -= 0.1 * p.grad  # one SGD step keeps the net exactly equivariant
    _exhaustive(model.eval(), x, tol=1e-10)


def test_cpu_slab_conv_matches_single_call(monkeypatch):
    import blastnet.models.gconv as gconv

    torch.manual_seed(0)
    model = GConvSR(multiplicity=1, n_blocks=1, ratio=2).double().eval()
    x = torch.randn(1, 4, 5, 5, 5, dtype=torch.float64)
    with torch.no_grad():
        ref = model(x)
        monkeypatch.setattr(gconv, "CPU_COL_BYTES", 1)  # force one x-plane per conv call
        assert torch.allclose(model(x), ref, rtol=0, atol=1e-13)
    _exhaustive(model, x, tol=1e-10)


def test_resize_upsampler_is_exactly_equivariant_float64():
    """gconvrc: trilinear (align_corners=False) x2 then a group conv, per stage; still exact."""
    torch.manual_seed(2)
    model = GConvSR(multiplicity=2, n_blocks=2, ratio=4, upsampler="resize").double().eval()
    _exhaustive(model, torch.randn(1, 4, 3, 3, 3, dtype=torch.float64), tol=1e-10)
    names = [type(m).__name__ for m in model.tail[0]]
    assert names == ["Upsample", "GroupConv3d"] * 2


def test_resize_family_builds_and_has_fewer_upsampler_weights():
    from blastnet.models import build

    sub, rc = build("gconv", "0.5M", 8), build("gconvrc", "0.5M", 8)
    n_sub = sum(p.numel() for p in sub.parameters())
    n_rc = sum(p.numel() for p in rc.parameters())
    up_sub = sum(p.numel() for p in sub.tail[0].parameters())
    up_rc = sum(p.numel() for p in rc.tail[0].parameters())
    assert n_sub == 409_997
    assert up_sub - up_rc == n_sub - n_rc and up_rc * 8 - up_sub == 7 * (2 + 1 + 1)  # 8x fewer weights; biases (2, 1, 1) unchanged
    x = torch.randn(1, 4, 2, 2, 2)
    with torch.no_grad():
        assert rc(x).shape == (1, 4, 16, 16, 16)
    with pytest.raises(ValueError):
        GConvSR(1, 1, ratio=2, upsampler="nearest")
