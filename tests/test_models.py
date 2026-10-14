import ast
import importlib
import importlib.util
import re
import sys
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

from blastnet import models
from blastnet.models import convfno, edsr, rcan, rrdb

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "reference" / "blastnet2_sr_benchmark" / "models"
REF_WEIGHTS = REF.parent / "weights" / "seed42"
SUBSET_WEIGHTS = ROOT / "blastnet-py" / "data" / "momentum128_subset" / "best_model_weights"

# Hyperparameters of the shipped files, from best_model_weights/model_summary.csv (all seeds).
SUMMARY = {
    "edsr": {s: {"channel_size": c} for s, c in zip(
        models.LABELS, (14, 20, 24, 34, 46, 68, 86, 120))},
    "rcan": {s: {"channel_size": c, "n_resblocks": b, "n_resgroups": g} for s, (c, b, g) in zip(
        models.LABELS, ((26, 1, 1), (34, 1, 1), (44, 1, 1), (60, 1, 1), (64, 10, 1), (64, 20, 2),
                        (64, 20, 3), (64, 20, 10)))},
    "rrdb": {s: {"first_channel_size": c, "n_rrdb_blocks": b} for s, (c, b) in zip(
        models.LABELS, ((4, 1), (16, 1), (32, 1), (64, 1), (64, 2), (64, 5), (64, 8), (64, 23)))},
}
SUMMARY["rrdbgrad"] = SUMMARY["rrdb"]
SHIPPED = {(n, s, 8) for n in SUMMARY for s in models.LABELS}
SHIPPED |= {(n, s, r) for n, s in [("edsr", "50M"), ("rcan", "17M"), ("rrdb", "50M"),
                                   ("rrdbgrad", "50M")] for r in (16, 32)}
REF_INIT = {"edsr": ("edsr", "init_edsr"), "rcan": ("rcan", "init_rcan"),
            "rrdb": ("rrdb", "init_rrdb"), "rrdbgrad": ("rrdb", "init_rrdb"),
            "convfno": ("convfno", "init_convfno")}
# LR edge and batch per ratio: CPU conv3d builds a [B, Cin * 27, X * Y * Z] im2col buffer, so the
# largest (edsr 50M, 120 channels at 32^3 output) stays under ~0.5 GB.
LR_EDGE = {8: 2, 16: 1, 32: 1}
BATCH = {8: 2, 16: 1, 32: 1}


def model_hparams(model: nn.Module) -> dict:
    """Hyperparameters read off a built model, in model_summary.csv naming."""
    if isinstance(model, edsr.EDSR):
        return {"channel_size": model.head[0].out_channels}
    if isinstance(model, rcan.RCAN):
        return {"channel_size": model.head[0].out_channels,
                "n_resblocks": len(model.body[0].body) - 1, "n_resgroups": len(model.body) - 1}
    if isinstance(model, rrdb.RRDBNet):
        return {"first_channel_size": model.conv_first.out_channels,
                "n_rrdb_blocks": len(model.RRDB_trunk)}
    raise TypeError(type(model))


@pytest.fixture(scope="module")
def ref():
    """The upstream models package, imported by path without writing bytecode into it."""
    if not REF.exists():
        pytest.skip("reference checkout absent")
    spec = importlib.util.spec_from_file_location(
        "_ref_sr_models", REF / "__init__.py", submodule_search_locations=[str(REF)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pkg
    dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # the reference checkout is read-only
    try:
        spec.loader.exec_module(pkg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # torch.cuda.amp.custom_fwd
            mods = {m: importlib.import_module(f"_ref_sr_models.{m}")
                    for m in ("edsr", "rcan", "rrdb", "convfno")}
        yield mods
    finally:
        sys.dont_write_bytecode = dont_write
        for k in [k for k in sys.modules if k.startswith("_ref_sr_models")]:
            del sys.modules[k]


def ref_build(ref, name, size, ratio):
    mod, fn = REF_INIT[name]
    return getattr(ref[mod], fn)(size, ratio)


def lr_input(ratio, seed=0, edge=None):
    n = edge or LR_EDGE[ratio]
    return torch.randn(BATCH[ratio], 4, n, n, n, generator=torch.Generator().manual_seed(seed))


CASES = sorted(models.available()) + [("convfno", s, 8) for s in convfno.SIZES]


@pytest.mark.parametrize(("name", "size", "ratio"), CASES)
def test_port_equals_reference(ref, name, size, ratio):
    torch.manual_seed(1234)
    ours = models.build(name, size, ratio)
    torch.manual_seed(1234)
    theirs = ref_build(ref, name, size, ratio)
    sd_ours, sd_ref = ours.state_dict(), theirs.state_dict()
    assert list(sd_ours) == list(sd_ref)
    for k, v in sd_ref.items():  # same init stream: identical random weights, not just shapes
        assert sd_ours[k].shape == v.shape and torch.equal(sd_ours[k], v), k
    ours.load_state_dict(sd_ref, strict=True)
    assert models.param_count(ours) == sum(p.numel() for p in theirs.parameters())
    if name in SUMMARY:
        assert model_hparams(ours) == SUMMARY[name][size]
    x = lr_input(ratio)
    with torch.no_grad():
        y_ours, y_ref = ours.eval()(x.clone()), theirs.eval()(x.clone())
    assert y_ours.shape == (BATCH[ratio], 4, *(LR_EDGE[ratio] * ratio,) * 3)
    assert y_ours.dtype == torch.float32 and torch.equal(y_ours, y_ref)


@pytest.mark.parametrize("fname", ["rcan_approx0.5M_8xSR.pt", "rrdb_approx0.5M_8xSR.pt"])
def test_shipped_weights_load_strictly_and_match_reference(ref, fname):
    path = REF_WEIGHTS / fname
    if not path.exists():
        pytest.skip(f"{path} absent")
    ours = models.load_pretrained(path)
    name = fname.split("_")[0]
    assert ours.blastnet_meta == dict(name=name, size="0.5M", ratio=8, seed=42, path=str(path))
    assert not ours.training
    state = torch.load(path, map_location="cpu", weights_only=True)
    theirs = ref_build(ref, name, "0.5M", 8)
    theirs.load_state_dict(state)  # as upstream test.py does (RCAN: its own non-strict override)
    for k, v in state.items():
        assert torch.equal(ours.state_dict()[k], v) and torch.equal(theirs.state_dict()[k], v), k
    x = lr_input(8, seed=1, edge=4)
    with torch.no_grad():
        assert torch.equal(ours(x.clone()), theirs.eval()(x.clone()))


def test_available_is_the_shipped_inventory():
    av = models.available()
    assert len(av) == len(set(av)) == 40 and set(av) == SHIPPED
    assert all(s in models.SIZES[n] for n, s, _ in av)
    assert not any(n == "convfno" for n, _, _ in av)
    assert list(models.SIZES["convfno"]) == ["0.5M", "1.4M", "2.7M", "5M", "11M", "17M", "50M"]


def test_size_tables_agree_with_model_summary():
    keys = {"edsr": {"channel_size": "n_feats"},
            "rcan": {"channel_size": "n_feats", "n_resblocks": "n_resblocks",
                     "n_resgroups": "n_resgroups"},
            "rrdb": {"first_channel_size": "first_channel", "n_rrdb_blocks": "model_blocks"}}
    keys["rrdbgrad"] = keys["rrdb"]
    for name, rows in SUMMARY.items():
        assert list(models.SIZES[name]) == list(models.LABELS)
        for size, hp in rows.items():
            assert {k: models.SIZES[name][size][v] for k, v in keys[name].items()} == hp


def test_rrdbgrad_is_rrdb():
    torch.manual_seed(0)
    a = models.build("rrdbgrad", "0.8M")
    torch.manual_seed(0)
    b = models.build("rrdb", "0.8M")
    assert type(a) is rrdb.RRDBNet
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def test_build_rejects_unknown():
    with pytest.raises(ValueError):
        models.build("srcnn", "0.5M")
    with pytest.raises(ValueError):
        models.build("convfno", "0.8M")  # upstream init_convfno has no 0.8M
    with pytest.raises(ValueError):
        models.build("edsr", "3M")
    with pytest.raises(ValueError):
        models.load_pretrained("edsr_0.5M.pt")


def test_load_pretrained_parses_names(tmp_path):
    for sub, seed in [("seed43", 43), ("elsewhere", None)]:
        path = tmp_path / sub / "edsr_approx0.5M_16xSR.pt"
        path.parent.mkdir()
        torch.manual_seed(5)
        src = models.build("edsr", "0.5M", 16)
        torch.save(src.state_dict(), path)
        model = models.load_pretrained(path)
        assert model.blastnet_meta == dict(name="edsr", size="0.5M", ratio=16, seed=seed,
                                           path=str(path))
        assert model.upscale == 16
        assert all(torch.equal(v, src.state_dict()[k]) for k, v in model.state_dict().items())


@pytest.mark.parametrize(("name", "size", "n_blocks"), [
    ("edsr", "0.5M", 32), ("rcan", "0.5M", 1), ("rcan", "11M", 40), ("rrdb", "0.5M", 1),
    ("rrdb", "5M", 2), ("convfno", "0.5M", 32), ("gconv", "0.5M", None)])
def test_sites_resolve_and_hooks_fire_once(name, size, n_blocks):
    torch.manual_seed(0)
    model = models.build(name, size, 8).eval()
    s = models.sites(model)
    assert set(s) == {"stem", "blocks", "head"} and s["blocks"]
    if n_blocks is not None:
        assert len(s["blocks"]) == n_blocks
    order = [s["stem"], *s["blocks"], s["head"]]
    assert len(set(order)) == len(order)
    calls, outs = [], {}

    def hook(site):
        def fn(module, args, out):
            calls.append(site)
            outs[site] = out.clone()
        return fn

    handles = [model.get_submodule(site).register_forward_hook(hook(site)) for site in order]
    with torch.no_grad():
        y = model(lr_input(8))
    for h in handles:
        h.remove()
    assert calls == order  # each fires exactly once, in forward order
    assert torch.equal(outs[s["head"]], y)


def test_sites_rejects_unknown_model():
    with pytest.raises(TypeError):
        models.sites(nn.Conv3d(4, 4, 3))


def test_param_count():
    m = nn.Conv3d(4, 2, 3)
    assert models.param_count(m) == 4 * 2 * 27 + 2


def _subset_files():
    return sorted(SUBSET_WEIGHTS.glob("**/*.pt")) if SUBSET_WEIGHTS.exists() else []


@pytest.mark.skipif(not _subset_files(), reason="no weights under data/momentum128_subset")
@pytest.mark.parametrize("path", _subset_files(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_subset_weights_load_strictly(path):
    model = models.load_pretrained(path)
    meta = model.blastnet_meta
    assert (meta["name"], meta["size"], meta["ratio"]) in SHIPPED
    assert model_hparams(model) == SUMMARY[meta["name"]][meta["size"]]


@pytest.mark.skipif(not (SUBSET_WEIGHTS / "model_summary.csv").exists(),
                    reason="no model_summary.csv under data/momentum128_subset")
def test_subset_model_summary_agrees():
    import pandas as pd

    df = pd.read_csv(SUBSET_WEIGHTS / "model_summary.csv")
    seen = set()
    for _, row in df.iterrows():
        m = re.match(r"^([a-z]+)_approx([0-9.]+M)_(\d+)xSR\.pt$", row["filename"])
        assert m, row["filename"]
        name, size, ratio = m[1], m[2], int(m[3])
        hp = ast.literal_eval(row["hyperparams"])  # e.g. {'channel_size': 14, 'upscale': 8}
        assert row["model"] == name and row["approx_num_params"] == size
        assert hp.pop("upscale") == ratio
        assert hp == SUMMARY[name][size], (row["filename"], hp)
        seen.add((name, size, ratio))
    assert seen == SHIPPED
