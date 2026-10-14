import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from blastnet import octahedral
from blastnet.data import (
    CHANNELS,
    COLUMNS,
    SPLITS,
    VARS,
    Momentum128,
    Momentum128Zarr,
    cube_path,
    load_meta,
    norm_stats,
    read_cube,
    read_dat,
    subset,
    synthetic,
    zarr_convert,
)

SIZE, RATIOS = 16, (4, 8)
REF = Path(__file__).resolve().parents[2] / "reference/blastnet2_sr_benchmark/common/data.py"
SUBSET = Path(__file__).resolve().parents[1] / "data/momentum128_subset"


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    return synthetic.write_tree(tmp_path_factory.mktemp("tree"), n_per_split=3, size=SIZE,
                                ratios=RATIOS)


def _upstream():
    """get_mean_std* and ScaleTransform from the upstream reader (needs only torch)."""
    if not REF.exists():
        pytest.skip("upstream reference checkout absent")
    names = {"get_mean_std", "get_mean_std_test", "get_mean_std_paramvar",
             "get_mean_std_forcedhit", "ScaleTransform"}
    body = [n for n in ast.parse(REF.read_text()).body if getattr(n, "name", None) in names]
    ns = {"torch": torch}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(REF), "exec"), ns)
    return ns


_UPSTREAM_FN = {"train": "get_mean_std", "val": "get_mean_std", "test": "get_mean_std_test",
                "paramvar": "get_mean_std_paramvar", "forcedhit": "get_mean_std_forcedhit"}


def _raw(root, split, h, ratio):
    return np.stack(read_cube(root, split, h, ratio, SIZE))


def test_norm_stats_match_upstream():
    mean, std = norm_stats("train")
    assert mean.tolist() == pytest.approx([0.24, 28.0, 28.0, 28.0])
    assert std.tolist() == pytest.approx([0.068, 48.0, 48.0, 48.0])
    assert torch.equal(norm_stats("val")[0], mean)
    with pytest.raises(ValueError):
        norm_stats("bogus")
    up = _upstream()
    for split, fn in _UPSTREAM_FN.items():
        um, us = up[fn]()
        m, s = norm_stats(split)
        assert torch.equal(m, um) and torch.equal(s, us) and m.dtype == torch.float32, split


def test_synthetic_cubes():
    d = synthetic.cubes(3, size=SIZE, ratio=8, seed=0)
    hr, lr = d["hr"], d["lr"]
    assert hr.shape == (3, 4, 16, 16, 16) and lr.shape == (3, 4, 2, 2, 2)
    assert hr.dtype == lr.dtype == torch.float32
    assert torch.equal(synthetic.cubes(3, seed=0)["hr"], hr)
    assert not torch.equal(synthetic.cubes(3, seed=1)["hr"], hr)
    h = hr.double()
    assert h[:, 0].mean().item() == pytest.approx(0.24, abs=0.01)
    assert h[:, 0].std().item() == pytest.approx(0.068, rel=0.1) and h[:, 0].min() > 0
    assert h[:, 1:].mean().item() == pytest.approx(28.0, abs=1.0)
    assert h[:, 1:].std().item() == pytest.approx(48.0, rel=0.05)
    # LR = box(rho), Favre(u, v, w) over 8^3 blocks
    def box(x):
        return torch.nn.functional.avg_pool3d(x, 8)

    rho = h[:, :1]
    ref = torch.cat([box(rho), box(rho * h[:, 1:]) / box(rho)], 1)
    assert torch.allclose(lr.double(), ref, rtol=1e-6, atol=1e-6)
    # divergence-weak, non-periodic, smooth
    u = h[:, 1:]
    grads = [torch.gradient(u[:, i], dim=(1, 2, 3)) for i in range(3)]
    div = sum(grads[i][i] for i in range(3))
    full = torch.stack([g for gs in grads for g in gs])
    assert div.pow(2).mean().sqrt() < 0.15 * full.pow(2).mean().mul(3).sqrt()
    assert not torch.allclose(u[..., 0], u[..., -1], atol=1.0)
    meta = d["meta"]
    assert len(meta) == 3 and meta["hash"].map(type).eq(str).all()
    assert (meta.loc[1, "dx"] != meta.loc[1, "dy"]) and meta.loc[1, "dy"] == meta.loc[1, "dz"]


def test_write_tree_layout_and_load_meta(tree):
    assert (tree / "best_model_weights").is_dir() and not any((tree / "best_model_weights").iterdir())
    total = 0
    for split in SPLITS:
        raw = pd.read_csv(tree / f"{split}_data_summary.csv", dtype=str, keep_default_na=False)
        assert tuple(raw.columns) == COLUMNS and len(raw) == 3
        total += len(raw)
        for h in raw["hash"]:
            assert int(h) and str(int(h)) == h
            for v in VARS:
                assert cube_path(tree, split, h, v).stat().st_size == SIZE**3 * 4
                for r in RATIOS:
                    p = tree / "dataset" / f"LR_{r}x" / split / f"{v}_id{h}.dat"
                    assert p.stat().st_size == (SIZE // r) ** 3 * 4
    assert len(pd.read_csv(tree / "all_data_summary.csv")) == total
    meta = load_meta(tree, "val")
    assert meta["hash"].map(type).eq(str).all() and (meta["split"] == "val").all()
    assert np.isnan(meta.loc[1, "dy [m]"]) and meta.loc[1, "dy"] == meta.loc[1, "dx"]
    assert meta.loc[1, "dz"] == meta.loc[1, "dx"]
    train = load_meta(tree, "train")
    assert train.loc[1, "dx"] != train.loc[1, "dy"]
    with pytest.raises(ValueError):
        read_dat(cube_path(tree, "val", meta["hash"][0], VARS[0]), (8, 8, 8))


def test_item_matches_files_and_dx_in_metres(tree):
    ds = Momentum128(tree, "train", ratio=8, normalize=False, hr_size=SIZE)
    assert len(ds) == 3 and ds.meta is not None
    for i in range(len(ds)):
        item = ds[i]
        h = ds.meta["hash"][i]
        assert set(item) == {"lr", "hr", "dx", "index", "hash"}
        assert item["hash"] == h and item["index"] == i
        assert torch.equal(item["hr"], torch.from_numpy(_raw(tree, "train", h, None)))
        assert torch.equal(item["lr"], torch.from_numpy(_raw(tree, "train", h, 8)))
        assert item["dx"].dtype == torch.float32
        row = ds.meta.iloc[i]
        assert torch.equal(item["dx"], torch.tensor([row.dx, row.dy, row.dz], dtype=torch.float32))
        assert 1e-6 < item["dx"].min() < 1e-4  # metres, not dx / DX_MIN
    with pytest.raises(ValueError):
        Momentum128(tree, "train", ratio=8, crop=12, hr_size=SIZE)
    with pytest.raises(IndexError):
        ds[3]


@pytest.mark.parametrize("split", SPLITS)
def test_normalize_matches_upstream_scale_transform(tree, split):
    up = _upstream()
    transform = up["ScaleTransform"](*up[_UPSTREAM_FN[split]]())
    ds = Momentum128(tree, split, ratio=4, normalize=True, hr_size=SIZE)
    item = ds[0]
    h = ds.meta["hash"][0]
    assert torch.equal(item["hr"], transform(torch.from_numpy(_raw(tree, split, h, None))))
    assert torch.equal(item["lr"], transform(torch.from_numpy(_raw(tree, split, h, 4))))


def test_crop_aligned_and_reproducible(tree):
    ds = Momentum128(tree, "train", ratio=4, crop=8, normalize=False, seed=3, hr_size=SIZE)
    offsets = set()
    for epoch in range(4):
        ds.set_epoch(epoch)
        for i in range(len(ds)):
            item = ds[i]
            off, g = ds.sample_params(i)
            assert g is None and all(o % 4 == 0 and 0 <= o <= SIZE - 8 for o in off)
            offsets.add(off)
            h = ds.meta["hash"][i]
            hr = _raw(tree, "train", h, None)[:, off[0]:off[0] + 8, off[1]:off[1] + 8,
                                              off[2]:off[2] + 8]
            lo = [o // 4 for o in off]
            lr = _raw(tree, "train", h, 4)[:, lo[0]:lo[0] + 2, lo[1]:lo[1] + 2, lo[2]:lo[2] + 2]
            assert item["hr"].shape == (4, 8, 8, 8) and item["lr"].shape == (4, 2, 2, 2)
            assert torch.equal(item["hr"], torch.from_numpy(hr))
            assert torch.equal(item["lr"], torch.from_numpy(lr))
            again = Momentum128(tree, "train", ratio=4, crop=8, normalize=False, seed=3,
                                hr_size=SIZE)
            again.set_epoch(epoch)
            assert torch.equal(again[i]["hr"], item["hr"])
    assert len(offsets) > 1


def test_augment_is_octahedral_apply_of_plain_item(tree):
    kw = dict(ratio=4, crop=8, normalize=True, seed=5, hr_size=SIZE)
    aug = Momentum128(tree, "train", augment=True, **kw)
    plain = Momentum128(tree, "train", augment=False, **kw)
    drawn = set()
    for epoch in range(6):
        aug.set_epoch(epoch)
        plain.set_epoch(epoch)
        for i in range(len(aug)):
            off, g = aug.sample_params(i)
            assert off == plain.sample_params(i)[0]
            drawn.add(g.index)
            a, p = aug[i], plain[i]
            assert torch.equal(a["hr"], octahedral.apply(p["hr"], g, (1, 2, 3)))
            assert torch.equal(a["lr"], octahedral.apply(p["lr"], g, (1, 2, 3)))
            assert torch.equal(a["dx"], p["dx"][list(g.perm)])
    assert len(drawn) > 3


def test_preload_and_dataloader(tree):
    kw = dict(ratio=8, crop=8, augment=True, seed=1, hr_size=SIZE)
    lazy, ram = Momentum128(tree, "test", **kw), Momentum128(tree, "test", preload=True, **kw)
    for i in range(len(lazy)):
        a, b = lazy[i], ram[i]
        assert all(torch.equal(a[k], b[k]) for k in ("lr", "hr", "dx"))
    batch = next(iter(torch.utils.data.DataLoader(lazy, batch_size=2)))
    assert batch["hr"].shape == (2, 4, 8, 8, 8) and batch["dx"].shape == (2, 3)
    assert batch["hash"] == list(lazy.meta["hash"][:2])


def test_zarr_roundtrip(tree, tmp_path):
    out = zarr_convert(tree, tmp_path / "m.zarr", splits=("train", "val"), ratios=RATIOS,
                       chunks=(1, 4, 128, 128, 128), hr_size=SIZE)
    import zarr

    g = zarr.open_group(out, mode="r")
    assert g.attrs["channels"] == list(CHANNELS)
    for split in ("train", "val"):
        meta = load_meta(tree, split)
        assert g[split]["hr"].shape == (3, 4, SIZE, SIZE, SIZE)
        for r in (None, *RATIOS):
            arr = g[split]["hr" if r is None else f"lr_{r}x"][...]
            ref = np.stack([_raw(tree, split, h, r) for h in meta["hash"]])
            assert arr.dtype == np.float32 and np.array_equal(arr, ref)
        for kw in (dict(normalize=False), dict(crop=8, augment=True, seed=2)):
            dz = Momentum128Zarr(out, split, ratio=4, **kw)
            dd = Momentum128(tree, split, ratio=4, hr_size=SIZE, **kw)
            pd.testing.assert_frame_equal(dz.meta, dd.meta)
            for i in range(len(dd)):
                a, b = dz[i], dd[i]
                assert a["hash"] == b["hash"]
                assert all(torch.equal(a[k], b[k]) for k in ("lr", "hr", "dx"))


def test_subset_roundtrip(tree, tmp_path):
    out = subset(tree, tmp_path / "sub", n_per_split=2)
    for split in SPLITS:
        src = pd.read_csv(tree / f"{split}_data_summary.csv", dtype=str, keep_default_na=False)
        dst = pd.read_csv(out / f"{split}_data_summary.csv", dtype=str, keep_default_na=False)
        pd.testing.assert_frame_equal(dst, src.iloc[:2])
        for r in RATIOS:
            a = Momentum128(out, split, ratio=r, hr_size=SIZE)
            b = Momentum128(tree, split, ratio=r, hr_size=SIZE)
            assert len(a) == 2
            for i in range(2):
                assert all(torch.equal(a[i][k], b[i][k]) for k in ("lr", "hr", "dx"))
    assert len(pd.read_csv(out / "all_data_summary.csv")) == 2 * len(SPLITS)
    s1 = subset(tree, tmp_path / "s1", n_per_split=1, ratios=(8,), splits=("val",), seed=7)
    s2 = subset(tree, tmp_path / "s2", n_per_split=1, ratios=(8,), splits=("val",), seed=7)
    h1, h2 = (load_meta(s, "val")["hash"].tolist() for s in (s1, s2))
    assert h1 == h2 and h1[0] in load_meta(tree, "val")["hash"].tolist()
    assert not (s1 / "dataset" / "LR_4x").exists()


@pytest.mark.skipif(not (SUBSET / "train_data_summary.csv").exists(),
                    reason="laptop subset absent")
def test_real_subset_reads():
    for split in SPLITS:
        if not (SUBSET / f"{split}_data_summary.csv").exists():
            continue
        ds = Momentum128(SUBSET, split, ratio=8, normalize=True)
        item = ds[0]
        assert item["hr"].shape == (4, 128, 128, 128) and item["lr"].shape == (4, 16, 16, 16)
        assert torch.isfinite(item["hr"]).all() and torch.isfinite(item["lr"]).all()
        assert (item["dx"] > 0).all() and item["dx"].max() < 1e-3
