import json

import numpy as np
import pytest
import torch

from blastnet import filters
from blastnet.data import VARS, Case, CubeDataset, open_case

N = (20, 16, 12)
SNAPS = 2
EXTRA = "TEMP_K"


def _write(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(arr, dtype="<f4").tofile(path)


def make_case(root, nonuniform=False, mislabel=False, variables=(*VARS, EXTRA), seed=0):
    rng = np.random.default_rng(seed)
    xs = [np.arange(n, dtype=np.float64) * d + o for n, d, o in zip(N, (0.5, 0.25, 2.0), (1, 0, 3))]
    if nonuniform:
        xs[1] = np.cumsum(np.linspace(0.1, 0.5, N[1]))
    X, Y, Z = np.meshgrid(*xs, indexing="ij")
    for a, g in zip("XYZ", (X, Y, Z), strict=True):
        _write(root / "grid" / f"{a}_m.dat", g)
    local = []
    for s in range(SNAPS):
        ent = {"id": s, "time": 0.1 * s}
        for v in variables:
            data = rng.random(N, dtype=np.float32) + (1.0 if v == VARS[0] else 0.0)
            _write(root / "data" / f"{v}_id{s:03d}.dat", data)
            ent[f"{v} filename"] = f"./data/{v}_id{s:03d}.dat"
        local.append(ent)
    if mislabel:
        local[1][f"{VARS[1]} filename"] = f"./data/{VARS[2]}_id001.dat"
    info = {"global": {"dataset_id": "synth", "Nxyz": list(N), "snapshots": SNAPS,
                       "variables": list(variables), "compression": None,
                       "grid": {a: f"./grid/{a.upper()}_m.dat" for a in "xyz"},
                       "bc": {"description": "periodic"}},
            "local": local}
    (root / "info.json").write_text(json.dumps(info))
    return root


@pytest.fixture
def case(tmp_path):
    return open_case(make_case(tmp_path / "c"))


def test_metadata(case):
    assert isinstance(case, Case)
    assert case.id == "synth" and case.shape == N and case.n_snapshots == SNAPS
    assert case.variables == (*VARS, EXTRA)
    assert case.bc == {"description": "periodic"}
    assert case.global_meta["Nxyz"] == list(N) and len(case.local_meta) == SNAPS
    assert Case(case.root).id == "synth"


def test_coords_spacing(case):
    x, y, z = case.coords()
    assert [len(a) for a in (x, y, z)] == list(N)
    np.testing.assert_allclose(x, np.arange(N[0]) * 0.5 + 1)
    np.testing.assert_allclose(z, np.arange(N[2]) * 2.0 + 3)
    dx = case.spacing()
    assert dx == pytest.approx((0.5, 0.25, 2.0))
    assert case.check() == []


def test_nonuniform_axis(tmp_path):
    c = open_case(make_case(tmp_path / "n", nonuniform=True))
    assert c.spacing()[1] is None and c.spacing()[0] == pytest.approx(0.5)
    assert any("non-uniform" in m and "y" in m for m in c.check())


def test_field_and_fields(case):
    f = case.field(EXTRA, 1)
    assert f.shape == N and f.dtype == np.float32 and not f.flags.writeable
    full = case.fields(VARS, snapshot=1)
    assert full.shape == (4, *N)
    np.testing.assert_array_equal(full[0], case.field(VARS[0], 1))
    reg = (slice(2, 10), slice(0, 4), slice(5, 12))
    part = case.fields(VARS, snapshot=1, region=reg)
    assert part.shape == (4, 8, 4, 7) and not isinstance(part, np.memmap)
    np.testing.assert_array_equal(part, full[(slice(None), *reg)])
    assert case.fields(EXTRA).shape == (1, *N)
    with pytest.raises(KeyError):
        case.field("NOPE")
    with pytest.raises(IndexError):
        case.field(EXTRA, 5)


def test_check_problems(tmp_path):
    root = make_case(tmp_path / "m", mislabel=True)
    (root / "data" / f"{EXTRA}_id000.dat").unlink()
    (root / "data" / f"{VARS[3]}_id001.dat").write_bytes(b"\0" * 8)
    msgs = open_case(root).check()
    assert any("missing file" in m and f"{EXTRA}_id000" in m for m in msgs)
    assert any("size mismatch" in m and f"{VARS[3]}_id001" in m for m in msgs)
    assert any("label mismatch" in m and VARS[1] in m for m in msgs)


@pytest.fixture
def momentum(tmp_path):
    return open_case(make_case(tmp_path / "mom", variables=VARS))


def test_tile(momentum):
    ds = CubeDataset(momentum, size=8)
    # 20x16x12 with size 8: 2 x 2 x 1 origins per snapshot
    assert len(ds) == 2 * 2 * 1 * SNAPS
    it = ds[0]
    assert it["hr"].shape == (4, 8, 8, 8) and "lr" not in it
    assert it["case"] == "synth" and it["snapshot"] == 0 and it["origin"] == (0, 0, 0)
    assert it["dx"].dtype == torch.float32
    np.testing.assert_allclose(it["dx"].numpy(), [0.5, 0.25, 2.0])
    np.testing.assert_array_equal(
        ds[1]["hr"].numpy(), momentum.fields(VARS, 0, tuple(slice(o, o + 8) for o in ds[1]["origin"])))
    assert {ds[i]["snapshot"] for i in range(len(ds))} == {0, 1}
    assert len(CubeDataset(momentum, size=8, snapshots=[1])) == 4


def test_tile_stride(momentum):
    ds = CubeDataset(momentum, size=8, stride=4, snapshots=[0])
    assert len(ds) == 4 * 3 * 2  # origins 0,4,8,12 / 0,4,8 / 0,4
    assert ds[len(ds) - 1]["origin"] == (12, 8, 4)


def test_random(momentum):
    a = CubeDataset(momentum, size=8, mode="random", n_random=7, seed=3)
    b = CubeDataset(momentum, size=8, mode="random", n_random=7, seed=3)
    c = CubeDataset(momentum, size=8, mode="random", n_random=7, seed=4)
    assert len(a) == 7 and a.items == b.items and a.items != c.items
    for i in range(len(a)):
        o = a[i]["origin"]
        assert all(0 <= x <= n - 8 for x, n in zip(o, N, strict=True))
    with pytest.raises(ValueError):
        CubeDataset(momentum, size=8, mode="random")
    with pytest.raises(ValueError):
        CubeDataset(momentum, size=8, mode="bogus")


@pytest.mark.parametrize("kind", filters.KINDS)
def test_lr_matches_make_lr(momentum, kind):
    ds = CubeDataset(momentum, size=8, ratio=4, lr_kind=kind, snapshots=[1])
    it = ds[2]
    assert it["lr"].shape == (4, 2, 2, 2)
    hr = torch.from_numpy(momentum.fields(
        VARS, 1, tuple(slice(o, o + 8) for o in it["origin"])))
    torch.testing.assert_close(it["lr"], filters.make_lr(hr[None], 4, kind=kind)[0], rtol=0, atol=0)


def test_normalize(momentum):
    mean, std = [1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 5.0, 10.0]
    raw = CubeDataset(momentum, size=8, ratio=4)[0]
    nrm = CubeDataset(momentum, size=8, ratio=4, normalize=(mean, std))[0]
    m = torch.tensor(mean).reshape(4, 1, 1, 1)
    s = torch.tensor(std).reshape(4, 1, 1, 1)
    torch.testing.assert_close(nrm["hr"], (raw["hr"] - m) / s)
    torch.testing.assert_close(nrm["lr"], (raw["lr"] - m) / s)


def test_nan_dx_and_validation(tmp_path, case):
    c = open_case(make_case(tmp_path / "n", nonuniform=True, variables=VARS))
    dx = CubeDataset([c], size=8)[0]["dx"].numpy()
    assert np.isnan(dx[1]) and not np.isnan(dx[[0, 2]]).any()
    with pytest.raises(ValueError, match="favre_box"):  # wrong variable order/set
        CubeDataset(case, variables=(VARS[1], VARS[0], VARS[2], VARS[3]), size=8, ratio=4)
    with pytest.raises(ValueError, match="multiple"):
        CubeDataset(case, size=8, ratio=3)
    with pytest.raises(ValueError, match="smaller"):
        CubeDataset(case, size=16)
    # a path is accepted in place of a Case, and a general variable set works without ratio
    ds = CubeDataset(case.root, variables=(EXTRA,), size=8)
    assert ds[0]["hr"].shape == (1, 8, 8, 8)


def test_free_text_bc_and_partial_local(tmp_path):
    # Regression from the real BLASTNet tutorial toy dataset (2026-10-10): global "bc" is a string,
    # and global "snapshots" (98) counts snapshots that the archive's "local" list does not ship.
    root = make_case(tmp_path / "c")
    info = json.loads((root / "info.json").read_text())
    info["global"]["bc"] = "Periodic in x-, y-, and z-directions."
    info["global"]["snapshots"] = 98
    (root / "info.json").write_text(json.dumps(info))
    c = open_case(root)
    assert c.bc == {"description": "Periodic in x-, y-, and z-directions."}
    assert c.n_snapshots == len(info["local"])
    assert c.field(VARS[0], snapshot=0).shape == tuple(N)
