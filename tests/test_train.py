"""Trainer smoke (16^3 synthetic, CPU), evaluator, shipped sweep, CLI, PBS scripts.

The evaluation tests use a 32^3 / 8x synthetic tree: the upstream metrics remove one LR cell off
every face and run a 9^3 SSIM window, which needs at least 4^3 LR cells (16^3 / 8x has 2^3).
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from blastnet import octahedral
from blastnet.cli import main
from blastnet.data import synthetic
from blastnet.eval import (
    UPSTREAM_KEYS,
    evaluate,
    evaluate_shipped,
    leaderboard,
    read_rows,
    shipped_files,
)
from blastnet.models import build, load_pretrained
from blastnet.train import (
    UpstreamAugment,
    export_state_dict,
    infer_hr_size,
    parse_time,
    run_name,
    train,
    upstream_augment,
)

REPO = Path(__file__).resolve().parents[1]
JOBS = REPO / "jobs" / "polaris"
REF_WEIGHTS = REPO.parent / "reference" / "blastnet2_sr_benchmark" / "weights" / "seed42"
CPU = dict(accelerator="cpu", devices=1, precision="32", num_workers=0)


@pytest.fixture(scope="module")
def tree16(tmp_path_factory):
    return synthetic.write_tree(tmp_path_factory.mktemp("t16"), n_per_split=2, size=16,
                                ratios=(8,), splits=("train", "val"))


@pytest.fixture(scope="module")
def tree32(tmp_path_factory):
    return synthetic.write_tree(tmp_path_factory.mktemp("t32"), n_per_split=2, size=32,
                                ratios=(8,), splits=("test", "paramvar"))


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_names_and_times(tree16):
    assert run_name("edsr", "0.8M", 8, "mse", True, 0) == "edsr-0.8M-r8-mse-aug-s0"
    assert run_name("gconv", "0.5M", 8, "mse_grad", False, 3) == "gconv-0.5M-r8-mse_grad-noaug-s3"
    assert parse_time("00:11:58:00") == 11 * 3600 + 58 * 60
    assert parse_time("01:20:00") == 4800 and parse_time(90) == 90.0 and parse_time(None) is None
    assert infer_hr_size(tree16, "train") == 16


def test_upstream_augment_distribution():
    # every upstream draw is an exact O_h action (lr and hr alike, dx permuted with g); the draw
    # reaches 32 of 48 elements and never one whose axis permutation is a 3-cycle
    x = torch.arange(4 * 8, dtype=torch.float64).reshape(4, 2, 2, 2) + 1
    key = {octahedral.apply(x, g, (1, 2, 3)).numpy().tobytes(): g for g in octahedral.elements()}
    dx = torch.tensor([1.0, 2.0, 3.0])
    seen = set()
    for s in range(600):
        lr, hr, d = upstream_augment(x.clone(), x.clone(), dx, torch.Generator().manual_seed(s))
        g = key[lr.numpy().tobytes()]
        assert torch.equal(hr, lr)
        assert torch.equal(d, dx[list(g.perm)])
        seen.add(g.index)
    perms = {tuple(octahedral.elements()[i].perm) for i in seen}
    assert len(seen) == 32
    assert all(sum(p != i for i, p in enumerate(perm)) != 3 for perm in perms)


def test_upstream_augment_dataset_is_seeded(tree16):
    from blastnet.data import Momentum128

    ds = UpstreamAugment(Momentum128(tree16, "train", 8, hr_size=16), seed=1)
    a, b = ds[0], ds[0]
    assert torch.equal(a["hr"], b["hr"]) and a["lr"].shape == (4, 2, 2, 2)
    draws = set()
    for e in range(8):
        ds.set_epoch(e)
        draws.add(ds[0]["hr"].numpy().tobytes())
    assert len(draws) > 1


def test_train_smoke_edsr(tree16, tmp_path):
    run = train(tree16, "edsr", "0.5M", 8, "mse", aug=True, seed=0, out=tmp_path, max_steps=2,
                batch_size=1, **CPU)
    assert run.name == "edsr-0.5M-r8-mse-aug-s0"
    rows = _rows(run / "metrics.jsonl")
    assert len(rows) == 1 and rows[0]["epoch"] == 0 and rows[0]["step"] == 2
    assert math.isfinite(rows[0]["train_loss"]) and math.isfinite(rows[0]["val_mseloss"])
    assert rows[0]["lr"] == pytest.approx(1e-4) and rows[0]["wall_s"] >= 0
    assert (run / f"{run.name}.ckpt").exists() and (run / "last.ckpt").exists()
    pt = run / "edsr_approx0.5M_8xSR.pt"
    model = load_pretrained(pt)  # our export reads like a shipped file (strict load)
    ckpt = torch.load(run / f"{run.name}.ckpt", map_location="cpu", weights_only=False)
    assert ckpt["hyper_parameters"]["loss"] == "mse"
    sd = model.state_dict()
    for k, v in ckpt["state_dict"].items():
        if k.startswith("model."):
            assert torch.equal(sd[k[6:]], v)
    out = export_state_dict(run / "last.ckpt", tmp_path / "x.pt")
    assert set(torch.load(out, weights_only=True)) == set(sd)


def test_train_gradient_clip_gets_its_own_run_dir(tree16, tmp_path):
    # 2026-10-10: gconv 0.8M/1.4M diverged under the unclipped upstream recipe; clipped reruns must
    # never resume the diverged run's last.ckpt, so they live in a "-clip<v>" directory.
    run = train(tree16, "edsr", "0.5M", 8, "mse", aug=False, seed=0, out=tmp_path, max_steps=1,
                batch_size=1, gradient_clip_val=1.0, **CPU)
    assert run.name == "edsr-0.5M-r8-mse-noaug-s0-clip1"
    assert math.isfinite(_rows(run / "metrics.jsonl")[0]["train_loss"])


def test_train_grad_loss_wallclock_and_resume(tree16, tmp_path):
    kw = dict(seed=0, out=tmp_path, **CPU)
    # max_time 0: stops after the first optimiser step (Lightning validates once on the stop
    # request), writing the best ckpt, last.ckpt and the .pt
    run = train(tree16, "gconv", "0.5M", 8, "mse_grad", aug=False, max_time=0, **kw)
    rows = _rows(run / "metrics.jsonl")
    assert rows[-1]["step"] == 1 and math.isfinite(rows[-1]["val_mseloss"])
    assert (run / f"{run.name}.ckpt").exists()
    assert "train_weightedgradloss" in rows[-1] and math.isfinite(rows[-1]["train_loss"])
    assert (run / "last.ckpt").exists() and (run / "gconv_approx0.5M_8xSR.pt").exists()
    # resume from last.ckpt and finish two more epochs (2 cubes, batch 1 -> 2 steps each)
    train(tree16, "gconv", "0.5M", 8, "mse_grad", aug=False, epochs=2, **kw)
    rows = _rows(run / "metrics.jsonl")
    assert rows[-1]["epoch"] == 1 and rows[-1]["step"] > 1
    assert math.isfinite(rows[-1]["val_mseloss"])
    load_pretrained(run / "gconv_approx0.5M_8xSR.pt")


def test_evaluate_keys_and_averaging(tree32):
    torch.manual_seed(0)
    model = build("edsr", "0.5M", 8)
    r1 = evaluate(model, tree32, "test", 8, batch_size=1, device="cpu")
    r2 = evaluate(model, tree32, "test", 8, batch_size=2, device="cpu", extras=False)
    assert list(r1)[:16] == list(UPSTREAM_KEYS) and len(UPSTREAM_KEYS) == 16
    assert {"spectrum_error", "dissipation_error"} <= set(r1)
    assert list(r2) == list(UPSTREAM_KEYS)
    assert all(np.isfinite(v) for v in r1.values())
    for k in UPSTREAM_KEYS:  # per-cube mean regardless of batch size, as upstream's test loop
        assert r2[k] == pytest.approx(r1[k], rel=1e-4, abs=1e-6), k


def test_evaluate_shipped_and_leaderboard(tree32, tmp_path):
    if not REF_WEIGHTS.exists():
        pytest.skip("reference checkout absent")
    wdir = tree32 / "best_model_weights" / "seed42"
    wdir.mkdir(parents=True, exist_ok=True)
    for p in REF_WEIGHTS.glob("*.pt"):
        shutil.copy(p, wdir / p.name)
    files = shipped_files(tree32 / "best_model_weights")
    assert [p.name for p in files] == ["rcan_approx0.5M_8xSR.pt", "rrdb_approx0.5M_8xSR.pt"]
    assert shipped_files(tree32 / "best_model_weights", families=["rrdb"], seeds=[42]) == files[1:]
    out = tmp_path / "rows.jsonl"
    rows = evaluate_shipped(tree32, out, splits=("test", "paramvar"), extras=False)
    assert len(rows) == 4 and len(read_rows(out)) == 4
    assert {(r["file"], r["split"]) for r in rows} == {
        (f"seed42/{f.name}", s) for f in files for s in ("test", "paramvar")}
    r = rows[0]
    assert r["seed"] == 42 and r["ratio"] == 8 and r["n"] == 2 and r["params"] > 0
    assert all(np.isfinite(r[k]) for k in UPSTREAM_KEYS)
    assert evaluate_shipped(tree32, out, splits=("test", "paramvar")) == []  # resumes: all done
    table = leaderboard(read_rows(out))
    lines = table.strip().splitlines()
    assert len(lines) == 2 + 4 and "test_ssim_divsgs3" in lines[0]
    assert lines[2].startswith("| test | shipped | rcan | 0.5M | 8x | 1 |")


def test_cli_eval_and_leaderboard(tree32, tmp_path, capsys):
    run = tmp_path / "edsr-0.5M-r8-mse-noaug-s0"
    run.mkdir()
    torch.manual_seed(0)
    torch.save(build("edsr", "0.5M", 8).state_dict(), run / "edsr_approx0.5M_8xSR.pt")
    out = tmp_path / "ours.jsonl"
    assert main(["eval", str(run / "edsr_approx0.5M_8xSR.pt"), "--root", str(tree32),
                 "--out", str(out), "--splits", "test", "--device", "cpu", "--num-workers", "0",
                 "--no-extras"]) == 0
    (row,) = read_rows(out)
    assert row["source"] == run.name and row["file"] == f"{run.name}/edsr_approx0.5M_8xSR.pt"
    md = tmp_path / "lb.md"
    assert main(["leaderboard", str(out), "--out", str(md), "--keys", "test_mse_ux"]) == 0
    assert md.read_text() == capsys.readouterr().out.split("\n", 1)[1]
    assert "| test | edsr-0.5M-r8-mse-noaug-s0 | edsr | 0.5M | 8x | 1 |" in md.read_text()


def test_cli_convert(tree16, tmp_path):
    out = tmp_path / "z.zarr"
    assert main(["convert", str(tree16), str(out), "--splits", "val", "--hr-size", "16"]) == 0
    from blastnet.data import Momentum128, Momentum128Zarr

    a, b = Momentum128(tree16, "val", 8, hr_size=16)[1], Momentum128Zarr(out, "val", 8)[1]
    assert torch.equal(a["hr"], b["hr"]) and torch.equal(a["lr"], b["lr"])


def test_pbs_scripts_parse_and_carry_the_polaris_header():
    for s in ("common.sh", "env.sh", "train.pbs", "eval_shipped.pbs", "run_python.pbs"):
        subprocess.run(["bash", "-n", str(JOBS / s)], check=True)
    for s in ("train.pbs", "eval_shipped.pbs", "run_python.pbs"):
        text = (JOBS / s).read_text()
        for line in ("#PBS -A YOUR_PROJECT", "#PBS -l filesystems=home:eagle",
                     "#PBS -l select=1:system=polaris", "#PBS -l place=scatter"):
            assert line in text, (s, line)
        assert "jobs/polaris/common.sh" in text
