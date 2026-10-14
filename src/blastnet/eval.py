"""Evaluate SR nets on Momentum128 splits with the upstream test metrics; shipped-weight sweep.

Averaging, as upstream test.py does it (pytorch_lightning 1.6.5 `trainer.test`, one GPU, batch
size 1, fp32): LitModel.test_step logs each metric with self.log defaults (on_epoch=True,
reduce_fx="mean"), so Lightning keeps a float32 running sum of (per-batch mean x batch size) and a
count, and reports sum / count, i.e. the batch-size-weighted mean of per-batch means. Every
metric is a mean over equal-size samples, so that is the per-cube mean over the split for any
batch size (exactly the plain mean over cubes at upstream's batch size 1). `evaluate` reproduces
it: float32 accumulator, value * batch size, divide by the cube count. No DDP sampler padding is
involved (upstream evaluates on a single GPU, and so do we).

Rows (`evaluate_weights`, `evaluate_shipped`) are JSON objects, one per (weights file, split),
written as JSON Lines: file, source, family, size, ratio, seed, split, n, params, seconds, then the
16 upstream test_* keys and the physics extras (spectrum_error, dissipation_error).
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

__all__ = [
    "EVAL_SPLITS",
    "META_KEYS",
    "UPSTREAM_KEYS",
    "evaluate",
    "evaluate_shipped",
    "evaluate_weights",
    "leaderboard",
    "read_rows",
    "shipped_files",
]

EVAL_SPLITS = ("test", "forcedhit", "paramvar")
UPSTREAM_KEYS = (
    "test_mse_rho", "test_ssim_rho", "test_mse_ux", "test_ssim_ux",
    "test_mse_uy", "test_ssim_uy", "test_mse_uz", "test_ssim_uz",
    "test_mse_sgs", "test_ssim_sgs",
    "test_mse_divsgs1", "test_ssim_divsgs1", "test_mse_divsgs2", "test_ssim_divsgs2",
    "test_mse_divsgs3", "test_ssim_divsgs3",
)
"""The 16 keys upstream LitModel.test_step logs, in its logging order (= blastnet.upstream.KEYS)."""
META_KEYS = ("file", "source", "family", "size", "ratio", "seed", "split", "n", "params",
             "seconds")

_FILENAME = re.compile(r"^(?P<family>[a-z]+)_approx(?P<size>[0-9.]+M)_(?P<ratio>\d+)xSR\.pt$")
_FAMILY_ORDER = ("edsr", "rcan", "rrdb", "rrdbgrad", "convfno", "gconv", "gconvrc")


@torch.no_grad()
def evaluate(model: nn.Module, root: str | os.PathLike, split: str = "test", ratio: int = 8,
             batch_size: int = 1, device: str | torch.device = "cpu", *, num_workers: int = 0,
             hr_size: int | None = None, extras: bool = True,
             allow_tf32: bool = False) -> dict[str, float]:
    """Split-mean upstream test metrics (+ physics extras) of `model`, averaged as upstream.

    The model sees upstream-normalised LR (norm_stats(split)) in float32 with autocast off; the
    metrics de-normalise with the same split statistics, as upstream test.py does per split.
    TF32 is disabled by default: on A100s it moved test_ssim_ux of rcan 0.5M by 2.4e-3 relative
    to the published value, against 4.7e-6 in strict fp32 (2026-10-10, all 173 test cubes).
    """
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    from blastnet import physics, upstream
    from blastnet.data import Momentum128
    from blastnet.train import infer_hr_size

    device = torch.device(device)
    if hr_size is None:
        hr_size = infer_hr_size(root, split)
    ds = Momentum128(root, split, ratio, normalize=True, augment=False, hr_size=hr_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=device.type == "cuda")
    model = model.to(device).eval()
    sums: dict[str, torch.Tensor] = {}
    count = 0
    for batch in dl:
        x, y = batch["lr"].to(device).float(), batch["hr"].to(device).float()
        dx = batch["dx"].to(device)
        with torch.autocast(device_type=device.type, enabled=False):
            y_hat = model(x).float()
        if extras:
            m = physics.metrics(y_hat, y, ratio, dx, ds.mean, ds.std)
        else:
            m = upstream.test_metrics(y_hat, y, ds.mean, ds.std, dx, ratio)
        b = x.shape[0]
        for k, v in m.items():
            acc = sums.get(k, torch.zeros((), dtype=torch.float32))
            sums[k] = acc + torch.tensor(float(v), dtype=torch.float32) * b
        count += b
    if count == 0:
        raise ValueError(f"split {split!r} under {root} is empty")
    keys = [*UPSTREAM_KEYS, *sorted(k for k in sums if k not in UPSTREAM_KEYS)]
    return {k: float(sums[k] / torch.tensor(count)) for k in keys}


def _parse(path: Path) -> dict:
    m = _FILENAME.match(path.name)
    if m is None:
        raise ValueError(f"cannot parse {path.name!r}; expected {{family}}_approx{{size}}_{{r}}xSR.pt")
    seed = re.fullmatch(r"seed(\d+)", path.parent.name)
    return dict(family=m["family"], size=m["size"], ratio=int(m["ratio"]),
                seed=int(seed[1]) if seed else None)


def read_rows(paths) -> list[dict]:
    """All JSON rows of one or more JSON Lines files (missing files are skipped)."""
    if isinstance(paths, str | os.PathLike):
        paths = [paths]
    rows = []
    for p in paths:
        p = Path(p)
        if p.exists():
            rows += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return rows


def evaluate_weights(path: str | os.PathLike, root: str | os.PathLike,
                     splits: tuple[str, ...] = EVAL_SPLITS, out_json: str | os.PathLike | None = None,
                     *, source: str = "shipped", file: str | None = None, batch_size: int = 1,
                     device: str | torch.device = "cpu", num_workers: int = 0,
                     hr_size: int | None = None, extras: bool = True,
                     skip: set | None = None) -> list[dict]:
    """Rows for one `{family}_approx{size}_{r}xSR.pt` over `splits`, appended to out_json (JSONL).

    (file, split) pairs in `skip` are not recomputed (resume after preemption).
    """
    from blastnet.data import load_meta
    from blastnet.models import load_pretrained, param_count

    path = Path(path)
    info = _parse(path)
    file = file or path.name
    model = None
    rows = []
    for split in splits:
        if skip and (file, split) in skip:
            continue
        if model is None:
            model = load_pretrained(path, map_location=device)
        t0 = time.monotonic()
        metrics = evaluate(model, root, split, info["ratio"], batch_size, device,
                           num_workers=num_workers, hr_size=hr_size, extras=extras)
        n = len(load_meta(root, split))
        row = dict(file=file, source=source, **info, split=split, n=n,
                   params=int(param_count(model)), seconds=round(time.monotonic() - t0, 2),
                   **metrics)
        rows.append(row)
        if out_json is not None:
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            with Path(out_json).open("a") as f:
                f.write(json.dumps(row) + "\n")
    return rows


def _size_value(size: str) -> float:
    return float(size.rstrip("M"))


def shipped_files(weights_dir: str | os.PathLike, families=None, seeds=None) -> list[Path]:
    """`seed*/{family}_approx{size}_{r}xSR.pt` under weights_dir, filtered, in cost order.

    Ordered by (size, ratio, family, seed) so that files[shard::num_shards] balances shards.
    """
    out = []
    for p in Path(weights_dir).glob("seed*/*.pt"):
        try:
            info = _parse(p)
        except ValueError:
            continue
        if families and info["family"] not in families:
            continue
        if seeds and info["seed"] not in {int(s) for s in seeds}:
            continue
        out.append((p, info))
    out.sort(key=lambda t: (_size_value(t[1]["size"]), t[1]["ratio"], t[1]["family"],
                            t[1]["seed"] if t[1]["seed"] is not None else -1))
    return [p for p, _ in out]


def evaluate_shipped(root: str | os.PathLike, out_json: str | os.PathLike, families=None,
                     seeds=None, splits: tuple[str, ...] = EVAL_SPLITS, *,
                     weights_dir: str | os.PathLike | None = None, shard: int = 0,
                     num_shards: int = 1, batch_size: int = 1,
                     device: str | torch.device = "cpu", num_workers: int = 0,
                     hr_size: int | None = None, extras: bool = True) -> list[dict]:
    """Evaluate every shipped weight file (root/best_model_weights/seed*/) on `splits`.

    Writes one JSON row per (file, split) to out_json (JSON Lines, appended; rows already there
    are skipped, so a preempted job resumes). shard/num_shards take files[shard::num_shards].
    Returns the rows computed in this call.
    """
    weights_dir = Path(weights_dir) if weights_dir else Path(root) / "best_model_weights"
    files = shipped_files(weights_dir, families, seeds)[shard::num_shards]
    done = {(r["file"], r["split"]) for r in read_rows(out_json)}
    rows = []
    for p in files:
        rel = p.relative_to(weights_dir).as_posix()
        rows += evaluate_weights(p, root, splits, out_json, source="shipped", file=rel,
                                 batch_size=batch_size, device=device, num_workers=num_workers,
                                 hr_size=hr_size, extras=extras, skip=done)
    return rows


def _fmt(values: list[float]) -> str:
    a = np.asarray(values, dtype=np.float64)
    if len(a) == 1:
        return f"{a[0]:.4g}"
    return f"{a.mean():.4g} ± {a.std(ddof=1):.2g}"


def leaderboard(rows: list[dict], keys: list[str] | None = None) -> str:
    """Markdown table: one line per (split, source, family, size, ratio), mean ± std over seeds.

    std is the sample standard deviation (ddof=1) over the rows of the group (seeds); a single
    row prints its value. keys default to the 16 upstream keys, then any other metric keys.
    """
    if keys is None:
        extra = sorted({k for r in rows for k in r if k not in META_KEYS and k not in UPSTREAM_KEYS})
        keys = [k for k in UPSTREAM_KEYS if any(k in r for r in rows)] + extra
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        g = (r["split"], r.get("source", "shipped"), r["family"], r["size"], int(r["ratio"]))
        groups.setdefault(g, []).append(r)

    def order(g):
        split, source, family, size, ratio = g
        si = EVAL_SPLITS.index(split) if split in EVAL_SPLITS else len(EVAL_SPLITS)
        fi = _FAMILY_ORDER.index(family) if family in _FAMILY_ORDER else len(_FAMILY_ORDER)
        return (si, split, source != "shipped", source, fi, family, _size_value(size), ratio)

    head = ["split", "source", "model", "size", "ratio", "seeds", "params", *keys]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for g in sorted(groups, key=order):
        rs = groups[g]
        params = sorted({r.get("params") for r in rs if r.get("params") is not None})
        cells = [g[0], g[1], g[2], g[3], f"{g[4]}x", str(len(rs)),
                 ",".join(f"{p:,}" for p in params)]
        for k in keys:
            vals = [r[k] for r in rs if r.get(k) is not None]
            cells.append(_fmt(vals) if vals else "")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
