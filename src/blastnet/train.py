"""Train SR nets on Momentum128 with the upstream BLASTNet 2.0 recipe (Lightning 2.x).

Upstream: reference/blastnet2_sr_benchmark (commit ae2a3a6) train.py + models/litmodel.py, run with
pytorch_lightning 1.6.5. Reproduced here (train.py defaults):

    Adam(lr=1e-4); StepLR(step_size=300 * lr_sched_factor, gamma=0.5) stepped per epoch, with
    lr_sched_factor 0.5 (so 150 epochs); 600 epochs; batch size 1 per GPU; fp16 mixed precision;
    loss "mse" or "mse_grad" = 0.99 * GradLoss + 0.01 * MSE (normalised units, GradLoss dx = 1);
    full 128^3 cubes (upstream never crops); train/val normalised with the train mean/std;
    validation every epoch on full cubes, in float32 with autocast off (upstream decorates
    validation_step with custom_fwd(cast_inputs=float32)); val_mseloss = MSE in normalised units;
    best checkpoint by val_mseloss (min) plus last; DDP, find_unused_parameters=False,
    sync_batchnorm=True, LightningEnvironment; pl.seed_everything(seed) (workers not seeded).
    Augmentation (train split only): upstream random_rot90_3D then random_flip_3D, after
    normalisation, ported literally in `upstream_augment`. Its per-sample draw is NOT uniform on
    O_h: it reaches 32 of the 48 elements (the 8 sign flips with p = 1/16 each, the 24 elements
    whose axis permutation is a transposition with p = 1/48 each, never the 16 with a 3-cycle).

Deviations forced by Lightning 2.x / our setup:
    precision 16 -> "16-mixed"; training_epoch_end -> on_train_epoch_end; torch.cuda.amp.custom_fwd
    -> an explicit autocast-off block; deterministic=True -> "warn" (warn_only, because group-conv
    gathers have no deterministic CUDA backward); `max_time` is a per-process wall clock
    (Lightning's Timer restores elapsed time from the checkpoint, which would stop a resumed job at
    once); one best checkpoint (upstream kept top 3 plus every 10th epoch); augmentation draws come
    from a generator seeded by (seed, epoch, index), not the global numpy RNG (same distribution;
    upstream's numpy state was duplicated across DataLoader workers).

Run directory `<out>/<stem>/`, stem = run_name(...) = `{name}-{size}-r{ratio}-{loss}-{aug|noaug}-s{seed}`:
    metrics.jsonl                         one row per epoch (rank 0), appended across resumes
    {stem}.ckpt                           best val_mseloss (Lightning checkpoint)
    last.ckpt                             resume point; when max_time runs out, Lightning 2.6
                                          validates once, checkpoints, and fit returns
    {name}_approx{size}_{ratio}xSR.pt     plain state_dict of the best model, shipped naming, so
                                          blastnet.models.load_pretrained reads it
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.plugins.environments import LightningEnvironment
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "LOSSES",
    "UPSTREAM",
    "SRModule",
    "UpstreamAugment",
    "WallClock",
    "export_state_dict",
    "infer_hr_size",
    "parse_time",
    "run_name",
    "train",
    "upstream_augment",
]

LOSSES = ("mse", "mse_grad")

UPSTREAM = dict(lr=1e-4, batch_size=1, epochs=600, precision="16-mixed", lr_sched_factor=0.5,
                grad_loss_weight=0.99, num_workers=6, save_period=1, seed=42)
"""Upstream train.py defaults (the LSF sample script overrode: batch 4 x 16 GPUs, 1500 epochs,
save_period 5)."""


def run_name(name: str, size: str, ratio: int, loss: str, aug: bool, seed: int) -> str:
    """Checkpoint stem `{name}-{size}-r{ratio}-{loss}-{aug|noaug}-s{seed}`."""
    return f"{name}-{size}-r{ratio}-{loss}-{'aug' if aug else 'noaug'}-s{seed}"


def parse_time(t: str | float | None) -> float | None:
    """Seconds from a number, "HH:MM:SS", or upstream-style "DD:HH:MM:SS" (None passes through)."""
    if t is None or isinstance(t, int | float):
        return None if t is None else float(t)
    parts = [float(p) for p in str(t).split(":")]
    if len(parts) == 1:
        return parts[0]
    weights = (86400.0, 3600.0, 60.0, 1.0)[-len(parts):]
    return sum(w * p for w, p in zip(weights, parts))


def infer_hr_size(root: str | os.PathLike, split: str) -> int:
    """HR cube edge of a .dat tree, from the byte size of the split's first rho file."""
    from blastnet.data import cube_path, load_meta

    h = load_meta(root, split)["hash"].iat[0]
    n = round((os.path.getsize(cube_path(root, split, h, "RHO_kgm-3")) / 4) ** (1 / 3))
    return int(n)


# --- upstream augmentation (common/data.py random_rot90_3D + random_flip_3D, ported literally) ---


def _rot90_3d(x: torch.Tensor, k: int, dims: tuple[int, int]) -> torch.Tensor:
    # x [C, X, Y, Z] with C = (rho, u, v, w); channel d + 1 is the velocity along spatial axis d
    a, b = dims[0] + 1, dims[1] + 1
    for _ in range(k):
        x = torch.rot90(x, k=1, dims=(a, b))
        x[[a, b]] = x[[b, a]]
        x[a] = -x[a]
    return x


def _flip_3d(x: torch.Tensor, axis: int) -> torch.Tensor:
    x = torch.flip(x, dims=[axis + 1])
    x[axis + 1] = -x[axis + 1]
    return x


def upstream_augment(lr: torch.Tensor, hr: torch.Tensor, dx: torch.Tensor,
                     generator: torch.Generator | None = None):
    """Upstream train-split augmentation on one (lr, hr, dx[3]) sample; returns the same triple.

    k ~ U{0..3} quarter turns in the plane of 2 distinct random axes (ordered), then each axis is
    flipped with p = 1/2; vectors transform with the grid (u, v, w negated/swapped) and dx is
    swapped with the rotation, as upstream rot_dx.
    """
    k = int(torch.randint(4, (1,), generator=generator))
    dims = tuple(int(d) for d in torch.randperm(3, generator=generator)[:2])
    flips = torch.rand(3, generator=generator) > 0.5
    lr, hr, dx = _rot90_3d(lr, k, dims), _rot90_3d(hr, k, dims), dx.clone()
    for _ in range(k):
        dx[[dims[0], dims[1]]] = dx[[dims[1], dims[0]]]
    for axis in range(3):
        if flips[axis]:
            lr, hr = _flip_3d(lr, axis), _flip_3d(hr, axis)
    return lr, hr, dx


class UpstreamAugment(Dataset):
    """Wrap a Momentum128 (augment=False) with upstream_augment, seeded by (seed, epoch, index)."""

    def __init__(self, base: Dataset, seed: int = 0):
        self.base, self.seed, self.epoch = base, int(seed), 0

    def __len__(self) -> int:
        return len(self.base)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch mixed into the per-item seed (and forward it to the base dataset)."""
        self.epoch = int(epoch)
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    def __getitem__(self, i: int) -> dict:
        """The base item with lr, hr, dx transformed by one upstream augmentation draw."""
        item = dict(self.base[i])
        state = np.random.SeedSequence([self.seed, self.epoch, int(i), 1]).generate_state(2)
        gen = torch.Generator().manual_seed(int(state[0]) << 32 | int(state[1]))
        item["lr"], item["hr"], item["dx"] = upstream_augment(item["lr"], item["hr"], item["dx"],
                                                              gen)
        return item


# --- Lightning module and callbacks ---


class SRModule(pl.LightningModule):
    """Upstream LitModel training/validation for any [B,4,n/r,..] -> [B,4,n,..] SR net."""

    def __init__(self, model: nn.Module, loss: str = "mse", mean=None, std=None, lr: float = 1e-4,
                 lr_sched_factor: float = 0.5, grad_loss_weight: float = 0.99):
        super().__init__()
        if loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}, got {loss!r}")
        self.save_hyperparameters(ignore=["model", "mean", "std"])
        self.model = model
        self.loss_type = loss
        self.lr = lr
        self.lr_sched_factor = lr_sched_factor
        self.grad_loss_weight = grad_loss_weight
        self.gradloss = None
        if loss == "mse_grad":
            from blastnet.upstream import GradLoss

            self.gradloss = GradLoss()
        from blastnet.data import norm_stats

        m, s = norm_stats("train")
        self.register_buffer("mean", torch.as_tensor(m if mean is None else mean,
                                                     dtype=torch.float32), persistent=False)
        self.register_buffer("std", torch.as_tensor(s if std is None else std,
                                                    dtype=torch.float32), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The wrapped model's forward (normalised LR -> normalised HR)."""
        return self.model(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Upstream training loss, logged as train_{loss}loss (epoch mean, synced)."""
        x, y = batch["lr"], batch["hr"]
        bs = x.shape[0]
        y_hat = self.model(x)
        if self.loss_type == "mse":
            loss = F.mse_loss(y_hat, y)
        else:
            gradloss = self.grad_loss_weight * self.gradloss(y_hat, y)
            mseloss = (1.0 - self.grad_loss_weight) * F.mse_loss(y_hat, y)
            loss = mseloss + gradloss
            self.log("train_weightedgradloss", gradloss, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=bs)
            self.log("train_weightedmseloss", mseloss, on_step=False, on_epoch=True,
                     sync_dist=True, batch_size=bs)
        self.log(f"train_{self.loss_type}loss", loss, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=bs)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """val_mseloss in normalised units, float32 with autocast off (upstream custom_fwd)."""
        x, y = batch["lr"].float(), batch["hr"].float()
        with torch.autocast(device_type=self.device.type, enabled=False):
            y_hat = self.model(x).float()
            mseloss = F.mse_loss(y_hat, y)
        self.log("val_mseloss", mseloss, on_step=False, on_epoch=True, sync_dist=True,
                 batch_size=x.shape[0])

    def on_train_epoch_end(self) -> None:
        """Log the current learning rate (upstream training_epoch_end)."""
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", lr, on_step=False, on_epoch=True, sync_dist=False)

    def configure_optimizers(self):
        """Adam + StepLR(step_size=300 * lr_sched_factor, gamma=0.5), stepped per epoch."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(300 * self.lr_sched_factor), gamma=0.5)
        return [optimizer], [scheduler]


class WallClock(Callback):
    """Stop training cleanly once `seconds` of this process's wall time have elapsed."""

    def __init__(self, seconds: float):
        self.seconds = float(seconds)
        self.start = time.monotonic()

    def on_train_batch_end(self, trainer: pl.Trainer, *args, **kwargs) -> None:
        """Set trainer.should_stop on every rank when rank 0's clock has run out."""
        stop = trainer.strategy.broadcast(time.monotonic() - self.start >= self.seconds)
        if stop:
            trainer.should_stop = True


class _EpochSeed(Callback):
    # datasets mix the epoch into their per-item seeds; set it before the loader iterator exists
    def __init__(self, dataset):
        self.dataset = dataset

    def on_train_epoch_start(self, trainer, pl_module):
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(trainer.current_epoch)


class _MetricsLog(Callback):
    # rank 0 appends one JSON row per training epoch to metrics.jsonl
    def __init__(self, path: Path, loss: str):
        self.path, self.loss, self.start, self._val = Path(path), loss, time.monotonic(), None

    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking and "val_mseloss" in trainer.callback_metrics:
            self._val = float(trainer.callback_metrics["val_mseloss"])

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            self._val = None
            return
        m = trainer.callback_metrics
        row = {"epoch": trainer.current_epoch, "step": trainer.global_step}
        key = f"train_{self.loss}loss"
        row["train_loss"] = float(m[key]) if key in m else None
        row["val_mseloss"] = self._val  # None if this epoch ran no validation
        row["lr"] = trainer.optimizers[0].param_groups[0]["lr"]
        row["wall_s"] = round(time.monotonic() - self.start, 3)
        row["time"] = time.time()
        row.update({k: float(v) for k, v in m.items()
                    if k.startswith("train_") and k != key})
        self._val = None
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


def export_state_dict(ckpt: str | os.PathLike, out: str | os.PathLike) -> Path:
    """Write the plain model state_dict (keys without the `model.` prefix) of a Lightning ckpt."""
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    sd = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    out = Path(out)
    torch.save(sd, out)
    return out


def _precision(p) -> str:
    return {"16": "16-mixed", "32": "32-true", "bf16": "bf16-mixed"}.get(str(p), str(p))


def train(root: str | os.PathLike, name: str, size: str, ratio: int = 8, loss: str = "mse",
          aug: bool = True, seed: int = UPSTREAM["seed"], out: str | os.PathLike = "runs", *,
          epochs: int = UPSTREAM["epochs"], batch_size: int = UPSTREAM["batch_size"],
          lr: float = UPSTREAM["lr"], lr_sched_factor: float = UPSTREAM["lr_sched_factor"],
          grad_loss_weight: float = UPSTREAM["grad_loss_weight"],
          precision: str = UPSTREAM["precision"], accelerator: str = "auto",
          devices: int | str = "auto", strategy: str = "auto", num_nodes: int = 1,
          num_workers: int = UPSTREAM["num_workers"], crop: int | None = None,
          preload: bool = False, max_time: str | float | None = None, max_steps: int = -1,
          save_period: int = UPSTREAM["save_period"], accumulate_grad_batches: int = 1,
          hr_size: int | None = None, resume: bool = True,
          gradient_clip_val: float | None = None) -> Path:
    """Train `name` at `size` on Momentum128 at `root`; returns the run directory.

    Resumes from <run>/last.ckpt when present (and resume is True). max_time (seconds or
    [DD:]HH:MM:SS) stops this process cleanly, writing last.ckpt and the .pt export.
    gradient_clip_val (default None, as upstream) clips the global gradient norm; a clipped run
    gets a "-clip<v>" suffix on its run directory so it never resumes an unclipped one.
    """
    from blastnet import models
    from blastnet.data import Momentum128, norm_stats

    run = Path(out) / run_name(name, size, ratio, loss, aug, seed)
    if gradient_clip_val is not None:
        run = run.with_name(f"{run.name}-clip{gradient_clip_val:g}")
    run.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(seed, verbose=False)
    if hr_size is None:
        hr_size = infer_hr_size(root, "train")
    train_ds = Momentum128(root, "train", ratio, crop=crop, normalize=True, augment=False,
                           seed=seed, hr_size=hr_size, preload=preload)
    if aug:
        train_ds = UpstreamAugment(train_ds, seed=seed)
    val_ds = Momentum128(root, "val", ratio, normalize=True, hr_size=hr_size, preload=preload)
    loader_kw = dict(batch_size=batch_size, num_workers=num_workers,
                     pin_memory=torch.cuda.is_available())
    train_dl = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kw)

    mean, std = norm_stats("train")
    module = SRModule(models.build(name, size, ratio), loss=loss, mean=mean, std=std, lr=lr,
                      lr_sched_factor=lr_sched_factor, grad_loss_weight=grad_loss_weight)
    stem = run.name
    best = ModelCheckpoint(dirpath=run, filename=stem, monitor="val_mseloss", mode="min",
                           save_top_k=1, save_last=True, every_n_epochs=save_period,
                           save_on_train_epoch_end=False, auto_insert_metric_name=False)
    callbacks = [best, _EpochSeed(train_ds), _MetricsLog(run / "metrics.jsonl", loss)]
    seconds = parse_time(max_time)
    if seconds is not None:
        callbacks.append(WallClock(seconds))
    trainer = pl.Trainer(
        default_root_dir=run, logger=False, max_epochs=epochs, max_steps=max_steps,
        accelerator=accelerator, devices=devices, strategy=strategy, num_nodes=num_nodes,
        sync_batchnorm=True, precision=_precision(precision), deterministic="warn",
        accumulate_grad_batches=accumulate_grad_batches, plugins=[LightningEnvironment()],
        callbacks=callbacks, check_val_every_n_epoch=save_period, enable_progress_bar=False,
        enable_model_summary=False, gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm="norm")
    last = run / "last.ckpt"
    trainer.fit(module, train_dl, val_dl, ckpt_path=str(last) if resume and last.exists() else None)
    if trainer.is_global_zero:
        src = best.best_model_path or best.last_model_path
        if src:
            export_state_dict(src, run / f"{name}_approx{size}_{ratio}xSR.pt")
    return run
