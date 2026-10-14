"""Render every blastnet.viz figure from the laptop subset (shipped edsr 0.8M + tricubic), both themes.

    python docs/viz_gallery.py OUT_DIR      # writes OUT_DIR/{light,dark}/<figure>.{png,svg}
"""
import sys
from pathlib import Path

import torch

from blastnet import physics, viz
from blastnet.data import Momentum128, norm_stats
from blastnet.models import build, load_pretrained, param_count

ROOT = Path("data/momentum128_subset")
OUT = Path(sys.argv[1])
CROP, RATIO, N_CUBES = 96, 8, 4
torch.set_grad_enabled(False)

mean, std = norm_stats("test")
den = lambda y: y * std[None, :, None, None, None] + mean[None, :, None, None, None]  # noqa: E731
ds = Momentum128(ROOT, "test", RATIO, crop=CROP, seed=3)
items = [ds[i] for i in range(min(N_CUBES, len(ds)))]
lr = torch.stack([it["lr"] for it in items])
hr = torch.stack([it["hr"] for it in items])
dx = torch.stack([it["dx"] for it in items])

edsr = load_pretrained(ROOT / "best_model_weights/seed42/edsr_approx0.8M_8xSR.pt")
tricubic = build("tricubic", None, RATIO)
sr_edsr = torch.cat([edsr(lr[i:i + 1]) for i in range(len(items))])
sr_tri = torch.cat([tricubic(lr[i:i + 1]) for i in range(len(items))])
y, yh, yt = den(hr), den(sr_edsr), den(sr_tri)
xl = den(lr)  # LR in physical units (nearest-upsampled for display by viz.slices)

# leaderboard rows: per-crop metrics of the shipped edsr/rcan seeds 42, 43 and the tricubic baseline
def metric_row(model, family, size, seed):
    m = {k: 0.0 for k in ("test_ssim_sgs", "spectrum_error")}
    for i in range(len(items)):
        r = physics.metrics(model(lr[i:i + 1]), hr[i:i + 1], RATIO, dx[i:i + 1], mean, std)
        for k in m:
            m[k] += float(r[k]) / len(items)
    return dict(family=family, size=size, ratio=RATIO, seed=seed, split="test",
                params=param_count(model), **m)


rows = []
for fam in ("edsr", "rcan"):
    for size in ("0.5M", "0.8M", "1.4M"):
        for seed in (42, 43):
            p = ROOT / f"best_model_weights/seed{seed}/{fam}_approx{size}_{RATIO}xSR.pt"
            if p.exists():
                rows.append(metric_row(load_pretrained(p), fam, size, seed))
base = metric_row(tricubic, "tricubic", "-", 0)

tau_hat, tau = physics.sgs_stress(yh, RATIO), physics.sgs_stress(y, RATIO)
fields = {"LR input": xl[0], "tricubic": yt[0], "EDSR 0.8M": yh[0], "DNS truth": y[0]}
vel = {"DNS truth": y, "EDSR 0.8M": yh, "tricubic": yt}
# dense curves: per-block profile of a simple diagnostic (relative energy error vs cube index)
curves = {"EDSR 0.8M": [float(physics.spectrum_error(yh[i:i + 1, 1:], y[i:i + 1, 1:], dx[i:i + 1]))
                        for i in range(len(items))],
          "tricubic": [float(physics.spectrum_error(yt[i:i + 1, 1:], y[i:i + 1, 1:], dx[i:i + 1]))
                       for i in range(len(items))]}

for theme in ("light", "dark"):
    out = OUT / theme
    figs = {
        "slices": viz.slices(fields, channel=1, residual=("EDSR 0.8M", "DNS truth"), theme=theme),
        "spectra": viz.spectra(vel, dx, reference="DNS truth", theme=theme),
        "spectra_ratio": viz.spectra_ratio(vel, dx, "DNS truth", theme=theme),
        "qr": viz.qr(vel, dx, theme=theme),
        "parity": viz.parity(tau_hat, tau, theme=theme),
        "leaderboard": viz.leaderboard(rows, "test_ssim_sgs",
                                       baselines={"tricubic": base["test_ssim_sgs"]}, theme=theme),
        "layer_curves": viz.layer_curves(curves, "spectrum error (mean |ln E ratio|)",
                                         xlabel="cube", theme=theme),
    }
    for name, fig in figs.items():
        print(*viz.save(fig, out / name), sep="\n")
