# blastnet

A pip-installable PyTorch data layer, physics metrics, and super-resolution models for
[BLASTNet](https://www.kaggle.com/datasets/waitongchung/blastnet-momentum-3d-sr-dataset) 3D
turbulent-combustion data. It reads the Momentum128 super-resolution dataset as shipped (memmapped
`.dat` cubes, aligned low/high-resolution crops, upstream normalisation), reads any BLASTNet case
described by an `info.json`, loads the 120 pretrained Momentum128 checkpoints with exact ports of the
upstream architectures, reproduces the published evaluation metrics, and adds the exact 48-element
cube symmetry group plus an O<sub>h</sub>-equivariant model (`gconv`). Architectures and metrics are
ported from the MIT-licensed benchmark code at
[blastnet/blastnet2_sr_benchmark](https://github.com/blastnet/blastnet2_sr_benchmark).

## Install

```bash
uv add blastnet                  # or: pip install blastnet
uv add "blastnet[kaggle]"        # adds kagglehub, for downloading from Kaggle
```

From a checkout: `uv sync` (laptop, CPU). Python 3.12 or newer.

## Quickstart

The snippets below were executed against real Momentum128 cubes and shipped checkpoints (CPU).
`ROOT` is a Momentum128 directory (layout below); a small subset with the same layout works too.

```python
from blastnet.data import Momentum128, load_meta

ROOT = "data/momentum128_subset"
ds = Momentum128(ROOT, split="test", ratio=8, crop=96)  # random aligned 96^3 crop of a 128^3 cube
item = ds[0]
print(item["lr"].shape, item["hr"].shape)      # channels (rho, u, v, w), upstream-normalised
print("dx [m]:", item["dx"].tolist())
print(load_meta(ROOT, "test")[["kaggle_data_id", "dx", "dy", "dz"]].head(3).to_string())
```

```python
from blastnet.models import load_pretrained, sites, param_count

net = load_pretrained(f"{ROOT}/best_model_weights/seed42/edsr_approx0.8M_8xSR.pt")  # strict load
s = sites(net)
print(net.blastnet_meta["name"], net.blastnet_meta["size"], f"{param_count(net):,} params")
print("stem:", s["stem"], "| blocks:", s["blocks"][0], "...", s["blocks"][-1], "| head:", s["head"])
```

```python
import torch
from blastnet import physics
from blastnet.data import norm_stats

mean, std = norm_stats("test")
with torch.no_grad():
    y_hat = net(item["lr"][None])                 # [1, 4, 96, 96, 96]
m = physics.metrics(y_hat, item["hr"][None], ratio=8, dx=item["dx"][None], mean=mean, std=std)
print({k: round(m[k], 4) for k in ("test_ssim_ux", "test_ssim_sgs", "spectrum_error")})
```

The cube symmetry group acts exactly (flips and transposes on the grid and on the velocity vector,
no interpolation):

```python
from blastnet import octahedral as O

g = O.elements()[17]
print(g.perm, g.signs, "det", g.det)
x = item["lr"][None]
assert torch.equal(O.apply(O.apply(x, g), O.inverse(g)), x)   # exact: flips + transposes
unseen = [e for e in O.elements() if all(p != i for i, p in enumerate(e.perm))]  # 3-cycles
print(len(unseen), "elements upstream augmentation never applies, e.g.", unseen[0].perm, unseen[0].signs)
```

## Data layouts

**Momentum128 as shipped on Kaggle.** Raw little-endian float32 cubes plus metadata CSVs:

```
<root>/dataset/HR/<split>/<VAR>_id<hash>.dat            128^3, C order (x, y, z)
<root>/dataset/LR_{8,16,32}x/<split>/<VAR>_id<hash>.dat (128/r)^3
<root>/<split>_data_summary.csv, <root>/all_data_summary.csv
<root>/best_model_weights/seed{42,43,44}/*.pt
```

Splits are `train`, `val`, `test`, `forcedhit`, `paramvar`; variables are `RHO_kgm-3`, `UX_ms-1`,
`UY_ms-1`, `UZ_ms-1`, and the channel order is (rho, u, v, w) everywhere. `Momentum128` is the
PyTorch dataset (items are `{"lr", "hr", "dx", "index", "hash"}`, with optional random crops and
octahedral augmentation); `Momentum128Zarr` and `zarr_convert` give a chunked Zarr copy for fast
training; `load_meta`, `norm_stats`, `read_cube`, and `subset` are the lower-level helpers.

**General BLASTNet cases.** Other BLASTNet datasets (for example the `fhitcase0` and `liftedh2full`
tutorials) describe themselves with an `info.json`:

```
<case>/info.json     global: dataset_id, Nxyz, snapshots, variables, grid, bc; local: per-snapshot files
<case>/data/*.dat    float32, C order (i, j, k) = (x, y, z)
<case>/grid/*.dat    one full Nx*Ny*Nz float32 array per axis
```

`open_case` reads this without loading any data; fields and grid coordinates are memmapped and sliced,
so multi-gigabyte domains are safe to open. `CubeDataset` tiles (or randomly samples) training cubes
from one or more cases and can produce matched low-resolution inputs with the Favre/box filter.

```python
from blastnet.data import CubeDataset, open_case

case = open_case("path/to/fhitcase0")
print(case.shape, case.variables, case.check())       # check() lists files that are missing, mis-sized, or mislabelled
ds = CubeDataset([case], size=64, ratio=8)             # default variables: rho, u, v, w
item = ds[0]                                           # {"hr", "lr", "dx", "case", "snapshot", "origin"}
```

Some `info.json` files carry mislabelled filename fields (see
[AI-ModCon/dsagt](https://github.com/AI-ModCon/dsagt)); `Case.check()` reports them.

## Models

```python
from blastnet.models import available, build, load_pretrained

net = build("edsr", "0.8M", ratio=8)           # random init, upstream hyperparameters
net = load_pretrained(path_to_pt)              # {name}_approx{size}_{r}xSR.pt, strict state_dict load, eval()
interp = build("trilinear", None, ratio=8)     # parameter-free baselines: nearest, trilinear, tricubic
equiv = build("gconv", "0.8M", ratio=8)        # exactly O_h-equivariant net
len(available())                               # 40 shipped (family, size, ratio) entries per seed
```

Families are `edsr`, `rcan`, `rrdb`, `rrdbgrad` (the RRDB architecture trained with the gradient loss),
`convfno`, and `gconv`; size labels run from `0.5M` to `50M`. Shipped weights cover every label at 8x
plus 16x and 32x at the largest label per family, for seeds 42, 43, and 44 (no `convfno` or `gconv`
weights are shipped). Upstream-family ports use upstream module names and parameter shapes, so the
state_dicts load with `strict=True` and outputs equal the reference.

## Metrics

`blastnet.upstream` holds the published BLASTNet 2.0 metrics ported bit for bit (SGS stress and its
divergence, 3D SSIM, MSEs), quirks included, so numbers are comparable with the published table.
`blastnet.physics` adds differentiable physics on top: velocity gradients, strain invariants,
Q-criterion, enstrophy, dissipation, energy spectra and spectrum error, SGS stress, and
`physics.metrics`, which returns the upstream keys plus `spectrum_error` and `dissipation_error`.

## Visualization

`pip install "blastnet[viz]"` adds `blastnet.viz` (matplotlib only, imported lazily): the standard
figures for 3D SR of turbulence in one colour-blind-checked style, with `theme="light"` or `"dark"`.

```python
from blastnet import viz

fig = viz.slices({"LR input": lr, "EDSR 0.8M": y_hat, "DNS truth": y}, channel=1,
                 residual=("EDSR 0.8M", "DNS truth"))        # [C, X, Y, Z] fields, physical units
viz.spectra(vel, dx, reference="DNS truth")                  # vel: label -> [B, 3, X, Y, Z]
viz.spectra_ratio(vel, dx, "DNS truth")                      # E(k) / E_ref(k)
viz.qr(vel, dx)                                              # joint (R*, Q*) PDF + Vieillefosse line
viz.parity(tau_hat, tau)                                     # predicted vs true SGS components
viz.leaderboard(rows, "test_ssim_sgs")                       # eval.py rows; mean +- std over seeds
viz.layer_curves({"trained": r2, "random init": r2_rand}, "held-out R^2")
viz.save(fig, "figures/slices")                              # -> .png and .svg
```

`docs/viz_gallery.py` renders every figure from the Momentum128 subset in both themes.

## Command line

```bash
blastnet convert ROOT OUT [--ratios 8 16 32]          # .dat tree to Zarr
blastnet train --root DATA --name edsr --size 0.8M --out RUNS
blastnet eval WEIGHTS.pt [...] --root DATA --out rows.jsonl
blastnet eval-shipped --root DATA --out rows.jsonl    # every shipped checkpoint
blastnet leaderboard rows.jsonl [...] --out table.md
blastnet download ...                                 # fetch data from Kaggle (needs the [kaggle] extra)
blastnet catalog ...                                  # list known BLASTNet datasets
```

`blastnet <command> --help` lists every option. `download` and `catalog` are added alongside this
README; check `--help` for their final flags.

## Polaris and Sophia (ALCF)

`jobs/polaris/` holds the PBS jobs used for training and evaluation on Polaris (4x A100 per node).
`source jobs/polaris/env.sh` loads the ALCF conda module, creates a venv with the system site
packages, installs `blastnet` editable, and sets `BLASTNET_DATA`, `BLASTNET_RUNS`,
and cache directories on the project filesystem (edit `BLASTNET_ROOT` at the top, and `#PBS -A YOUR_PROJECT` in each `.pbs`, for your allocation).
`common.sh` is sourced by every `.pbs` file (NCCL settings, proxy, working directory); `train.pbs`,
`eval_shipped.pbs`, and `run_python.pbs` take their parameters through `qsub -v`. Submit from the
repository root. There are no Sophia-specific scripts; on Sophia, adapt the
module line in `env.sh` and the PBS `select`/`system` header of the `.pbs` files.

## Related work

[aPrioriDNS](https://pypi.org/project/aPrioriDNS/) (GPL-3.0; numpy, cantera, scipy) reads the
`info.json`/`.dat` format and does a-priori filtering for reacting DNS, without PyTorch.
[ORNL/MATEY](https://github.com/ORNL/MATEY) embeds the BLASTNet 3D data in a foundation-model
codebase, reading pre-converted HDF5.
[AI-ModCon/dsagt](https://github.com/AI-ModCon/dsagt) converts BLASTNet to The Well's HDF5 format.
[blastnet/blastnet2_sr_benchmark](https://github.com/blastnet/blastnet2_sr_benchmark) is the
research code behind the published benchmark and the shipped checkpoints.
[the_well](https://github.com/PolymathicAI/the_well) is a PyTorch collection of physics-simulation
datasets and loaders. This package is a packaged, PyTorch-native reader plus the upstream models and
metrics under one MIT-licensed install.

## Licences

Code: MIT. The Momentum128 data and the shipped model weights are CC BY-NC-SA 4.0 (non-commercial,
share-alike); check the licence of any other BLASTNet dataset you use.

## Citation

Cite the dataset and the papers listed on the
[dataset page](https://www.kaggle.com/datasets/waitongchung/blastnet-momentum-3d-sr-dataset):

- Dataset: doi:[10.5281/zenodo.7242864](https://doi.org/10.5281/zenodo.7242864)
- Applied Energy and Combustion Science paper: https://doi.org/10.1016/j.jaecs.2022.100087
- ICML workshop paper: https://openreview.net/forum?id=LxGTZM7L6qn
- W. T. Chung, B. Akoush, P. Sharma, A. Tamkin, K. S. Jung, J. H. Chen, J. Guo, D. Brouzet,
  M. Talei, B. Savard, A. Y. Poludnenko, M. Ihme, "Turbulence in Focus: Benchmarking Scaling
  Behavior of 3D Volumetric Super-Resolution with BLASTNet 2.0 Data", Advances in Neural
  Information Processing Systems 36 (2023), Datasets and Benchmarks Track,
  [arXiv:2309.13457](https://arxiv.org/abs/2309.13457).
