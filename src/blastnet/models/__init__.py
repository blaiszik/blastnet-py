"""Super-resolution models: exact ports of the BLASTNet 2.0 SR benchmark nets, plus gconv.

The four upstream families (edsr, rcan, rrdb, convfno; `rrdbgrad` is the rrdb architecture trained
with `mse_grad`) are ported from github.com/blastnet/blastnet2_sr_benchmark (MIT), commit ae2a3a6,
with upstream module names and parameter shapes, so the shipped Momentum128 state_dicts load with
`strict=True` and outputs equal the reference bit for bit. Forward: [B, 4, n/r, n/r, n/r]
normalised (rho, u, v, w) -> [B, 4, n, n, n].

`sites(model)` names the modules an adapter hooks, in forward order:

    edsr     stem `head`        blocks `body.0` .. `body.31` (ResBlocks)              head `tail`
    rcan     stem `head`        blocks `body.<g>.body.<b>` (every RCAB, group-major)  head `tail`
    rrdb     stem `conv_first`  blocks `RRDB_trunk.0` .. (RRDB blocks)                head `conv_last`
    convfno  stem `FNOEncoder.fc0`  blocks `FNOEncoder.body.0` .. `.31` (FNOResLayers)
             head `UpsamplerDecoder`
    gconv    see `blastnet.models.gconv.gconv_sites`

RCAN blocks are the RCABs (the residual units that play the role of EDSR's ResBlocks; 1 for the
0.5M-2.7M nets, up to 200 for 50M); residual-group outputs are the modules `body.<g>`. Every `head`
output is the model output. ConvFNO quirks: `FNOEncoder.fc0` output is channels-last
[B, X, Y, Z, C], and block outputs are channels-first on the domain zero-padded by 6 on the high
side of each axis. Several upstream forwards update tensors in place after a module returns them
(e.g. `res += x` on an RCAB's CALayer output, in-place ReLUs on conv outputs); the site outputs
listed above are not modified afterwards, but hooks on other submodules should clone.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from torch import nn

from . import baselines, convfno, edsr, rcan, rrdb

__all__ = [
    "FAMILIES",
    "LABELS",
    "SIZES",
    "available",
    "build",
    "load_pretrained",
    "param_count",
    "sites",
]

LABELS = ("0.5M", "0.8M", "1.4M", "2.7M", "5M", "11M", "17M", "50M")
FAMILIES = ("edsr", "rcan", "rrdb", "rrdbgrad", "convfno", "gconv", "gconvrc")

# Upstream hyperparameters per family and size label (upstream init_* local names).
SIZES = {
    "edsr": edsr.SIZES,
    "rcan": rcan.SIZES,
    "rrdb": rrdb.SIZES,
    "rrdbgrad": rrdb.SIZES,
    "convfno": convfno.SIZES,
}

_INIT = {
    "edsr": edsr.init_edsr,
    "rcan": rcan.init_rcan,
    "rrdb": rrdb.init_rrdb,
    "rrdbgrad": rrdb.init_rrdb,
    "convfno": convfno.init_convfno,
}

# Shipped Momentum128 weights (per seed 42/43/44): every label at 8x, plus 16x and 32x at the
# largest label of each family. No convfno or gconv weights are shipped.
_LARGEST = {"edsr": "50M", "rcan": "17M", "rrdb": "50M", "rrdbgrad": "50M"}

_FILENAME = re.compile(r"^(?P<name>[a-z]+)_approx(?P<size>[0-9.]+M)_(?P<ratio>\d+)xSR\.pt$")
_SEED_DIR = re.compile(r"^seed(\d+)$")


def build(name: str, size: str | None, ratio: int = 8) -> nn.Module:
    """Build a randomly initialised model (upstream init) by family name, size label, and ratio.

    `nearest`, `trilinear`, `tricubic` are parameter-free interpolation baselines (`size=None`).
    """
    if name in baselines.MODES:
        return baselines.Interp(ratio, name)
    if name in ("gconv", "gconvrc"):  # gconvrc: resize-convolution upsampler (see gconv docstring)
        from .gconv import build_gconv

        return build_gconv(size, ratio, upsampler="resize" if name == "gconvrc" else "subpixel")
    if name not in _INIT:
        raise ValueError(f"unknown model {name!r}; expected one of {FAMILIES}")
    if size not in SIZES[name]:
        raise ValueError(f"{name} size must be one of {list(SIZES[name])}, got {size!r}")
    return _INIT[name](size, ratio)


def load_pretrained(path: str | Path, map_location: str | torch.device = "cpu") -> nn.Module:
    """Build from a `{name}_approx{size}_{r}xSR.pt` filename, load its state_dict strictly, eval()."""
    path = Path(path)
    m = _FILENAME.match(path.name)
    if m is None:
        raise ValueError(f"cannot parse {path.name!r}; expected {{name}}_approx{{size}}_{{r}}xSR.pt")
    name, size, ratio = m["name"], m["size"], int(m["ratio"])
    seed_match = _SEED_DIR.match(path.parent.name)
    model = build(name, size, ratio)
    state = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state, strict=True)
    if isinstance(map_location, str | torch.device):
        model.to(map_location)
    model.eval()
    model.blastnet_meta = dict(name=name, size=size, ratio=ratio,
                               seed=int(seed_match[1]) if seed_match else None, path=str(path))
    return model


def sites(model: nn.Module) -> dict:
    """Qualified module names (stem, blocks in forward order, head) for hooking `model`."""
    if isinstance(model, edsr.EDSR):
        blocks = [f"body.{i}" for i, m in enumerate(model.body) if isinstance(m, edsr.ResBlock)]
        return dict(stem="head", blocks=blocks, head="tail")
    if isinstance(model, rcan.RCAN):
        blocks = [f"body.{g}.body.{b}"
                  for g, group in enumerate(model.body) if isinstance(group, rcan.ResidualGroup)
                  for b, m in enumerate(group.body) if isinstance(m, rcan.RCAB)]
        return dict(stem="head", blocks=blocks, head="tail")
    if isinstance(model, rrdb.RRDBNet):
        blocks = [f"RRDB_trunk.{i}" for i in range(len(model.RRDB_trunk))]
        return dict(stem="conv_first", blocks=blocks, head="conv_last")
    if isinstance(model, convfno.ConvFNO):
        blocks = [f"FNOEncoder.body.{i}" for i in range(len(model.FNOEncoder.body))]
        return dict(stem="FNOEncoder.fc0", blocks=blocks, head="UpsamplerDecoder")
    if isinstance(model, baselines.Interp):
        return dict(stem=None, blocks=[], head=None)
    from .gconv import GConvSR, gconv_sites

    if isinstance(model, GConvSR):
        return gconv_sites(model)
    raise TypeError(f"no sites for {type(model).__name__}")


def param_count(model: nn.Module) -> int:
    """Number of trainable scalars (complex FNO weights count twice, as stored)."""
    return sum(p.numel() for p in model.parameters())


def available() -> list[tuple[str, str, int]]:
    """The shipped pretrained inventory per seed, as (name, size, ratio); 40 entries."""
    out = [(name, size, 8) for name in _LARGEST for size in LABELS]
    out += [(name, _LARGEST[name], r) for name in _LARGEST for r in (16, 32)]
    return out
