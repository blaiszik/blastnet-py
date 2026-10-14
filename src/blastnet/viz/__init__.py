"""Publication-style figures for BLASTNet (matplotlib is imported lazily; `pip install blastnet[viz]`).

    from blastnet import viz
    fig = viz.slices({"truth": hr, "edsr": sr}, channel=1, residual=("edsr", "truth"))
    viz.save(fig, "slices")             # slices.png + slices.svg
    with viz.style(theme="dark"): ...   # or viz.use_style("dark"), or theme= on every figure

Figures: slices, spectra, spectra_ratio, qr, parity, leaderboard, layer_curves.
Colours follow the entity (explicit `colors=` mapping or insertion order), PALETTE has 8 slots.
"""

from blastnet.viz._spectra import spectra, spectra_ratio
from blastnet.viz._style import PALETTE, THEMES, save, style, use_style
from blastnet.viz.fields import slices
from blastnet.viz.gradients import parity, qr, vieillefosse
from blastnet.viz.tables import aggregate, layer_curves, leaderboard

__all__ = [
    "PALETTE",
    "THEMES",
    "aggregate",
    "layer_curves",
    "leaderboard",
    "parity",
    "qr",
    "save",
    "slices",
    "spectra",
    "spectra_ratio",
    "style",
    "use_style",
    "vieillefosse",
]
