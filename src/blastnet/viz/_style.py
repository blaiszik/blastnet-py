"""Theme, palette, colormaps and small shared helpers. matplotlib is imported lazily."""

from __future__ import annotations

import contextlib
import itertools
from pathlib import Path

__all__ = ["PALETTE", "THEMES", "save", "style", "use_style"]

PALETTE = {
    "light": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
              "#e34948"),
    "dark": ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9",
             "#e66767"),
}
THEMES = {
    "light": dict(bg="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#8a8984", grid="#ecebe7"),
    "dark": dict(bg="#1a1a19", text="#f0efec", text2="#c3c2b7", muted="#8f8e86", grid="#262624"),
}
DIV = ("#2a78d6", "#f0efec", "#e34948")
SEQ = ("#cde2fb", "#0d366b")
LINE_WIDTH = 1.6


def _check(theme: str) -> dict:
    if theme not in THEMES:
        raise ValueError(f"theme must be one of {tuple(THEMES)}, got {theme!r}")
    return THEMES[theme]


def _register() -> None:
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap

    # dark theme: the low end of the sequential ramp must recede toward the dark surface, so reverse it
    for name, cols in (("blastnet_div", DIV), ("blastnet_seq", SEQ), ("blastnet_seq_dark", tuple(reversed(SEQ)))):
        if name in mpl.colormaps:
            continue
        mpl.colormaps.register(LinearSegmentedColormap.from_list(name, cols, N=256),
                               name=name, force=True)


def _rc(theme: str) -> dict:
    t = _check(theme)
    from cycler import cycler

    return {
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"],
        "savefig.edgecolor": t["bg"], "text.color": t["text"], "axes.titlecolor": t["text"],
        "axes.labelcolor": t["text2"], "axes.edgecolor": t["muted"], "axes.linewidth": 0.8,
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "xtick.labelcolor": t["text2"], "ytick.labelcolor": t["text2"],
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": t["grid"], "grid.linestyle": "-", "grid.linewidth": 0.8,
        "lines.linewidth": LINE_WIDTH, "lines.markersize": 4.5,
        "axes.prop_cycle": cycler(color=list(PALETTE[theme])),
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.default": "regular", "font.size": 10, "axes.titlesize": 10,
        "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False,
        "legend.fontsize": 9, "legend.labelcolor": t["text2"],
        "image.cmap": "blastnet_seq", "image.interpolation": "nearest", "figure.dpi": 100,
        "savefig.dpi": 200, "figure.constrained_layout.use": False,
    }


@contextlib.contextmanager
def style(theme: str = "light"):
    """Context manager applying the blastnet theme ("light" or "dark") to matplotlib rcParams."""
    import matplotlib as mpl

    _register()
    with mpl.rc_context(_rc(theme)):
        yield


def use_style(theme: str = "light") -> None:
    """Apply the blastnet theme globally (until changed or mpl.rcdefaults())."""
    import matplotlib as mpl

    _register()
    mpl.rcParams.update(_rc(theme))


def save(fig, path, formats=("png", "svg"), dpi: int = 200) -> list[Path]:
    """Save `fig` as `path` with each of `formats` as suffix (a given suffix is replaced)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for fmt in formats:
        p = path.with_suffix("." + fmt.lstrip("."))
        fig.savefig(p, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
        out.append(p)
    return out


# ------------------------------------------------------------------ shared helpers

def to_numpy(x):
    import numpy as np

    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def assign_colors(keys, colors: dict | None, theme: str) -> dict:
    """key -> colour: explicit `colors` first, others take PALETTE slots in insertion order."""
    _check(theme)
    pal = PALETTE[theme]
    colors = dict(colors or {})
    out, n = {}, 0
    for k in keys:
        if k in colors:
            out[k] = colors[k]
        else:
            if n >= len(pal):
                raise ValueError("at most 8 series: fold the tail into 'other' or use panels")
            out[k] = pal[n]
            n += 1
    return out


def seq_cmap(theme: str) -> str:
    """Name of the sequential colormap whose low end recedes toward this theme's surface."""
    _check(theme)
    return "blastnet_seq_dark" if theme == "dark" else "blastnet_seq"


def new_figure(theme: str, nrows: int = 1, ncols: int = 1, size=(5.2, 3.6), **kw):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(nrows, ncols, figsize=size, layout="constrained", **kw)
    fig.patch.set_facecolor(THEMES[theme]["bg"])
    return fig, axs


def log_axis(ax, which: str = "y") -> None:
    from matplotlib.ticker import LogFormatterSciNotation

    getattr(ax, f"set_{which}scale")("log")
    axis = getattr(ax, f"{which}axis")
    axis.set_major_formatter(LogFormatterSciNotation(base=10))
    axis.set_minor_formatter(LogFormatterSciNotation(base=10, minor_thresholds=(1.5, 0.4)))


def plot_series(ax, x, y, color, label, n_markers: int = 12, **kw):
    """One 1.6 pt line; markers only when the series has <= 12 points."""
    dense = len(x) > n_markers
    return ax.plot(x, y, color=color, label=label, lw=LINE_WIDTH, solid_capstyle="round",
                   marker=None if dense else "o", ms=4.0, **kw)[0]


def end_labels(ax, items, theme: str, min_gap: float = 0.055) -> None:
    """Direct labels (secondary ink) at the line ends; items: [(label, x_end, y_end)]."""
    import numpy as np

    t = THEMES[theme]
    ax.autoscale_view()
    ax.get_ylim()
    inv = ax.transAxes.inverted()
    ys = np.array([inv.transform(ax.transData.transform((x, y)))[1] for _, x, y in items])
    order = np.argsort(ys)
    pos = ys.copy()
    for a, b in itertools.pairwise(order):  # push up, then recentre, to keep min_gap
        pos[b] = max(pos[b], pos[a] + min_gap)
    pos -= pos.mean() - ys.mean()
    pos = np.clip(pos, 0.03, 0.97)
    for (label, x, _), yp in zip(items, pos):
        ax.annotate(label, xy=(x, yp), xycoords=("data", "axes fraction"), xytext=(5, 0),
                    textcoords="offset points", va="center", ha="left", fontsize=9,
                    color=t["text2"], annotation_clip=False)


def finish_legend(ax, n: int) -> None:
    if n >= 2:
        ax.legend(loc="best", handlelength=1.6)
