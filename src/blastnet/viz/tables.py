"""viz.leaderboard (mean +- std over seeds) and viz.layer_curves."""

from __future__ import annotations

from . import _style as S

_FAMILY_ORDER = ("edsr", "rcan", "rrdb", "rrdbgrad", "convfno", "gconv", "gconvrc")


def aggregate(rows, metric: str, x: str = "params", by: str = "family", split: str = "test",
              ratio: int | None = None) -> dict:
    """{by-value: [dict(x, mean, std, n, size), ...]} sorted by x; std is the sample std (ddof=1)."""
    import numpy as np

    rows = [r for r in rows if r.get("split", split) == split and r.get(metric) is not None]
    ratios = {r.get("ratio") for r in rows}
    if ratio is not None:
        rows = [r for r in rows if r.get("ratio") == ratio]
    elif len(ratios) > 1:
        raise ValueError(f"rows span several ratios {sorted(ratios)}; pass ratio=")
    groups: dict = {}
    for r in rows:
        groups.setdefault((r[by], r.get("size", r[x])), []).append(r)
    out: dict = {}
    for (key, size), rs in groups.items():
        m = np.array([float(r[metric]) for r in rs])
        out.setdefault(key, []).append(dict(
            x=float(np.mean([float(r[x]) for r in rs])), mean=float(m.mean()),
            std=float(m.std(ddof=1)) if len(m) > 1 else 0.0, n=len(m), size=size))
    for v in out.values():
        v.sort(key=lambda d: d["x"])
    return out


def leaderboard(rows, metric: str, x: str = "params", by: str = "family",
                baselines: dict[str, float] | None = None, split: str = "test",
                theme: str = "light", ratio: int | None = None, colors: dict | None = None):
    """Mean +- sample std over seeds per (by, size); log-x; baselines as labelled hlines."""
    agg = aggregate(rows, metric, x, by, split, ratio)
    if not agg:
        raise ValueError(f"no rows with split={split!r} and metric {metric!r}")
    keys = sorted(agg, key=lambda k: (_FAMILY_ORDER.index(k) if k in _FAMILY_ORDER
                                      else len(_FAMILY_ORDER), str(k)))
    if colors is None and by == "family":
        pal = S.PALETTE[theme]
        known = [k for k in _FAMILY_ORDER if k in agg]  # stable colour per family
        extra = [k for k in agg if k not in _FAMILY_ORDER]
        colors = {k: pal[i] for i, k in enumerate(known + extra) if i < len(pal)}
    cmap = S.assign_colors(keys, colors, theme)
    with S.style(theme):
        t = S.THEMES[theme]
        fig, ax = S.new_figure(theme, size=(5.8, 3.8))
        items = []
        for k in keys:
            d = agg[k]
            xs = [g["x"] for g in d]
            ms = [g["mean"] for g in d]
            sd = [g["std"] for g in d]
            ln = S.plot_series(ax, xs, ms, cmap[k], str(k))
            ax.errorbar(xs, ms, yerr=sd, fmt="none", ecolor=cmap[k], elinewidth=1.0,
                        capsize=0, zorder=ln.get_zorder())
            items.append((str(k), xs[-1], ms[-1]))
        for name, val in (baselines or {}).items():
            ax.axhline(val, color=t["muted"], lw=1.2, zorder=1)
            ax.annotate(name, xy=(0.01, val), xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points", fontsize=8,
                        color=t["muted"], va="bottom")
        S.log_axis(ax, "x")
        ax.set_xlabel("parameters" if x == "params" else x)
        ax.set_ylabel(metric)
        if len(items) <= 4:
            S.end_labels(ax, items, theme)
        S.finish_legend(ax, len(items))
    return fig


def layer_curves(curves, ylabel: str, xlabel: str = "block", logy: bool = False,
                 ref: float | None = None, bands=None, theme: str = "light",
                 colors: dict | None = None):
    """One line per entry of `curves` over block index; optional shaded bands and ref line."""
    import numpy as np

    cmap = S.assign_colors(list(curves), colors, theme)
    with S.style(theme):
        t = S.THEMES[theme]
        fig, ax = S.new_figure(theme, size=(5.6, 3.6))
        items = []
        for k, v in curves.items():
            v = np.asarray(v, dtype=float)
            x = np.arange(len(v))
            if bands and k in bands:
                lo, hi = (np.asarray(b, dtype=float) for b in bands[k])
                ax.fill_between(x, lo, hi, color=cmap[k], alpha=0.18, lw=0)
            S.plot_series(ax, x, v, cmap[k], k)
            items.append((k, x[-1], v[-1]))
        if ref is not None:
            ax.axhline(ref, color=t["muted"], lw=1.2, zorder=1)
        if logy:
            S.log_axis(ax, "y")
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if len(items) <= 4:
            S.end_labels(ax, items, theme)
        S.finish_legend(ax, len(items))
    return fig
