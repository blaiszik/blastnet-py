"""viz.slices: side-by-side cross-sections of scalar fields on one shared diverging scale."""

from __future__ import annotations

from . import _style as S

_UNITS = {
    0: r"$\rho$ (kg m$^{-3}$)", 1: r"$u$ (m s$^{-1}$)", 2: r"$v$ (m s$^{-1}$)",
    3: r"$w$ (m s$^{-1}$)",
}


def _plane(a, channel, axis, index, target):
    import numpy as np

    a = S.to_numpy(a)
    if a.ndim == 4:
        if channel is None:
            raise ValueError("4-d field [C, X, Y, Z] needs an integer `channel`")
        a = a[channel]
    if a.ndim != 3:
        raise ValueError(f"field must be [C, X, Y, Z] or [X, Y, Z], got ndim {a.ndim}")
    s = a.shape
    li = min(int(index * s[axis] // target[axis]), s[axis] - 1)
    p = np.take(a, li, axis=axis)
    rest = [i for i in range(3) if i != axis]
    ia, ib = (np.arange(target[i]) * s[i] // target[i] for i in rest)  # nearest upsampling
    return p[np.ix_(ia, ib)]


def slices(fields, channel: int | None = 1, axis: int = 2, index: int | None = None,
           residual: tuple[str, str] | None = None, shared_scale: bool = True,
           theme: str = "light"):
    """One panel per field (+ optional residual `a - b` panel with its own symmetric scale).

    fields: label -> [C, X, Y, Z] or [X, Y, Z]; coarser grids are nearest-upsampled to the
    largest. Signed fields use blastnet_div centred on the (shared) median with symmetric limits
    from the 99th percentile of |x - median|; the residual is centred on 0.
    """
    import numpy as np

    if not fields:
        raise ValueError("fields is empty")
    if residual is not None and not set(residual) <= set(fields):
        raise ValueError(f"residual {residual} must name keys of fields")
    shapes = []
    for a in fields.values():
        sh = S.to_numpy(a).shape
        shapes.append(sh[-3:])
    target = tuple(max(sh[i] for sh in shapes) for i in range(3))
    if index is None:
        index = target[axis] // 2
    planes = {k: _plane(v, channel, axis, index, target) for k, v in fields.items()}
    unit = _UNITS.get(channel, "value") if channel is not None else "value"
    names = "xyz"
    rest = [names[i] for i in range(3) if i != axis]

    def limits(arrs):
        v = np.concatenate([x.ravel() for x in arrs]).astype(float)
        c = float(np.median(v))
        a = float(np.percentile(np.abs(v - c), 99))
        return c, (a if a > 0 else 1.0)

    n = len(planes)
    cols, ratios = [], []
    for _ in range(n):
        cols.append("ax")
        ratios.append(1.0)
        if not shared_scale:
            cols.append("cb")
            ratios.append(0.05)
    if shared_scale:
        cols.append("cb")
        ratios.append(0.05)
    if residual:
        cols += ["res", "rescb"]
        ratios += [1.0, 0.05]
    import matplotlib.pyplot as plt

    with S.style(theme):
        t = S.THEMES[theme]
        fig = plt.figure(figsize=(2.75 * (n + bool(residual)) + 0.9, 3.35), layout="constrained")
        fig.patch.set_facecolor(t["bg"])
        gs = fig.add_gridspec(1, len(cols), width_ratios=ratios)
        axes = [fig.add_subplot(gs[0, i]) for i in range(len(cols))]
        shared = limits(planes.values()) if shared_scale else None
        fi = 0
        im_last = None
        for ax, kind in zip(axes, cols):
            if kind == "ax":
                label, img = list(planes.items())[fi]
                fi += 1
                c, a = shared or limits([img])
                im_last = ax.imshow(img.T, origin="lower", cmap="blastnet_div", vmin=c - a,
                                    vmax=c + a, interpolation="nearest")
                ax.set_title(label)
                ax.set_label(f"field:{label}")
                if not shared_scale:
                    cax = axes[cols.index("cb", axes.index(ax))]
                    fig.colorbar(im_last, cax=cax).set_label(unit)
            elif kind == "res":
                a_l, b_l = residual
                res = planes[a_l] - planes[b_l]
                r = float(np.percentile(np.abs(res), 99))
                r = r if r > 0 else 1.0
                im = ax.imshow(res.T, origin="lower", cmap="blastnet_div", vmin=-r, vmax=r,
                               interpolation="nearest")
                ax.set_title(f"{a_l} − {b_l}")
                ax.set_label("residual")
                fig.colorbar(im, cax=axes[-1]).set_label("residual, " + unit)
            if kind in ("ax", "res"):
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.set_aspect("equal")
                ax.set_xlabel(rest[0])
        if shared_scale:
            fig.colorbar(im_last, cax=axes[cols.index("cb")]).set_label(unit)
        axes[0].set_ylabel(rest[1])
        for ax, kind in zip(axes, cols):
            if kind in ("cb", "rescb"):
                ax.set_label("colorbar")
                ax.grid(False)
                ax.tick_params(length=2)
                for sp in ax.spines.values():
                    sp.set_visible(False)
        fig.suptitle(f"plane {names[axis]} = {index} of {target[axis]}", x=0.0, ha="left",
                     fontsize=8, color=t["muted"])
    return fig
