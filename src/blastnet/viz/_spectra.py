"""viz.spectra and viz.spectra_ratio: shell-binned kinetic-energy spectra (physics.energy_spectrum)."""

from __future__ import annotations

from . import _style as S


def _spec(f, dx):
    import torch

    from blastnet import physics

    t = torch.as_tensor(S.to_numpy(f)) if not hasattr(f, "dim") else f
    if t.dim() != 5 or t.shape[1] not in (3, 4):
        raise ValueError(f"velocity must be [B, 3, X, Y, Z] or [B, 4, ...], got {tuple(t.shape)}")
    u = t[:, 1:4] if t.shape[1] == 4 else t
    k, E = physics.energy_spectrum(u, dx)
    gm = E.clamp_min(torch.finfo(E.dtype).tiny).log().mean(0).exp()  # geometric mean over B
    return k.mean(0).double().numpy(), gm.double().numpy()


def _xaxis(dx, xunit):
    import torch

    if xunit == "k_dx":
        return float(torch.as_tensor(dx, dtype=torch.float64).mean()), r"$k\,\Delta x$"
    if xunit == "k":
        return 1.0, r"$k$ (rad m$^{-1}$)"
    raise ValueError(f"xunit must be 'k_dx' or 'k', got {xunit!r}")


def _draw(ax, curves, ref, colors, theme, xlabel):
    """curves: label -> (x, y); the reference (if any) is drawn in primary ink, on top."""
    t = S.THEMES[theme]
    cmap = S.assign_colors([k for k in curves if k != ref], colors, theme)
    items = []
    for k, (x, y) in curves.items():
        if k == ref:
            continue
        S.plot_series(ax, x, y, cmap[k], k)
        items.append((k, x[-1], y[-1]))
    if ref is not None and ref in curves:
        x, y = curves[ref]
        S.plot_series(ax, x, y, t["text"], ref, zorder=3)
        items.append((ref, x[-1], y[-1]))
    ax.set_xlabel(xlabel)
    return items


def spectra(fields, dx, reference: str | None = None, xunit: str = "k_dx",
            theme: str = "light", colors: dict | None = None):
    """Energy spectra E(k) (geometric mean over the batch), log-log; reference drawn in ink."""
    if reference is not None and reference not in fields:
        raise KeyError(f"reference {reference!r} not in fields")
    s, xlabel = _xaxis(dx, xunit)
    curves = {}
    for label, f in fields.items():
        k, E = _spec(f, dx)
        curves[label] = (k * s, E)
    with S.style(theme):
        fig, ax = S.new_figure(theme, size=(5.6, 3.8))
        items = _draw(ax, curves, reference, colors, theme, xlabel)
        S.log_axis(ax, "y")
        S.log_axis(ax, "x")
        ax.set_ylabel(r"$E(k)$ (m$^3$ s$^{-2}$)")
        if len(items) <= 4:
            S.end_labels(ax, items, theme)
        S.finish_legend(ax, len(items))
    return fig


def spectra_ratio(fields, dx, reference: str, theme: str = "light", xunit: str = "k_dx",
                  colors: dict | None = None):
    """E(k) / E_ref(k) on log-y with a 1.0 reference line."""
    if reference not in fields:
        raise KeyError(f"reference {reference!r} not in fields")
    s, xlabel = _xaxis(dx, xunit)
    spec = {k: _spec(f, dx) for k, f in fields.items()}
    Er = spec[reference][1]
    curves = {}
    for label, (k, E) in spec.items():
        if label == reference:
            continue
        m = min(len(E), len(Er))
        curves[label] = (k[:m] * s, E[:m] / Er[:m])
    with S.style(theme):
        fig, ax = S.new_figure(theme, size=(5.6, 3.8))
        ax.axhline(1.0, color=S.THEMES[theme]["muted"], lw=1.0, zorder=1)
        items = _draw(ax, curves, None, colors, theme, xlabel)
        S.log_axis(ax, "y")
        S.log_axis(ax, "x")
        ax.set_ylabel(f"E(k) / E(k) of {reference}")
        if len(items) <= 4:
            S.end_labels(ax, items, theme)
        S.finish_legend(ax, len(items))
    return fig
