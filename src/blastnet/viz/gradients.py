"""viz.qr (invariant joint PDF) and viz.parity (hexbin predicted-vs-true per component)."""

from __future__ import annotations

from . import _style as S


def vieillefosse(R):
    """Q on the Vieillefosse line 27/4 R^2 + Q^3 = 0 (the non-positive real root)."""
    import numpy as np

    R = np.asarray(R, dtype=float)
    return -np.cbrt(27.0 / 4.0 * R**2)


def _invariants(f, dx):
    import numpy as np
    import torch

    from blastnet import physics

    t = f if hasattr(f, "dim") else torch.as_tensor(S.to_numpy(f))
    if t.dim() != 5 or t.shape[1] not in (3, 4):
        raise ValueError(f"velocity must be [B, 3, X, Y, Z] or [B, 4, ...], got {tuple(t.shape)}")
    u = t[:, 1:4] if t.shape[1] == 4 else t
    with torch.no_grad():
        _, Q, R = physics.strain_invariants(u, dx).unbind(1)
        qw = float(physics.enstrophy(u, dx).mean()) / 2.0  # <omega^2> / 4, enstrophy = omega^2 / 2
    if not qw > 0:
        raise ValueError("field has zero enstrophy; cannot normalise")
    return (R.reshape(-1).double().numpy() / qw**1.5, Q.reshape(-1).double().numpy() / qw,
            np.float64(qw))


def qr(fields, dx, bins: int = 128, lim=None, theme: str = "light"):
    """Joint PDF of (R*, Q*) = (R / <Q_w>^{3/2}, Q / <Q_w>), <Q_w> = <omega^2> / 4, per field.

    lim: float or (R*, Q*) half-ranges; default the 99.5th percentile of |R*|, |Q*| over fields.
    """
    import numpy as np

    pts = {k: _invariants(f, dx) for k, f in fields.items()}
    if lim is None:
        rl = max(float(np.percentile(np.abs(r[::7]), 98.5)) for r, _, _ in pts.values())
        ql = max(float(np.percentile(np.abs(q[::7]), 98.5)) for _, q, _ in pts.values())
    elif isinstance(lim, int | float):
        rl = ql = float(lim)
    else:
        rl, ql = map(float, lim)
    hists, lmax = {}, -np.inf
    for k, (r, q, _) in pts.items():
        H, re, qe = np.histogram2d(r, q, bins=bins, range=[[-rl, rl], [-ql, ql]], density=True)
        with np.errstate(divide="ignore"):
            L = np.where(H > 0, np.log10(np.where(H > 0, H, 1)), np.nan)
        hists[k] = (L, re, qe)
        lmax = max(lmax, np.nanmax(L))
    levels = np.arange(np.floor(lmax) - 4, np.ceil(lmax) + 1)
    n = len(pts)
    with S.style(theme):
        t = S.THEMES[theme]
        fig, axs = S.new_figure(theme, 1, n, size=(3.4 * n + 0.9, 3.5), sharex=True, sharey=True,
                                squeeze=False)
        axs = axs[0]
        cs = None
        Rg = np.linspace(-rl, rl, 400)
        for ax, (k, (L, re, qe)) in zip(axs, hists.items()):
            rc, qc = 0.5 * (re[1:] + re[:-1]), 0.5 * (qe[1:] + qe[:-1])
            # empty and below-range cells stay unfilled (the surface shows), so the PDF's shape reads clearly
            Lm = np.ma.masked_where(~np.isfinite(L) | (levels[0] > L), L).T
            cs = ax.contourf(rc, qc, Lm, levels=levels, cmap=S.seq_cmap(theme))
            ax.plot(Rg, vieillefosse(Rg), color=t["muted"], lw=1.2, label="Vieillefosse line")
            ax.axhline(0, color=t["grid"], lw=0.8, zorder=0)
            ax.axvline(0, color=t["grid"], lw=0.8, zorder=0)
            ax.set_xlim(-rl, rl)
            ax.set_ylim(-ql, ql)
            ax.set_title(k)
            ax.set_xlabel(r"$R^*=R\,/\,\langle Q_w\rangle^{3/2}$")
            ax.set_label(f"qr:{k}")
        axs[0].set_ylabel(r"$Q^*=Q\,/\,\langle Q_w\rangle$, $Q_w=\omega^2/4$")
        xr = -0.92 * rl
        axs[0].text(xr, vieillefosse(xr) + 0.05 * ql, "Vieillefosse", color=t["muted"],
                    fontsize=8, ha="left", va="bottom")
        fig.colorbar(cs, ax=list(axs), shrink=0.85, pad=0.02).set_label(r"$\log_{10}$ PDF")
    return fig


def parity(pred, true, labels=None, theme: str = "light"):
    """Hexbin density of pred vs true per component (dim 1), y = x line, R^2 per panel."""
    import numpy as np

    p, y = S.to_numpy(pred), S.to_numpy(true)
    if p.shape != y.shape:
        raise ValueError(f"pred {p.shape} and true {y.shape} differ")
    if p.ndim == 1:
        p, y = p[:, None], y[:, None]
    C = p.shape[1]
    p = np.moveaxis(p, 1, 0).reshape(C, -1).astype(float)
    y = np.moveaxis(y, 1, 0).reshape(C, -1).astype(float)
    if labels is None:
        labels = (["11", "22", "33", "12", "13", "23"] if C == 6
                  else [f"component {i}" for i in range(C)])
        labels = [rf"$\tau_{{{s}}}$" if C == 6 else s for s in labels]
    ncol = min(C, 3)
    nrow = -(-C // ncol)
    rng = np.random.default_rng(0)
    with S.style(theme):
        t = S.THEMES[theme]
        fig, axs = S.new_figure(theme, nrow, ncol, size=(3.0 * ncol + 0.9, 3.1 * nrow),
                                squeeze=False)
        hbs = []
        for i, ax in enumerate(axs.ravel()):
            if i >= C:
                ax.set_visible(False)
                continue
            lo, hi = np.percentile(np.concatenate([y[i], p[i]]), [0.1, 99.9])
            pad = 0.05 * (hi - lo) or 1.0
            lo, hi = lo - pad, hi + pad
            sel = rng.choice(y.shape[1], 400_000, replace=False) if y.shape[1] > 400_000 else slice(None)
            hb = ax.hexbin(y[i][sel], p[i][sel], gridsize=45, extent=(lo, hi, lo, hi),
                           bins="log", mincnt=1, cmap=S.seq_cmap(theme), linewidths=0.0)
            hbs.append(hb)
            ax.plot([lo, hi], [lo, hi], color=t["muted"], lw=1.2)
            ss = ((y[i] - y[i].mean()) ** 2).sum()
            r2 = 1.0 - ((y[i] - p[i]) ** 2).sum() / ss if ss > 0 else float("nan")
            ax.text(0.04, 0.95, rf"$R^2$ = {r2:.3f}", transform=ax.transAxes, va="top",
                    color=t["text"], fontsize=9)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(labels[i])
            ax.set_label(f"parity:{i}")
            if i // ncol == nrow - 1 or i + ncol >= C:
                ax.set_xlabel("true")
            if i % ncol == 0:
                ax.set_ylabel("predicted")
        vmax = max(float(h.get_array().max()) for h in hbs)
        for h in hbs:
            h.set_clim(1, max(vmax, 2.0))
        fig.colorbar(hbs[-1], ax=list(axs.ravel()), shrink=0.8, pad=0.02).set_label("count")
    return fig
