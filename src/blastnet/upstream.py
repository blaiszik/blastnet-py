"""Exact ports of the upstream BLASTNet 2.0 SR benchmark metrics (github.com/blastnet/
blastnet2_sr_benchmark, commit ae2a3a6, MIT): common/functions.py, models/metrics.py, and the
metric block of models/litmodel.py::LitModel.test_step.

Every function here is bit-for-bit equal to the reference on CPU (tests/test_physics.py checks
torch.equal against the reference checkout). Upstream quirks are kept on purpose, so our numbers
are comparable with the published table:

- torch_dx/dy/dz: torch.gradient with edge_order=1, i.e. second-order central differences in the
  interior and FIRST-order one-sided differences on the two boundary planes; looped over the
  batch with a per-sample scalar spacing h[i]. (blastnet.physics uses edge_order=2 instead.)
- torch_filter: float32 conv3d with an r^3 kernel of 1/r^3, stride r, no padding (a block mean;
  float32 kernel, so float64 inputs raise). favre_filter = torch_filter(rho*phi) / torch_filter(rho).
- sgs: Favre SGS stress filtered to the LR grid and MULTIPLIED BY the filtered density,
  rho_bar * (favre(u_i u_j) - favre(u_i) favre(u_j)), 6 channels ordered 11, 22, 33, 12, 13, 23.
- SSIM3D: Gaussian window 9 (sigma 1.5), zero padding window//2 (edges biased), absolute
  constants C1 = 0.1^2, C2 = 0.3^2 in whatever units the inputs have (physical units in
  test_step), size_average=True = one mean over batch, channels and voxels. size_average=False
  returns ssim_map.mean(1).mean(1).mean(1), which is [B, Z], not [B] (kept).
- GradLoss: dx = 1 for everything (normalised units), sum of 12 MSEs.
- test_step: metrics on de-normalised tensors (y * std + mean, mean/std cast to dx's dtype),
  spacing dx / dx_min * dx_min, SGS divergence with spacing dx * ratio (LR grid), remove_edges
  (one cell off every face) on stresses, their divergences, and the HR fields before MSE / SSIM.
"""

from __future__ import annotations

from math import exp

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "DX_MIN",
    "KEYS",
    "SSIM3D",
    "GradLoss",
    "create_window_3D",
    "divergence_sgs_separate",
    "favre_filter",
    "remove_edges",
    "sgs",
    "ssim3D",
    "ssim3d",
    "test_metrics",
    "torch_dx",
    "torch_dy",
    "torch_dz",
    "torch_filter",
]

DX_MIN = 3.906250185536919e-06
"""LitModel default dx_min [m]; upstream's dataset divides dx by it and test_step multiplies back."""

KEYS = (
    "test_mse_rho", "test_ssim_rho", "test_mse_ux", "test_ssim_ux",
    "test_mse_uy", "test_ssim_uy", "test_mse_uz", "test_ssim_uz",
    "test_mse_sgs", "test_ssim_sgs",
    "test_mse_divsgs1", "test_ssim_divsgs1", "test_mse_divsgs2", "test_ssim_divsgs2",
    "test_mse_divsgs3", "test_ssim_divsgs3",
)
"""The 16 keys LitModel.test_step logs, in its logging order."""


def _grad(phi: torch.Tensor, h: torch.Tensor, dim: int) -> torch.Tensor:
    assert len(phi.shape) == 5
    batch_size = phi.shape[0]
    h = h.reshape(-1)
    my_list = []
    for i in range(batch_size):
        my_list.append(torch.gradient(phi[i:i + 1], spacing=h[i], dim=dim, edge_order=1)[0])
        assert len(my_list[-1].shape) == 5
    return torch.cat(my_list, 0)


def torch_dx(phi: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """d/dx along dim 2 of [B, C, X, Y, Z]; per-sample spacing h [B] (upstream, edge_order=1)."""
    return _grad(phi, h, 2)


def torch_dy(phi: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """d/dy along dim 3 of [B, C, X, Y, Z]; per-sample spacing h [B] (upstream, edge_order=1)."""
    return _grad(phi, h, 3)


def torch_dz(phi: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """d/dz along dim 4 of [B, C, X, Y, Z]; per-sample spacing h [B] (upstream, edge_order=1)."""
    return _grad(phi, h, 4)


def remove_edges(arr: torch.Tensor) -> torch.Tensor:
    """Drop one cell from every face of [B, C, X, Y, Z] (upstream)."""
    return arr[:, :, 1:-1, 1:-1, 1:-1]


def torch_filter(phi: torch.Tensor, fw: int) -> torch.Tensor:
    """Block mean by float32 conv3d, stride fw (upstream; [B, 1, ...] input)."""
    kernel = torch.ones((1, fw, fw, fw), device=phi.device, dtype=torch.float32) / torch.tensor(
        fw**3, dtype=torch.float32, device=phi.device)
    return F.conv3d(phi, kernel.unsqueeze(0), stride=fw)


def favre_filter(rho: torch.Tensor, phi: torch.Tensor, fw: int) -> torch.Tensor:
    """torch_filter(rho * phi) / torch_filter(rho) (upstream)."""
    return torch_filter(rho * phi, fw) / torch_filter(rho, fw)


def sgs(ruvw: torch.Tensor, fw: int) -> torch.Tensor:
    """rho_bar * Favre SGS stress [B, 6, X/fw, ...], order 11, 22, 33, 12, 13, 23 (upstream)."""
    assert ruvw.shape[1] == 4
    rho, u, v, w = ruvw[:, 0:1], ruvw[:, 1:2], ruvw[:, 2:3], ruvw[:, 3:4]
    rhof = torch_filter(rho, fw)
    uf = favre_filter(rho, u, fw)
    vf = favre_filter(rho, v, fw)
    wf = favre_filter(rho, w, fw)

    uuf = favre_filter(rho, u * u, fw)
    uvf = favre_filter(rho, u * v, fw)
    uwf = favre_filter(rho, u * w, fw)
    vvf = favre_filter(rho, v * v, fw)
    vwf = favre_filter(rho, v * w, fw)
    wwf = favre_filter(rho, w * w, fw)
    out = torch.cat([uuf - uf * uf, vvf - vf * vf, wwf - wf * wf, uvf - uf * vf, uwf - uf * wf,
                     vwf - vf * wf], dim=1)
    return out * rhof


def divergence_sgs_separate(stress: torch.Tensor, dx, dy, dz):
    """(div tau)_i for i = 1, 2, 3 from the 6-channel stress, each [B, 1, ...] (upstream)."""
    tau11 = stress[:, 0:1]
    tau22 = stress[:, 1:2]
    tau33 = stress[:, 2:3]
    tau12 = stress[:, 3:4]
    tau13 = stress[:, 4:5]
    tau23 = stress[:, 5:6]
    divtau1 = torch_dx(tau11, dx) + torch_dy(tau12, dy) + torch_dz(tau13, dz)
    divtau2 = torch_dx(tau12, dx) + torch_dy(tau22, dy) + torch_dz(tau23, dz)
    divtau3 = torch_dx(tau13, dx) + torch_dy(tau23, dy) + torch_dz(tau33, dz)
    return divtau1, divtau2, divtau3


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.Tensor(
        [exp(-(x - window_size // 2) ** 2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window_3D(window_size: int, channel: int) -> torch.Tensor:
    """Upstream 3D Gaussian SSIM window (sigma 1.5), [channel, 1, w, w, w] float32."""
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = (_1D_window.mm(_2D_window.reshape(1, -1))
                  .reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0))
    # upstream wraps this in autograd.Variable (a no-op on modern torch)
    return _3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()


def _ssim_map(img1, img2, window, window_size, channel):
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = (0.1) ** 2
    C2 = (0.3) ** 2
    return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
            / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)))


def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    ssim_map = _ssim_map(img1, img2, window, window_size, channel)
    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(1).mean(1).mean(1)  # upstream: [B, Z], not [B]


class SSIM3D(nn.Module):
    """Upstream SSIM3D module (window 9, caches its window per channel count and dtype)."""

    def __init__(self, window_size: int = 9, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """SSIM of two [B, C, X, Y, Z] tensors (scalar when size_average)."""
        (_, channel, _, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)


def ssim3D(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 9,
           size_average: bool = True) -> torch.Tensor:
    """Upstream functional ssim3D (scalar over batch, channels and voxels by default)."""
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    window = window.type_as(img1)
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)


def ssim3d(a: torch.Tensor, b: torch.Tensor, window: int = 9) -> torch.Tensor:
    """Per-sample upstream SSIM of [B, C, X, Y, Z] tensors -> [B] (mean of the upstream map).

    Same map as ssim3D; its batch mean equals ssim3D(a, b) up to summation-order rounding.
    """
    channel = a.shape[1]
    w = create_window_3D(window, channel).type_as(a)
    return _ssim_map(a, b, w, window, channel).flatten(1).mean(1)


class GradLoss(nn.Module):
    """Upstream gradient loss: sum of MSEs of d/dx, d/dy, d/dz of all 4 channels, dx = 1."""

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss between [B, 4, X, Y, Z] prediction and target (normalised units)."""
        nbatch = input.shape[0]
        dx = torch.ones(nbatch, device=input.device)
        dy = torch.ones(nbatch, device=input.device)
        dz = torch.ones(nbatch, device=input.device)
        total = None
        for c in range(4):  # upstream order: r, u, v, w; each d/dx, d/dy, d/dz
            i, t = input[:, c:c + 1], target[:, c:c + 1]
            for fn, h in ((torch_dx, dx), (torch_dy, dy), (torch_dz, dz)):
                term = F.mse_loss(fn(i, h), fn(t, h))
                total = term if total is None else total + term
        return total


def _spacing_m(dx, batch: int) -> torch.Tensor:
    # metres -> float32 [B, 3]
    d = torch.as_tensor(dx, dtype=torch.float32)
    if d.dim() == 0:
        d = d.expand(3)
    if d.dim() == 1:
        d = d.expand(batch, 3)
    if tuple(d.shape) != (batch, 3):
        raise ValueError(f"dx must be a float, 3 values, or [B, 3]; got shape {tuple(d.shape)}")
    return d


@torch.no_grad()
def test_metrics(y_hat: torch.Tensor, y: torch.Tensor, mean, std, dx, ratio: int,
                 dx_min: float = DX_MIN) -> dict[str, float]:
    """LitModel.test_step metrics for one batch: the 16 test_* keys as Python floats.

    y_hat, y: [B, 4, n, n, n] upstream-normalised (rho, u, v, w); mean, std: [4] (norm_stats);
    dx: HR spacing in METRES ([B, 3], 3 values, or a float), as blastnet.data returns it.
    Upstream's dataset divides metres by dx_min and test_step multiplies back; that round trip
    is reproduced here (dx / dx_min * dx_min in float32), so the values equal test_step's.
    Values are per-batch means (MSE and SSIM are means over all voxels of equal-size samples, so
    the per-sample average over a split does not depend on the batch size).
    """
    d = _spacing_m(dx, y.shape[0]).to(y.device) / dx_min  # what upstream's dataloader yields
    dx_s, dy_s, dz_s = d[:, 0].contiguous(), d[:, 1].contiguous(), d[:, 2].contiguous()
    SSIM = SSIM3D()
    mean = torch.as_tensor(mean)[None, :, None, None, None]
    std = torch.as_tensor(std)[None, :, None, None, None]
    upscale = ratio

    mean = mean.type_as(dx_s).to(y.device)
    std = std.type_as(dx_s).to(y.device)
    rescaled_y_hat = y_hat * std + mean
    rescaled_y = y * std + mean
    rescaled_dx = dx_s * dx_min
    rescaled_dy = dy_s * dx_min
    rescaled_dz = dz_s * dx_min

    stressp = sgs(rescaled_y_hat, upscale)
    stresst = sgs(rescaled_y, upscale)
    div1p, div2p, div3p = divergence_sgs_separate(
        stressp, dx=rescaled_dx * upscale, dy=rescaled_dy * upscale, dz=rescaled_dz * upscale)
    div1t, div2t, div3t = divergence_sgs_separate(
        stresst, dx=rescaled_dx * upscale, dy=rescaled_dy * upscale, dz=rescaled_dz * upscale)
    div1t, div2t, div3t = remove_edges(div1t), remove_edges(div2t), remove_edges(div3t)
    div1p, div2p, div3p = remove_edges(div1p), remove_edges(div2p), remove_edges(div3p)
    stressp = remove_edges(stressp)
    stresst = remove_edges(stresst)

    out = {}
    out["test_ssim_divsgs1"] = SSIM(div1p, div1t)
    out["test_ssim_divsgs2"] = SSIM(div2p, div2t)
    out["test_ssim_divsgs3"] = SSIM(div3p, div3t)
    out["test_mse_divsgs1"] = F.mse_loss(div1p, div1t)
    out["test_mse_divsgs2"] = F.mse_loss(div2p, div2t)
    out["test_mse_divsgs3"] = F.mse_loss(div3p, div3t)
    out["test_ssim_sgs"] = SSIM(stressp, stresst)
    out["test_mse_sgs"] = F.mse_loss(stressp, stresst)

    rescaled_y_hat = remove_edges(rescaled_y_hat)
    rescaled_y = remove_edges(rescaled_y)
    for c, name in enumerate(("rho", "ux", "uy", "uz")):
        out[f"test_mse_{name}"] = F.mse_loss(rescaled_y_hat[:, c:c + 1], rescaled_y[:, c:c + 1])
    for c, name in enumerate(("rho", "ux", "uy", "uz")):
        out[f"test_ssim_{name}"] = SSIM(rescaled_y_hat[:, c:c + 1], rescaled_y[:, c:c + 1])
    return {k: float(out[k]) for k in KEYS}
