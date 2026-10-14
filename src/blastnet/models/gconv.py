"""Exactly O_h-equivariant EDSR-shaped super-resolution net (regular group convolutions).

The input and output carry the representation rho_in = rho_out = 1 scalar + 1 polar vector, in the
upstream channel order (rho, u, v, w) with vector channels (1, 2, 3), acted on by
`octahedral.apply`. Every hidden feature carries the regular representation of the 48-element group
O_h: a hidden tensor is [B, C * 48, X, Y, Z], channel `c * 48 + k` holding the copy of feature `c`
belonging to group element `k = octahedral.elements()[k].index` (group axis innermost). As in
Agdestein and Sanderse (group-equivariant CNNs for LES closure), equivariance holds exactly by weight
tying, so for every g, `model(apply(x, g)) == apply(model(x), g)` up to float rounding, on a cube
with zero padding.

How g acts on a hidden feature h (the left regular representation, see `feature_action`):

    (g . h)[:, c * 48 + k](p) = h[:, c * 48 + table[g^-1, k]](M_g^-1 p)

i.e. move the grid exactly as `octahedral.apply(h, g, vector_channels=None)` does, then permute the
group axis so that slot k reads slot g^-1 k (`octahedral.table()[inverse(g).index, k]`).

Tied kernels, written for offsets q in {-1, 0, 1}^3 (conv3d cross-correlation, out(p) = sum_q
W(q) in(p + q)), with R(g) the input/output representation (identity on rho, M_g on (u, v, w)):

    lift     (rep -> regular)      W_k(q)          = K(M_k^-1 q) R(k)^-1
    group    (regular -> regular)  W_k(q, l)       = K(M_k^-1 q, k^-1 l)
    subpixel (regular -> regular,  W_{k,s}(q, l)   = K(M_k^-1 s, M_k^-1 q, k^-1 l), then a 3D
              x2 upsampling)                         pixel shuffle; s = sub-voxel offset in {-1/2, 1/2}^3
    project  (regular -> rep)      W(q, l)         = R(l) K(M_l^-1 q)

K is the only free parameter; the dense conv3d kernel is built from it each forward by one gather
through a precomputed index (plus a sign for R), not by summing 48 transformed copies. Biases are
shared over the group axis (and over s); the output bias is free on rho and zero on (u, v, w), the
only R-invariant choice.

Architecture (module names mirror upstream EDSR so `sites()` and hooks line up):

    head    = Sequential(LiftConv3d(4 -> C))                                       the stem
    body.i  = ResBlock(GroupConv3d(C -> C), ReLU, GroupConv3d(C -> C)) * 0.1 + x,  i < n_blocks
    body.n  = GroupConv3d(C -> C)                                                  closes the trunk
    tail    = Sequential(Upsampler(log2(ratio) x [GroupSubpixelConv3d, PixelShuffle3d(2)]),
                         ProjectConv3d(-> 4))
    forward: h = head(x); tail(body(h) + h)

ReLU is pointwise, so it commutes with the group-axis permutation. The upsampler stage i maps
multiplicity m_i -> m_{i+1} = max(1, C >> i) (C, then halving) so the 128^3 features stay at 48
channels; the tail projection reads multiplicity 1 for every size.

upsampler="resize" (family `gconvrc`) replaces each [GroupSubpixelConv3d, PixelShuffle3d(2)] stage by
[Upsample(x2, trilinear, align_corners=False), GroupConv3d], the resize-convolution of Odena et al.
(2016). A sub-pixel shuffle can imprint peaks at the harmonics of the 8-voxel LR cell
(k dx = m pi / 4), which O_h symmetry does not forbid. Trilinear interpolation with
align_corners=False is symmetric about the cube centre and acts on each channel alone, so it
commutes with the regular action and the net stays exactly equivariant.
Same FLOPs per stage (one conv at the fine grid instead of an 8x wider conv at the coarse grid), 8x
fewer upsampler weights, but 4-5x the wall time per epoch on A100s (8x the activation traffic).
"""

from __future__ import annotations

import math
from functools import cache

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from blastnet import octahedral

__all__ = [
    "EDSR_PARAMS",
    "SIZES",
    "GConvSR",
    "GroupConv3d",
    "GroupSubpixelConv3d",
    "LiftConv3d",
    "ProjectConv3d",
    "ResBlock",
    "build_gconv",
    "feature_action",
    "gconv_sites",
]

N_GROUP = 48
REP_CHANNELS = 4  # (rho, u, v, w): 1 scalar + 1 polar vector on channels (1, 2, 3)

# Parameter counts of upstream init_edsr(size, ratio) (reference/blastnet2_sr_benchmark,
# commit ae2a3a6), counted by importing the reference by path: n_feats 14 / 20 / 24, 32 blocks.
EDSR_PARAMS = {
    ("0.5M", 8): 475_276, ("0.5M", 16): 517_724, ("0.5M", 32): 560_172,
    ("0.8M", 8): 967_324, ("0.8M", 16): 1_053_884, ("0.8M", 32): 1_140_444,
    ("1.4M", 8): 1_391_476, ("1.4M", 16): 1_516_084, ("1.4M", 32): 1_640_692,
}

# size -> (multiplicity C, residual blocks). Free parameters (exact, see test_gconv):
#   0.5M: C=2 ->   409,997 (r8, -13.7 %),   420,366 (r16, -18.8 %),   430,735 (r32, -23.1 %)
#   0.8M: C=3 ->   893,580 (r8,  -7.6 %),   903,949 (r16, -14.2 %),   914,318 (r32, -19.8 %)
#   1.4M: C=4 -> 1,618,220 (r8, +16.3 %), 1,628,589 (r16,  +7.4 %), 1,638,958 (r32,  -0.1 %)
# The trunk (65 group convs) holds ~78-85 % of them; every ratio stays within 30 % of EDSR.
SIZES = {"0.5M": (2, 32), "0.8M": (3, 32), "1.4M": (4, 32)}


# ---------------------------------------------------------------------------------------------
# closed-form index tables


@cache
def _grid_index(k: int) -> Tensor:
    """[48, k^3] long: row g lists, for each flat offset a, the flat offset of M_g^-1 q_a."""
    base = torch.arange(k**3).reshape(1, 1, k, k, k)
    return torch.stack([octahedral.apply(base, g, None).flatten() for g in octahedral.elements()])


@cache
def _group_index() -> Tensor:
    """[48, 48] long: entry (k, l) = index of k^-1 l."""
    inv = torch.tensor([octahedral.inverse(g).index for g in octahedral.elements()])
    return octahedral.table()[inv]


@cache
def _rep_signed_perm() -> tuple[Tensor, Tensor]:
    """(src, sign), each [48, 4]: (R(g) v)_i = sign[g, i] * v[src[g, i]] for v = (rho, u, v, w)."""
    src = torch.zeros(N_GROUP, REP_CHANNELS, dtype=torch.long)
    sign = torch.ones(N_GROUP, REP_CHANNELS)
    for g in octahedral.elements():
        for i in range(3):
            src[g.index, 1 + i] = 1 + g.perm[i]
            sign[g.index, 1 + i] = g.signs[i]
    return src, sign


@cache
def _lift_index(ks: int) -> tuple[Tensor, Tensor]:
    """Index [48, 4, ks^3] into K.reshape(Cout, 4 * ks^3) and sign [48, 4, 1] for the lift."""
    src, sign = _rep_signed_perm()
    b = _grid_index(ks)  # [48, ks^3]
    # W_k[:, ci](q) = sum_cj K[:, cj](M_k^-1 q) R(k)^T[cj, ci] = sign[k, ci] K[:, src[k, ci]](...)
    idx = src[:, :, None] * ks**3 + b[:, None, :]
    return idx, sign[:, :, None]


@cache
def _group_conv_index(ks: int) -> Tensor:
    """Index [48(k), 48(l), ks^3] into K.reshape(Cout, Cin, 48 * ks^3)."""
    j = _group_index()  # [48, 48]
    b = _grid_index(ks)  # [48, ks^3]
    return j[:, :, None] * ks**3 + b[:, None, :]


@cache
def _subpixel_index(ks: int) -> Tensor:
    """Index [48(k), 8(s), 48(l), ks^3] into K.reshape(Cout, Cin, 48 * 8 * ks^3)."""
    j = _group_index()
    s = _grid_index(2)  # [48, 8]: sub-voxel offsets of a x2 shuffle, about the block centre
    b = _grid_index(ks)
    return (j[:, None, :, None] * 8 + s[:, :, None, None]) * ks**3 + b[:, None, None, :]


@cache
def _project_index(cin: int, ks: int) -> tuple[Tensor, Tensor]:
    """Index [4, cin, 48, ks^3] into K.flatten() (K: [4, cin, ks^3]) and sign [4, 1, 48, 1]."""
    src, sign = _rep_signed_perm()  # [48, 4]
    b = _grid_index(ks)  # [48, ks^3]
    ci = torch.arange(cin)
    # W[co, (ci, l)](q) = sum_cj R(l)[co, cj] K[cj, ci](M_l^-1 q) = sign[l, co] K[src[l, co], ci](..)
    idx = (src.T[:, None, :, None] * cin + ci[None, :, None, None]) * ks**3 + b[None, None, :, :]
    return idx, sign.T[:, None, :, None]


def _kaiming(t: Tensor, fan_in: int, gain: float, scale: float) -> Tensor:
    """Upstream tools.initialize_weights: kaiming normal on the dense fan-in, times `scale`."""
    with torch.no_grad():
        return t.normal_(0.0, gain / math.sqrt(fan_in)).mul_(scale)


_GAIN = {"relu": math.sqrt(2.0), "linear": 1.0}

# CPU conv3d lowers to im2col with a [B, Cin * k^3, X * Y * Z] buffer: 10.9 GB for the 48-channel
# 3^3 projection at 128^3. Above this many buffer bytes, CPU convs run over x-slabs (same math,
# zero padding applied once up front), bounding peak memory. CUDA/MPS always take one call.
CPU_COL_BYTES = 256 * 2**20


def _conv3d(x: Tensor, w: Tensor, b: Tensor, pad: int) -> Tensor:
    """F.conv3d(x, w, b, padding=pad), split into x-slabs on CPU when im2col would be large."""
    bsz, cin, nx, ny, nz = x.shape
    k = w.shape[-1]
    per_plane = bsz * cin * k**3 * ny * nz * x.element_size()
    if x.device.type != "cpu" or per_plane * nx <= CPU_COL_BYTES:
        return F.conv3d(x, w, b, padding=pad)
    slab = max(1, CPU_COL_BYTES // per_plane)
    xp = F.pad(x, (pad,) * 6)
    outs = [F.conv3d(xp[:, :, i : min(i + slab, nx) + 2 * pad], w, b) for i in range(0, nx, slab)]
    return torch.cat(outs, dim=2)


def _init(weight: Tensor, bias: Tensor | None, fan_in: int, init: str | tuple[str, float]):
    """init = (nonlinearity, scale) as upstream initialize_weights, or "default" (nn.Conv3d)."""
    with torch.no_grad():
        if init == "default":
            bound = 1.0 / math.sqrt(fan_in)
            weight.uniform_(-bound, bound)
            if bias is not None:
                bias.uniform_(-bound, bound)
        else:
            act, scale = init
            _kaiming(weight, fan_in, _GAIN[act], scale)
            if bias is not None:
                bias.zero_()


# ---------------------------------------------------------------------------------------------
# tied convolutions


class LiftConv3d(nn.Module):
    """rep (rho, u, v, w) -> regular [B, Cout * 48, ...]; tied kernel W_k(q) = K(M_k^-1 q) R(k)^-1."""

    def __init__(self, cout: int, kernel_size: int = 3, init=("linear", 0.1)):
        super().__init__()
        self.cout, self.ks = cout, kernel_size
        # free parameters: cout * 4 * ks^3 weights + cout biases (e.g. 2 -> 216 + 2)
        self.weight = nn.Parameter(torch.empty(cout, REP_CHANNELS, kernel_size**3))
        self.bias = nn.Parameter(torch.empty(cout))
        idx, sign = _lift_index(kernel_size)
        self.register_buffer("idx", idx, persistent=False)
        self.register_buffer("sign", sign, persistent=False)
        _init(self.weight, self.bias, REP_CHANNELS * kernel_size**3, init)

    def dense(self) -> tuple[Tensor, Tensor]:
        """Expanded conv3d (weight [Cout*48, 4, k, k, k], bias [Cout*48])."""
        k = self.ks
        w = self.weight.reshape(self.cout, -1)[:, self.idx] * self.sign.to(self.weight.dtype)
        return w.reshape(self.cout * N_GROUP, REP_CHANNELS, k, k, k), self.bias.repeat_interleave(
            N_GROUP)

    def forward(self, x: Tensor) -> Tensor:
        w, b = self.dense()
        return _conv3d(x, w, b, self.ks // 2)


class GroupConv3d(nn.Module):
    """regular -> regular group conv; tied kernel W_k(q, l) = K(M_k^-1 q, k^-1 l)."""

    def __init__(self, cin: int, cout: int, kernel_size: int = 3, init=("linear", 0.1)):
        super().__init__()
        self.cin, self.cout, self.ks = cin, cout, kernel_size
        # free parameters: cout * cin * 48 * ks^3 weights + cout biases (C=2: 5,184 + 2;
        # the dense kernel it fills has 96 * 96 * 27 = 248,832 entries)
        self.weight = nn.Parameter(torch.empty(cout, cin, N_GROUP * kernel_size**3))
        self.bias = nn.Parameter(torch.empty(cout))
        self.register_buffer("idx", _group_conv_index(kernel_size), persistent=False)
        _init(self.weight, self.bias, cin * N_GROUP * kernel_size**3, init)

    def dense(self) -> tuple[Tensor, Tensor]:
        """Expanded conv3d (weight [Cout*48, Cin*48, k, k, k], bias [Cout*48])."""
        k = self.ks
        w = self.weight[:, :, self.idx]  # [Cout, Cin, 48(k), 48(l), k^3]
        w = w.permute(0, 2, 1, 3, 4).reshape(self.cout * N_GROUP, self.cin * N_GROUP, k, k, k)
        return w, self.bias.repeat_interleave(N_GROUP)

    def forward(self, x: Tensor) -> Tensor:
        w, b = self.dense()
        return _conv3d(x, w, b, self.ks // 2)


class GroupSubpixelConv3d(nn.Module):
    """regular -> regular x 8 sub-voxels (pre-shuffle); W_{k,s}(q, l) = K(M_k^-1 s, M_k^-1 q, k^-1 l).

    Followed by `PixelShuffle3d(2)` it is an equivariant x2 upsampler (upstream conv + shuffle).
    """

    def __init__(self, cin: int, cout: int, kernel_size: int = 3, init=("relu", 0.1)):
        super().__init__()
        self.cin, self.cout, self.ks = cin, cout, kernel_size
        # free parameters: cout * cin * 48 * 8 * ks^3 weights + cout biases (2 -> 2: 41,472 + 2)
        self.weight = nn.Parameter(torch.empty(cout, cin, N_GROUP * 8 * kernel_size**3))
        self.bias = nn.Parameter(torch.empty(cout))
        self.register_buffer("idx", _subpixel_index(kernel_size), persistent=False)
        _init(self.weight, self.bias, cin * N_GROUP * kernel_size**3, init)

    def dense(self) -> tuple[Tensor, Tensor]:
        """Expanded conv3d (weight [Cout*48*8, Cin*48, k, k, k], bias [Cout*48*8])."""
        k = self.ks
        w = self.weight[:, :, self.idx]  # [Cout, Cin, 48(k), 8(s), 48(l), k^3]
        w = w.permute(0, 2, 3, 1, 4, 5).reshape(self.cout * N_GROUP * 8, self.cin * N_GROUP, k, k, k)
        return w, self.bias.repeat_interleave(N_GROUP * 8)

    def forward(self, x: Tensor) -> Tensor:
        w, b = self.dense()
        return _conv3d(x, w, b, self.ks // 2)


class ProjectConv3d(nn.Module):
    """regular -> rep (rho, u, v, w); tied kernel W(q, l) = R(l) K(M_l^-1 q); bias on rho only."""

    def __init__(self, cin: int, kernel_size: int = 3, init=("linear", 0.1)):
        super().__init__()
        self.cin, self.ks = cin, kernel_size
        # free parameters: 4 * cin * ks^3 weights + 1 bias (rho; u, v, w bias must be 0);
        # cin=1: 108 + 1, filling a dense 4 x 48 x 27 = 5,184 kernel (W(q, e) = K(q) is free)
        self.weight = nn.Parameter(torch.empty(REP_CHANNELS, cin, kernel_size**3))
        self.bias = nn.Parameter(torch.empty(1))
        idx, sign = _project_index(cin, kernel_size)
        self.register_buffer("idx", idx, persistent=False)
        self.register_buffer("sign", sign, persistent=False)
        _init(self.weight, self.bias, cin * N_GROUP * kernel_size**3, init)

    def dense(self) -> tuple[Tensor, Tensor]:
        """Expanded conv3d (weight [4, Cin*48, k, k, k], bias [4] = (b, 0, 0, 0))."""
        k = self.ks
        w = self.weight.flatten()[self.idx] * self.sign.to(self.weight.dtype)
        b = torch.cat([self.bias, self.bias.new_zeros(REP_CHANNELS - 1)])
        return w.reshape(REP_CHANNELS, self.cin * N_GROUP, k, k, k), b

    def forward(self, x: Tensor) -> Tensor:
        w, b = self.dense()
        return _conv3d(x, w, b, self.ks // 2)


def _pixel_shuffle3d(x: Tensor, r: int) -> Tensor:
    """Upstream rcan.PixelShuffle3d: channel c * r^3 + (s1 r^2 + s2 r + s3) -> voxel r p + s."""
    b, c, d, h, w = x.shape
    n = c // r**3
    y = x.contiguous().view(b, n, r, r, r, d, h, w).permute(0, 1, 5, 2, 6, 3, 7, 4)
    return y.contiguous().view(b, n, d * r, h * r, w * r)


class PixelShuffle3d(nn.Module):
    """3D pixel shuffle with upstream's channel layout (no parameters)."""

    def __init__(self, scale: int):
        super().__init__()
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        return _pixel_shuffle3d(x, self.scale)


class ResBlock(nn.Module):
    """EDSR residual block on regular features: x + res_scale * conv(relu(conv(x)))."""

    def __init__(self, c: int, kernel_size: int = 3, res_scale: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            GroupConv3d(c, c, kernel_size, init=("relu", 0.1)),
            nn.ReLU(inplace=True),
            GroupConv3d(c, c, kernel_size, init=("linear", 0.1)),
        )
        self.res_scale = res_scale

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x) * self.res_scale + x


class GConvSR(nn.Module):
    """EDSR-shaped, exactly O_h-equivariant SR net: [B, 4, n, n, n] -> [B, 4, n*ratio, ...]."""

    def __init__(self, multiplicity: int, n_blocks: int, ratio: int, kernel_size: int = 3,
                 res_scale: float = 0.1, upsampler: str = "subpixel"):
        super().__init__()
        if ratio < 2 or ratio & (ratio - 1):
            raise ValueError(f"ratio must be a power of 2 >= 2, got {ratio}")
        if kernel_size % 2 != 1:
            # padding is kernel_size // 2, so only odd kernels keep the grid size (and the residuals) aligned
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        c = multiplicity
        if upsampler not in ("subpixel", "resize"):
            raise ValueError(f"upsampler must be 'subpixel' or 'resize', got {upsampler!r}")
        self.multiplicity, self.n_blocks, self.upscale, self.upsampler = c, n_blocks, ratio, upsampler
        self.head = nn.Sequential(LiftConv3d(c, kernel_size))
        self.body = nn.Sequential(
            *[ResBlock(c, kernel_size, res_scale) for _ in range(n_blocks)],
            GroupConv3d(c, c, kernel_size, init="default"),
        )
        up, m = [], c
        for i in range(int(math.log2(ratio))):
            m_out = max(1, c >> i)
            if upsampler == "subpixel":
                up += [GroupSubpixelConv3d(m, m_out, kernel_size), PixelShuffle3d(2)]
            else:
                up += [nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                       GroupConv3d(m, m_out, kernel_size, init=("relu", 0.1))]
            m = m_out
        self.tail = nn.Sequential(nn.Sequential(*up), ProjectConv3d(m, kernel_size))

    def forward(self, x: Tensor) -> Tensor:
        h = self.head(x)
        return self.tail(self.body(h) + h)


# ---------------------------------------------------------------------------------------------
# public API


def build_gconv(size: str, ratio: int = 8, upsampler: str = "subpixel") -> nn.Module:
    """Build gconv at an upstream size label ("0.5M", "0.8M", "1.4M"), params matched to EDSR."""
    if size not in SIZES:
        raise ValueError(f"gconv size must be one of {sorted(SIZES)}, got {size!r}")
    c, n_blocks = SIZES[size]
    return GConvSR(c, n_blocks, ratio, upsampler=upsampler)


def gconv_sites(model: nn.Module) -> dict:
    """Qualified module names for the adapter: stem, residual blocks, head (EDSR naming)."""
    blocks = [f"body.{i}" for i, m in enumerate(model.body) if isinstance(m, ResBlock)]
    return dict(stem="head", blocks=blocks, head="tail")


def feature_action(h: Tensor, g: octahedral.Element) -> Tensor:
    """Act with g on a regular hidden feature [B, C*48, X, Y, Z]: grid move + group-axis permutation.

    (g . h)[:, c*48 + k](p) = h[:, c*48 + (g^-1 k)](M_g^-1 p); a group action, so
    feature_action(feature_action(h, b), a) == feature_action(h, compose(a, b)).
    """
    if h.shape[-4] % N_GROUP:
        raise ValueError(f"channel dim {h.shape[-4]} is not a multiple of {N_GROUP}")
    y = octahedral.apply(h, g, vector_channels=None)
    perm = octahedral.table()[octahedral.inverse(g).index].to(h.device)  # slot k reads g^-1 k
    shape = y.shape
    y = y.reshape(*shape[:-4], shape[-4] // N_GROUP, N_GROUP, *shape[-3:])
    return y.index_select(-4, perm).reshape(shape)
