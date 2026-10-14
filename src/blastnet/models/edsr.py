"""EDSR (3D), exact port of upstream `models/edsr.py`.

Ported from github.com/blastnet/blastnet2_sr_benchmark (MIT, Copyright (c) 2023 blastnet),
commit ae2a3a6. Module names, parameter shapes, init order, and forward arithmetic are unchanged.
Upstream quirks kept: the trunk-closing conv `body.<n_resblocks>` keeps PyTorch's default init
(no kaiming * 0.1), and `ResBlock`'s default `act` is one ReLU instance created at import time,
shared by every ResBlock built without an explicit `act` (ConvFNO's `u0` blocks).
"""

# written by W.T. Chung and B. Akoush
# modified from https://github.com/sanghyun-son/EDSR-PyTorch
from torch import nn

from ._common import initialize_weights
from .rcan import Upsampler, default_conv

# label -> n_feats (upstream init_edsr; 32 residual blocks, kernel 3, res_scale 0.1).
# "50M" is really ~35M parameters (upstream: 50M ran out of memory).
SIZES = {
    "0.5M": dict(n_feats=14),
    "0.8M": dict(n_feats=20),
    "1.4M": dict(n_feats=24),
    "2.7M": dict(n_feats=34),
    "5M": dict(n_feats=46),
    "11M": dict(n_feats=68),
    "17M": dict(n_feats=86),
    "50M": dict(n_feats=120),
}


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1,  # noqa: B008 (upstream shared default)
    ):
        super().__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if i == 0:
                initialize_weights(m[-1], "relu", None, 0.1)
            else:
                initialize_weights(m[-1], "linear", None, 0.1)
            if bn:
                m.append(nn.BatchNorm3d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class EDSR(nn.Module):
    def __init__(self, in_channels, n_feats, kernel_size, res_scale, n_resblocks, scale,
                 conv=default_conv):
        super().__init__()

        # define head module
        m_head = [conv(in_channels, n_feats, kernel_size)]
        initialize_weights(m_head[-1], "linear", None, 0.1)

        # define body module
        m_body = [
            ResBlock(conv, n_feats, kernel_size, act=nn.ReLU(True), res_scale=res_scale)
            for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [Upsampler(conv, scale, n_feats, act=False), conv(n_feats, in_channels, kernel_size)]
        initialize_weights(m_tail[-1], "linear", None, 0.1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.upscale = scale

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x


def init_edsr(approx_param: str, upscale: int) -> EDSR:
    """Build EDSR at an upstream size label, as upstream init_edsr."""
    if approx_param not in SIZES:
        raise ValueError(f"edsr size must be one of {list(SIZES)}, got {approx_param!r}")
    return EDSR(in_channels=4, n_feats=SIZES[approx_param]["n_feats"], kernel_size=3,
                res_scale=0.1, n_resblocks=32, scale=upscale)
