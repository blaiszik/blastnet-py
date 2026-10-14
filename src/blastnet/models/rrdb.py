"""RRDBNet (3D ESRGAN generator), exact port of upstream `models/rrdb.py`; also `rrdbgrad`.

Ported from github.com/blastnet/blastnet2_sr_benchmark (MIT, Copyright (c) 2023 blastnet),
commit ae2a3a6. Module names, parameter shapes, init order, and forward arithmetic are unchanged.
`rrdbgrad` is this architecture trained with upstream's `mse_grad` loss.
"""

# written by W.T. Chung
# modified from https://github.com/xinntao/ESRGAN/blob/master/RRDBNet_arch.py
import functools
import math

import torch
import torch.nn.functional as F
from torch import nn

from ._common import initialize_weights

# label -> (first_channel = nf, model_blocks = nb), as in upstream init_rrdb (gc = 32)
SIZES = {
    "0.5M": dict(first_channel=4, model_blocks=1),
    "0.8M": dict(first_channel=16, model_blocks=1),
    "1.4M": dict(first_channel=32, model_blocks=1),
    "2.7M": dict(first_channel=64, model_blocks=1),
    "5M": dict(first_channel=64, model_blocks=2),
    "11M": dict(first_channel=64, model_blocks=5),
    "17M": dict(first_channel=64, model_blocks=8),
    "50M": dict(first_channel=64, model_blocks=23),
}


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super().__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv3d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv3d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv3d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv3d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv3d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # initialization
        initialize_weights(self.conv1, activation="leaky_relu", a=0.2, scale=0.1)
        initialize_weights(self.conv2, activation="leaky_relu", a=0.2, scale=0.1)
        initialize_weights(self.conv3, activation="leaky_relu", a=0.2, scale=0.1)
        initialize_weights(self.conv4, activation="leaky_relu", a=0.2, scale=0.1)
        initialize_weights(self.conv5, activation="linear", a=0, scale=0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""

    def __init__(self, nf, gc=32):
        super().__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class InterpolateLayer(nn.Module):
    def __init__(self, scale_factor=2, mode="nearest"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4):
        super().__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv3d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = self.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv3d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.HRconv = nn.Conv3d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv3d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.upscale = upscale
        upscale_list = []
        for _ in range(int(math.log(upscale, 2))):
            upscale_list.append(InterpolateLayer(scale_factor=2, mode="nearest"))
            upscale_list.append(nn.Conv3d(nf, nf, 3, 1, 1, bias=True))
            upscale_list.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.upscale_x = nn.Sequential(*upscale_list)

        # initialization taken from supplementary material in ESRGAN paper
        initialize_weights(self.conv_first, activation="linear", a=0, scale=0.1)
        initialize_weights(self.trunk_conv, activation="linear", a=0, scale=0.1)
        initialize_weights(self.HRconv, activation="leaky_relu", a=0.2, scale=0.1)
        initialize_weights(self.conv_last, activation="linear", a=0, scale=0.1)
        for layer in self.upscale_x:
            if isinstance(layer, nn.Conv3d):
                initialize_weights(layer, activation="leaky_relu", a=0.2, scale=0.1)

    def make_layer(self, block, n_layers):
        layers = []
        for _ in range(n_layers):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = torch.add(fea, trunk)
        fea = self.upscale_x(fea)
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out


def init_rrdb(approx_param: str, upscale: int) -> RRDBNet:
    """Build RRDBNet at an upstream size label, as upstream init_rrdb."""
    if approx_param not in SIZES:
        raise ValueError(f"rrdb size must be one of {list(SIZES)}, got {approx_param!r}")
    hp = SIZES[approx_param]
    return RRDBNet(4, 4, hp["first_channel"], hp["model_blocks"], gc=32, upscale=upscale)
