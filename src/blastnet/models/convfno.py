"""ConvFNO (3D), exact port of upstream `models/convfno.py` (no shipped weights).

Ported from github.com/blastnet/blastnet2_sr_benchmark (MIT, Copyright (c) 2023 blastnet),
commit ae2a3a6. Module names, parameter shapes, init order, and forward arithmetic are unchanged.
Modernised only: `torch.cuda.amp.custom_fwd(cast_inputs=...)` (deprecated) is spelled
`torch.amp.custom_fwd(device_type="cuda", cast_inputs=...)`, which is what the old name calls.
Upstream quirks kept: there is no "0.8M" label; the bottleneck width equals the FNO width; every
`FNOResLayer.u0` ResBlock shares one ReLU module (EDSR ResBlock's default argument); the body runs
on the input zero-padded by 6 voxels on the high side of each axis, cropped before `fc1`.
"""

# written by W.T. Chung
# modfied from Conv-FNO and U-FNO: https://github.com/gegewen/ufno
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ._common import initialize_weights
from .edsr import ResBlock
from .rcan import Upsampler, default_conv

# label -> n_feats (= fno_width = bottleneck), modes = 2, 32 FNO layers, as in upstream init_convfno.
SIZES = {
    "0.5M": dict(n_feats=12),
    "1.4M": dict(n_feats=20),
    "2.7M": dict(n_feats=24),
    "5M": dict(n_feats=34),
    "11M": dict(n_feats=46),
    "17M": dict(n_feats=68),
    "50M": dict(n_feats=86),  # this is actually ~35M since 50M runs out of memory
}


class UpsamplerDecoder(nn.Module):
    def __init__(self, n_feats, out_channel, kernel_size, scale, conv=default_conv):
        super().__init__()

        # define tail module
        m_tail = [Upsampler(conv, scale, n_feats, act=False), conv(n_feats, out_channel, kernel_size)]
        initialize_weights(m_tail[-1], "linear", None, 0.1)

        self.tail = nn.Sequential(*m_tail)
        self.upscale = scale

    def forward(self, x):
        x = self.tail(x)
        return x


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        shape = (in_channels, out_channels, self.modes1, self.modes2, self.modes3)
        weights1 = self.scale * torch.rand(*shape, dtype=torch.cfloat)
        self.weights1 = nn.Parameter(torch.view_as_real(weights1))
        weights2 = self.scale * torch.rand(*shape, dtype=torch.cfloat)
        self.weights2 = nn.Parameter(torch.view_as_real(weights2))
        weights3 = self.scale * torch.rand(*shape, dtype=torch.cfloat)
        self.weights3 = nn.Parameter(torch.view_as_real(weights3))
        weights4 = self.scale * torch.rand(*shape, dtype=torch.cfloat)
        self.weights4 = nn.Parameter(torch.view_as_real(weights4))
        self.einsum_op = "bixyz,ioxyz->boxyz"

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum(self.einsum_op, input, weights)

    # todo take mixed precision treatment from updated repo
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        # removed //2 + 1 since all input dims are spatial
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1),
                             dtype=torch.cfloat, device=x.device)
        m1, m2, m3 = self.modes1, self.modes2, self.modes3
        out_ft[:, :, :m1, :m2, :m3] = self.compl_mul3d(
            x_ft[:, :, :m1, :m2, :m3], torch.view_as_complex(self.weights1))
        out_ft[:, :, -m1:, :m2, :m3] = self.compl_mul3d(
            x_ft[:, :, -m1:, :m2, :m3], torch.view_as_complex(self.weights2))
        out_ft[:, :, :m1, -m2:, :m3] = self.compl_mul3d(
            x_ft[:, :, :m1, -m2:, :m3], torch.view_as_complex(self.weights3))
        out_ft[:, :, -m1:, -m2:, :m3] = self.compl_mul3d(
            x_ft[:, :, -m1:, -m2:, :m3], torch.view_as_complex(self.weights4))

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNOResLayer(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super().__init__()
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.u0 = ResBlock(default_conv, self.width, 3, res_scale=0.1)

        initialize_weights(self.w0, "relu", None, 0.1)

    def forward(self, input):
        x1 = self.conv0(input)
        x2 = self.w0(input)
        x3 = self.u0(input)
        return F.relu(x1 + x2 + x3)


class FNO3dEncoder(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, in_channel, out_channel=128, n_resblocks=32):
        super().__init__()

        # Lift (rho, u, v, w) + 3 grid coordinates to `width` with fc0, apply n_resblocks
        # FNOResLayers on the padded domain, crop, project to `out_channel` with fc1 + ReLU.

        self.padding = 6  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(in_channel + 3, width)
        initialize_weights(self.fc0, "linear", None, 0.1)

        m_body = [FNOResLayer(modes1, modes2, modes3, width) for _ in range(n_resblocks)]
        self.body = nn.Sequential(*m_body)

        self.fc1 = nn.Linear(width, out_channel)
        initialize_weights(self.fc1, "relu", None, 0.1)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        grid = self.get_grid(x)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        # pad the domain if input is non-periodic for all spatial domains
        x = F.pad(x, [0, self.padding, 0, self.padding, 0, self.padding])

        x = self.body(x)

        x = x[..., : -self.padding, : -self.padding, : -self.padding]
        x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.relu(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x

    def get_grid(self, x):
        batchsize, size_x, size_y, size_z = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float, device=x.device)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float, device=x.device)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float, device=x.device)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1)


class ConvFNO(nn.Module):
    def __init__(self, modes, bottleneck, kernel_size, fno_width=20, in_channel=4, out_channel=4,
                 upscale=8, n_resblocks=32):
        super().__init__()
        self.upscale = upscale
        self.FNOEncoder = FNO3dEncoder(modes, modes, modes, fno_width, in_channel, bottleneck,
                                       n_resblocks=n_resblocks)
        self.UpsamplerDecoder = UpsamplerDecoder(bottleneck, out_channel, kernel_size, upscale)

    def forward(self, x):
        x = self.FNOEncoder(x)
        x = self.UpsamplerDecoder(x)
        return x


def init_convfno(approx_param: str, upscale: int) -> ConvFNO:
    """Build ConvFNO at an upstream size label, as upstream init_convfno."""
    if approx_param not in SIZES:
        raise ValueError(f"convfno size must be one of {list(SIZES)}, got {approx_param!r}")
    modes = 2
    n_feats = SIZES[approx_param]["n_feats"]
    return ConvFNO(modes, n_feats, 3, fno_width=n_feats, in_channel=4, out_channel=4,
                   upscale=upscale)
