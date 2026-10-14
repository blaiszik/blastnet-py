"""Shared helpers for the upstream ports: the weight initialiser (upstream `models/tools.py`).

Ported from github.com/blastnet/blastnet2_sr_benchmark (MIT, Copyright (c) 2023 blastnet),
commit ae2a3a6.
"""

# written by W.T. Chung

import torch


def initialize_weights(m, activation, a, scale):
    """Kaiming-normal init of a Conv3d/ConvTranspose3d/Linear, zero bias, weight scaled by `scale`."""
    if isinstance(m, torch.nn.Conv3d | torch.nn.ConvTranspose3d | torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity=activation, a=a)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    else:
        print("no init for ", m)
    m.weight.data *= scale
