"""The octahedral group O_h (48 signed axis permutations) acting on [B, C, X, Y, Z] grids.

Frozen contract module, owned by the main loop. An element g is a signed permutation matrix M
(M[i, perm[i]] = signs[i]) acting on positions about the cube centre, p' = M p. On a field this is
(g . f)(p) = R(g) f(M^-1 p), where R(g) = M on the vector channels (velocity is a polar vector, so
reflections flip it) and the identity on every other channel.

    from blastnet import octahedral
    G = octahedral.elements()                         # 48, G[0] is the identity
    y = octahedral.apply(x, G[5], vector_channels=(1, 2, 3))
    octahedral.apply(octahedral.apply(x, h), g) == octahedral.apply(x, octahedral.compose(g, h))

Exact on any cube with symmetric boundary handling (no interpolation, only flips and transposes).
Physical equivariance additionally needs dx == dy == dz; anisotropic spacings permute with M.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cache

import torch

__all__ = ["Element", "apply", "compose", "elements", "identity", "inverse", "random", "table"]


@dataclass(frozen=True)
class Element:
    """One group element: M[i, perm[i]] = signs[i]; index is its position in elements()."""

    perm: tuple[int, int, int]
    signs: tuple[int, int, int]
    index: int

    @property
    def matrix(self) -> torch.Tensor:
        """3x3 signed permutation matrix M (float64)."""
        m = torch.zeros(3, 3, dtype=torch.float64)
        for i in range(3):
            m[i, self.perm[i]] = self.signs[i]
        return m

    @property
    def det(self) -> int:
        """+1 for proper rotations (24 of them), -1 for improper ones."""
        return round(torch.linalg.det(self.matrix).item())

    @property
    def is_rotation(self) -> bool:
        """True for the 24 proper rotations (the chiral subgroup O)."""
        return self.det == 1


@cache
def elements() -> tuple[Element, ...]:
    """All 48 elements in a fixed order; elements()[0] is the identity."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            out.append(Element(tuple(perm), tuple(signs), len(out)))
    return tuple(out)


def identity() -> Element:
    """The identity element."""
    return elements()[0]


def _lookup(perm: tuple[int, ...], signs: tuple[int, ...]) -> Element:
    return elements()[_INDEX[(tuple(perm), tuple(signs))]]


def compose(g: Element, h: Element) -> Element:
    """g . h, i.e. apply h first, then g (matrix product M_g M_h)."""
    perm = tuple(h.perm[g.perm[i]] for i in range(3))
    signs = tuple(g.signs[i] * h.signs[g.perm[i]] for i in range(3))
    return _lookup(perm, signs)


def inverse(g: Element) -> Element:
    """g^-1 (the transpose of M_g)."""
    perm = [0, 0, 0]
    signs = [0, 0, 0]
    for i in range(3):
        perm[g.perm[i]] = i
        signs[g.perm[i]] = g.signs[i]
    return _lookup(tuple(perm), tuple(signs))


@cache
def table() -> torch.Tensor:
    """[48, 48] Cayley table: table()[a, b] = index of elements()[a] . elements()[b]."""
    G = elements()
    return torch.tensor([[compose(a, b).index for b in G] for a in G])


def apply(x: torch.Tensor, g: Element, vector_channels: tuple[int, ...] | None = (1, 2, 3),
          spatial_dims: tuple[int, int, int] = (-3, -2, -1)) -> torch.Tensor:
    """Act with g on a channels-first field: move the grid and rotate the vector channels.

    `x` is [..., C, X, Y, Z] (or any layout, given `spatial_dims` and a channel axis at
    spatial_dims[0] - 1). `vector_channels` lists the 3 channels forming a polar vector, in x, y, z
    order; None treats every channel as a scalar (the grid moves, channels do not mix).
    """
    nd = x.dim()
    sd = [d % nd for d in spatial_dims]
    # (g.f)(p) = f(M^-1 p): output axis i reads input axis perm[i], reversed where signs[i] < 0.
    order = list(range(nd))
    for i in range(3):
        order[sd[i]] = sd[g.perm[i]]
    y = x.permute(order)
    flips = [sd[i] for i in range(3) if g.signs[i] < 0]
    if flips:
        y = torch.flip(y, flips)
    if vector_channels is None:
        return y.contiguous()
    if len(vector_channels) != 3:
        raise ValueError(f"vector_channels must name 3 channels, got {vector_channels}")
    cax = sd[0] - 1
    vc = list(vector_channels)
    v = y.index_select(cax, torch.tensor(vc, device=x.device))
    # v'_i = sum_j M[i, j] v_j = signs[i] * v_{perm[i]}
    rot = torch.stack([g.signs[i] * v.select(cax, g.perm[i]) for i in range(3)], dim=cax)
    y = y.clone()
    y.index_copy_(cax, torch.tensor(vc, device=x.device), rot)
    return y


def random(x: torch.Tensor, vector_channels: tuple[int, ...] | None = (1, 2, 3),
           generator: torch.Generator | None = None) -> tuple[torch.Tensor, Element]:
    """Apply a uniformly random group element; returns (transformed x, the element)."""
    i = int(torch.randint(48, (1,), generator=generator).item())
    g = elements()[i]
    return apply(x, g, vector_channels), g


_INDEX = {(e.perm, e.signs): e.index for e in elements()}
