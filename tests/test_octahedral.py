import torch

from blastnet import octahedral as O


def _field(c=4, n=6, seed=0):
    return torch.randn(2, c, n, n, n, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def test_group_structure():
    G = O.elements()
    assert len(G) == 48 and len({(g.perm, g.signs) for g in G}) == 48
    assert G[0] == O.identity() and G[0].matrix.equal(torch.eye(3, dtype=torch.float64))
    assert sum(g.is_rotation for g in G) == 24
    T = O.table()
    assert all(sorted(row.tolist()) == list(range(48)) for row in T)  # Latin square
    for g in G:
        assert O.compose(g, O.inverse(g)) == O.identity()
        for h in G:
            assert torch.equal(O.compose(g, h).matrix, g.matrix @ h.matrix)


def test_action_is_homomorphism():
    x = _field()
    G = O.elements()
    for g in G:
        gx = O.apply(x, g)
        for h in G:
            assert torch.equal(O.apply(O.apply(x, h), g), O.apply(x, O.compose(g, h)))
        assert torch.equal(O.apply(gx, O.inverse(g)), x)
    assert torch.equal(O.apply(x, O.identity()), x)


def test_vector_channels_transform_like_a_gradient():
    # grad(g . phi) == g . grad(phi) on a periodic grid: fixes the sign/permutation convention.
    n = 8
    phi = _field(c=1, n=n)

    def grad(f):
        return torch.cat([(f.roll(-1, d) - f.roll(1, d)) / 2 for d in (2, 3, 4)], dim=1)

    for g in O.elements():
        lhs = grad(O.apply(phi, g, vector_channels=None))
        rhs = O.apply(grad(phi), g, vector_channels=(0, 1, 2))
        assert torch.allclose(lhs, rhs), g


def test_scalars_and_positions():
    x = _field()
    g = O.elements()[13]
    y = O.apply(x, g, vector_channels=None)
    assert torch.equal(y[:, 0], O.apply(x[:, :1], g, None)[:, 0])
    assert y.shape == x.shape


def test_random_is_seeded():
    x = _field()
    a, ga = O.random(x, generator=torch.Generator().manual_seed(3))
    b, gb = O.random(x, generator=torch.Generator().manual_seed(3))
    assert ga == gb and torch.equal(a, b)
