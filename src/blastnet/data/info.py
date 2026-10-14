"""General BLASTNet case layout (info.json + .dat), as used by the Kaggle BLASTNet datasets.

    <case>/info.json    global: dataset_id, Nxyz, snapshots, variables, compression, grid, bc
                        local:  [{id, "<VAR> filename": "./data/<VAR>_id000.dat", ...}, ...]
    <case>/data/*.dat, <case>/grid/*.dat    little-endian float32, C order (i, j, k) = (x, y, z)

    case = open_case("liftedh2full")
    u = case.fields(["UX_ms-1", "UY_ms-1"], snapshot=0, region=(slice(0, 64),) * 3)   # [2, 64, 64, 64]

Domains can be large (several GB per variable), so nothing here reads a whole field: files are
memory-mapped and only the requested slices are copied. A "snapshot" is the position in the
`local` list.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from blastnet.data.dat import read_dat

__all__ = ["Case", "open_case"]


class Case:
    """One BLASTNet case directory. Construction reads info.json only, never any data."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        with open(self.root / "info.json") as f:
            info = json.load(f)
        g = info.get("global", info)
        self.global_meta: dict = dict(g)
        self.local_meta: list[dict] = list(info.get("local", []))
        self.id: str = str(g.get("dataset_id", self.root.name))
        self.shape: tuple[int, int, int] = tuple(int(n) for n in g["Nxyz"])  # type: ignore[assignment]
        self.variables: tuple[str, ...] = tuple(g["variables"])
        self.n_snapshots: int = len(self.local_meta)
        bc = g.get("bc", {})
        # real info.json files carry bc as free text too (e.g. the BLASTNet tutorial toy dataset:
        # "Periodic in x-, y-, and z-directions."), not only as a dict
        self.bc: dict = dict(bc) if isinstance(bc, dict) else {"description": str(bc)}

    def __repr__(self) -> str:
        return (f"Case({self.id!r}, shape={self.shape}, variables={len(self.variables)}, "
                f"snapshots={self.n_snapshots})")

    def _path(self, rel: str) -> Path:
        return self.root / rel

    def _grid(self, axis: str) -> np.memmap:
        return read_dat(self._path(self.global_meta["grid"][axis]), self.shape)

    def coords(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """1-D x, y, z coordinates (metres), from memmap slices X[:,0,0], Y[0,:,0], Z[0,0,:]."""
        x, y, z = (self._grid(a) for a in "xyz")
        return (np.array(x[:, 0, 0]), np.array(y[0, :, 0]), np.array(z[0, 0, :]))

    def spacing(self, rtol: float = 1e-4) -> tuple[float | None, float | None, float | None]:
        """Grid spacing (dx, dy, dz) in metres; None on a non-uniform (or single-point) axis."""
        out: list[float | None] = []
        for c in self.coords():
            if len(c) < 2:
                out.append(None)
                continue
            d = np.diff(c.astype(np.float64))
            mean = d.mean()
            ok = mean != 0 and np.abs(d - mean).max() <= rtol * abs(mean)
            out.append(float(mean) if ok else None)
        return (out[0], out[1], out[2])

    def _file(self, var: str, snapshot: int) -> Path:
        if var not in self.variables:
            raise KeyError(f"{var!r} not in variables of case {self.id!r}: {self.variables}")
        if not -self.n_snapshots <= snapshot < self.n_snapshots:
            raise IndexError(f"snapshot {snapshot} out of range for {self.n_snapshots} snapshots")
        return self._path(self.local_meta[snapshot][f"{var} filename"])

    def field(self, var: str, snapshot: int = 0) -> np.memmap:
        """Read-only [Nx, Ny, Nz] float32 memmap of one variable at one snapshot."""
        return read_dat(self._file(var, snapshot), self.shape)

    def fields(self, vars, snapshot: int = 0,
               region: tuple[slice, slice, slice] | None = None) -> np.ndarray:
        """Stacked [C, ...] float32 copy of `vars`; with `region` (3 slices) only that block is read."""
        if isinstance(vars, str):
            vars = [vars]
        if region is not None and len(region) != 3:
            raise ValueError(f"region must be 3 slices, got {region!r}")
        reg = tuple(region) if region is not None else (slice(None),) * 3
        return np.stack([np.array(self.field(v, snapshot)[reg]) for v in vars])

    def check(self) -> list[str]:
        """Problems found without reading field data: an empty list means the case looks sound."""
        msgs: list[str] = []
        want = int(np.prod(self.shape)) * 4

        def size_problem(path: Path, what: str) -> bool:
            if not path.exists():
                msgs.append(f"missing file: {what} {path}")
                return False
            size = os.path.getsize(path)
            if size != want:
                msgs.append(f"size mismatch: {what} {path} is {size} bytes, expected {want}")
                return False
            return True

        grid_ok = True
        for a in "xyz":
            rel = self.global_meta.get("grid", {}).get(a)
            if rel is None:
                msgs.append(f"missing grid entry for axis {a}")
                grid_ok = False
            else:
                grid_ok &= size_problem(self._path(rel), f"grid {a}")
        for i, loc in enumerate(self.local_meta):
            for var in self.variables:
                rel = loc.get(f"{var} filename")
                if rel is None:
                    msgs.append(f"snapshot {i}: no '{var} filename' entry")
                    continue
                if not Path(rel).name.startswith(f"{var}_"):
                    msgs.append(f"label mismatch: snapshot {i} variable {var} points at {rel}")
                size_problem(self._path(rel), f"snapshot {i} {var}")
        if grid_ok:
            for a, dx in zip("xyz", self.spacing(), strict=True):
                if dx is None:
                    msgs.append(f"non-uniform grid along {a}")
        return msgs


def open_case(root: str | Path) -> Case:
    """Open a BLASTNet case directory (the one holding info.json)."""
    return Case(root)
