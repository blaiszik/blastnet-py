"""Convert the shipped .dat tree to one Zarr v3 group: <out>/<split>/{hr, lr_{r}x}.

Each array is float32 [N, 4, n, n, n] in CSV row order, channels (rho, u, v, w). The split's
CSV rows are kept verbatim (every cell a string) in the split group's attributes "columns" and
"rows"; the root attributes record channels, vars, hr_size, and ratios.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from blastnet.data.dat import CHANNELS, SPLITS, VARS, _read_csv_raw, read_cube

__all__ = ["zarr_convert"]


def zarr_convert(root: str | os.PathLike, out: str | os.PathLike,
                 splits: tuple[str, ...] = SPLITS, ratios: tuple[int, ...] = (8,),
                 chunks: tuple[int, ...] = (1, 4, 128, 128, 128), hr_size: int = 128) -> Path:
    """Write the HR and LR_{r}x cubes of each split to a Zarr group at `out`; returns out."""
    import zarr

    root, out = Path(root), Path(out)
    group = zarr.open_group(out, mode="w")
    group.attrs.update({"channels": list(CHANNELS), "vars": list(VARS), "hr_size": hr_size,
                        "ratios": list(ratios)})
    for split in splits:
        raw = _read_csv_raw(root / f"{split}_data_summary.csv")
        sg = group.create_group(split)
        sg.attrs.update({"columns": list(raw.columns), "rows": raw.to_numpy().tolist()})
        for r in (None, *ratios):
            n = hr_size if r is None else hr_size // r
            shape = (len(raw), 4, n, n, n)
            arr = sg.create_array("hr" if r is None else f"lr_{r}x", shape=shape,
                                  chunks=tuple(max(1, min(c, s)) for c, s in zip(chunks, shape)),
                                  dtype="float32")
            for i, h in enumerate(raw["hash"]):
                arr[i] = np.stack(read_cube(root, split, h, r, hr_size))
    return out
