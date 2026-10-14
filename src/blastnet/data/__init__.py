"""BLASTNet Momentum128 data layer: shipped .dat layout, datasets, synthetic cubes, Zarr.

    from blastnet.data import Momentum128, norm_stats, synthetic
    ds = Momentum128(root, split="train", ratio=8, crop=64, augment=True)
    item = ds[0]   # {"lr", "hr", "dx" (metres), "index", "hash"}
"""

from blastnet.data import synthetic
from blastnet.data.cubes import CubeDataset
from blastnet.data.dat import (
    CHANNELS,
    COLUMNS,
    DX_MIN,
    RATIOS,
    SPLITS,
    VARS,
    VECTOR_CHANNELS,
    cube_path,
    load_meta,
    norm_stats,
    read_cube,
    read_dat,
    subset,
)
from blastnet.data.info import Case, open_case
from blastnet.data.momentum import Momentum128, Momentum128Zarr
from blastnet.data.zarr_store import zarr_convert

__all__ = [
    "CHANNELS",
    "COLUMNS",
    "DX_MIN",
    "RATIOS",
    "SPLITS",
    "VARS",
    "VECTOR_CHANNELS",
    "Case",
    "CubeDataset",
    "Momentum128",
    "Momentum128Zarr",
    "cube_path",
    "load_meta",
    "norm_stats",
    "open_case",
    "read_cube",
    "read_dat",
    "subset",
    "synthetic",
    "zarr_convert",
]
