# tests/conftest.py
from __future__ import annotations

import numpy as np
import dask.array as da
import pytest


@pytest.fixture
def image_level0():
    arr = np.arange(2 * 2 * 4 * 16 * 16, dtype=np.uint16).reshape(2, 2, 4, 16, 16)
    return arr


@pytest.fixture
def image_pyramid(image_level0):
    lvl0 = da.from_array(image_level0, chunks=(1, 1, 2, 8, 8))
    lvl1 = da.from_array(image_level0[:, :, ::2, ::2, ::2], chunks=(1, 1, 1, 4, 4))
    lvl2 = da.from_array(image_level0[:, :, ::4, ::4, ::4], chunks=(1, 1, 1, 2, 2))
    return [lvl0, lvl1, lvl2]


@pytest.fixture
def label_pyramid():
    lvl0 = np.zeros((2, 4, 16, 16), dtype=np.uint16)
    lvl0[:, 1:3, 4:12, 4:12] = 7

    lvl1 = lvl0[:, ::2, ::2, ::2]
    lvl2 = lvl0[:, ::4, ::4, ::4]

    return [
        da.from_array(lvl0, chunks=(1, 2, 8, 8)),
        da.from_array(lvl1, chunks=(1, 1, 4, 4)),
        da.from_array(lvl2, chunks=(1, 1, 2, 2)),
    ]


@pytest.fixture
def metadata():
    return {
        "size": [
            (2, 2, 4, 16, 16),
            (2, 2, 2, 8, 8),
            (2, 2, 1, 4, 4),
        ],
        "chunksize": [
            (1, 1, 2, 8, 8),
            (1, 1, 1, 4, 4),
            (1, 1, 1, 2, 2),
        ],
        "scales": [
            (2.0, 0.5, 0.5),
            (4.0, 1.0, 1.0),
            (8.0, 2.0, 2.0),
        ],
        "units": ("micrometer", "micrometer", "micrometer"),
        "time_increment": 60.0,
        "time_increment_unit": "second",
        "channel_names": ["A", "B"],
        "channel_colors": ["FF0000", "00FF00"],
        "dtype": "uint16",
        "plane_files": None,
        "axes": "tczyx",
        "name": "test_image",
    }

def label_metadata_from_image_metadata(image_metadata, name="nuclei", dtype="uint16"):
    """Create explicit label metadata by removing the channel axis from image metadata."""
    md = dict(image_metadata)
    axes = str(md["axes"]).lower()
    if "c" in axes:
        c_axis = axes.index("c")
        md["axes"] = "".join(ax for ax in axes if ax != "c")
        md["size"] = [tuple(v for i, v in enumerate(size) if i != c_axis) for size in md["size"]]
        md["chunksize"] = [tuple(v for i, v in enumerate(chunks) if i != c_axis) for chunks in md["chunksize"]]
    md["name"] = name
    md["dtype"] = dtype
    md["data_type"] = "label"
    md["channel_names"] = []
    md["channel_colors"] = []
    return md


@pytest.fixture
def label_metadata(metadata):
    return label_metadata_from_image_metadata(metadata)
