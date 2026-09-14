# tests/test_zarr_sharding.py
from __future__ import annotations

import zarr
import numpy as np
import pytest

import pymif.microscope_manager as mm
from pymif.microscope_manager.utils.ngff import _resolve_shards_for_levels


def test_no_shards_by_default(tmp_path, image_pyramid, metadata):
    """Backward compatibility: omitting `shards` writes plain chunked arrays."""
    out = tmp_path / "no_shards.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    writer.to_zarr(str(out), ngff_version="0.5", zarr_format=3, overwrite=True)

    root = zarr.open_group(str(out), mode="r")
    for i in range(3):
        assert root[str(i)].shards is None


def test_auto_shards_reduce_file_count(tmp_path, image_pyramid, metadata):
    out = tmp_path / "auto_shards.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    writer.to_zarr(
        str(out),
        ngff_version="0.5",
        zarr_format=3,
        shards="auto",
        shard_target_mb=1,
        overwrite=True,
    )

    root = zarr.open_group(str(out), mode="r")
    for i in range(3):
        z = root[str(i)]
        # Every axis of a resolved shard must be a whole multiple of the chunk.
        assert z.shards is not None
        for shard_dim, chunk_dim in zip(z.shards, z.chunks):
            assert shard_dim % chunk_dim == 0

    # Data still round-trips correctly through the high-level reader.
    d = mm.ZarrManager(str(out), mode="r")
    np.testing.assert_array_equal(np.asarray(d.data[0]), image_pyramid[0].compute())


def test_explicit_shard_shape_snaps_to_chunk_multiple(tmp_path, image_pyramid, metadata):
    out = tmp_path / "explicit_shards.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    # Level 0 chunks are (1,1,2,8,8); ask for a shard 3x too small in y/x and
    # oversized in z, it should still snap to a valid multiple of the chunk.
    writer.to_zarr(
        str(out),
        ngff_version="0.5",
        zarr_format=3,
        shards=(2, 2, 100, 10, 10),
        overwrite=True,
    )

    root = zarr.open_group(str(out), mode="r")
    z0 = root["0"]
    assert z0.shards is not None
    for shard_dim, chunk_dim in zip(z0.shards, z0.chunks):
        assert shard_dim % chunk_dim == 0
    # z0 shape is (2,2,4,16,16); the oversized z request must be clipped to it.
    assert z0.shards[2] <= z0.shape[2]


def test_per_level_shard_shapes(tmp_path, image_pyramid, metadata):
    out = tmp_path / "per_level_shards.zarr"

    per_level = [
        (2, 2, 4, 16, 16),
        (2, 2, 2, 8, 8),
        (2, 2, 1, 4, 4),
    ]
    writer = mm.ArrayManager(image_pyramid, metadata)
    writer.to_zarr(
        str(out),
        ngff_version="0.5",
        zarr_format=3,
        shards=per_level,
        overwrite=True,
    )

    root = zarr.open_group(str(out), mode="r")
    for i, expected_shape in enumerate(per_level):
        z = root[str(i)]
        assert z.shards is not None
        assert tuple(z.shards) == tuple(expected_shape)


def test_shards_rejected_for_zarr_v2(tmp_path, image_pyramid, metadata):
    out = tmp_path / "bad_v2_shards.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    with pytest.raises(ValueError, match="zarr_format=3"):
        writer.to_zarr(
            str(out),
            ngff_version="0.4",
            zarr_format=2,
            shards="auto",
            overwrite=True,
        )


def test_invalid_shards_string_rejected(tmp_path, image_pyramid, metadata):
    out = tmp_path / "bad_shards_value.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    with pytest.raises(ValueError, match="Unsupported shards value"):
        writer.to_zarr(
            str(out),
            ngff_version="0.5",
            zarr_format=3,
            shards="yes-please",
            overwrite=True,
        )


def test_create_empty_dataset_with_auto_shards(tmp_path):
    """metadata['shards'] drives sharding when creating an empty pyramid."""
    out = tmp_path / "empty_with_shards.zarr"

    metadata = {
        "axes": "cyx",
        "size": [(3, 1024, 1024), (3, 512, 512), (3, 256, 256)],
        "chunksize": [(1, 256, 256), (1, 256, 256), (1, 256, 256)],
        "dtype": "uint16",
        "scales": [(1.0, 1.0), (2.0, 2.0), (4.0, 4.0)],
        "shards": "auto",
        "shard_target_mb": 1,
    }

    manager = mm.ZarrManager(str(out), mode="w", metadata=metadata)
    assert manager.raw.metadata["zarr_format"] == 3

    root = zarr.open_group(str(out), mode="r")
    z0 = root["0"]
    assert z0.shards is not None
    for shard_dim, chunk_dim in zip(z0.shards, z0.chunks):
        assert shard_dim % chunk_dim == 0


def test_auto_shard_skips_when_chunk_already_large(tmp_path, image_pyramid, metadata):
    """A tiny shard_target_mb (smaller than one chunk) must not force sharding."""
    out = tmp_path / "auto_shards_skip.zarr"

    writer = mm.ArrayManager(image_pyramid, metadata)
    writer.to_zarr(
        str(out),
        ngff_version="0.5",
        zarr_format=3,
        shards="auto",
        shard_target_mb=1e-6,
        overwrite=True,
    )

    root = zarr.open_group(str(out), mode="r")
    for i in range(3):
        assert root[str(i)].shards is None


@pytest.mark.parametrize(
    "shape,chunk,target_mb,expect_none",
    [
        # Single chunk already covers the whole level -> nothing to gain.
        ((1, 8, 8), (1, 8, 8), 64, True),
        # Many small chunks and a generous budget -> shard shape should grow.
        ((1, 512, 512), (1, 16, 16), 4, False),
    ],
)
def test_resolve_shards_for_levels_auto(shape, chunk, target_mb, expect_none):
    result = _resolve_shards_for_levels(
        [shape],
        [chunk],
        np.dtype("uint16"),
        "auto",
        zarr_format=3,
        target_bytes=int(target_mb * 1024 * 1024),
    )
    assert len(result) == 1
    if expect_none:
        assert result[0] is None
    else:
        assert result[0] is not None
        for shard_dim, chunk_dim, shape_dim in zip(result[0], chunk, shape):
            assert shard_dim % chunk_dim == 0
            assert shard_dim <= shape_dim
