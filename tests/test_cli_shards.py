# tests/test_cli_shards.py
from __future__ import annotations

import numpy as np
import pytest
import zarr

import pymif.microscope_manager as mm
from pymif.cli.__arguments import parse_shard_exclude_axes_spec, parse_shards_spec
from pymif.cli.pymif import (
    _normalize_shard_exclude_axes,
    _normalize_shards,
    zarr_convert,
)


# --- parse_shards_spec / parse_shard_exclude_axes_spec (batch CSV / string form) ---

def test_parse_shards_spec_auto():
    assert parse_shards_spec("auto") == "auto"
    assert parse_shards_spec("AUTO") == "auto"


def test_parse_shards_spec_shape_string():
    assert parse_shards_spec("1 1 8 2160 4096") == (1, 1, 8, 2160, 4096)
    assert parse_shards_spec("1,1,8,2160,4096") == (1, 1, 8, 2160, 4096)


def test_parse_shards_spec_empty_and_none():
    assert parse_shards_spec(None) is None
    assert parse_shards_spec("") is None
    assert parse_shards_spec("none") is None


def test_parse_shards_spec_invalid_raises():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        parse_shards_spec("1 1 z 2160 4096")


def test_parse_shard_exclude_axes_spec():
    assert parse_shard_exclude_axes_spec("t c") == ("t", "c")
    assert parse_shard_exclude_axes_spec("none") == ()
    assert parse_shard_exclude_axes_spec(None) is None


def test_parse_shard_exclude_axes_spec_invalid_raises():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        parse_shard_exclude_axes_spec("t q")


# --- _normalize_shards / _normalize_shard_exclude_axes (pymif.py, CLI list or CSV string) ---

def test_normalize_shards_from_cli_list():
    assert _normalize_shards(["auto"]) == "auto"
    assert _normalize_shards(["1", "1", "8", "2160", "4096"]) == (1, 1, 8, 2160, 4096)


def test_normalize_shards_from_csv_string():
    assert _normalize_shards("auto") == "auto"
    assert _normalize_shards("1 1 8 2160 4096") == (1, 1, 8, 2160, 4096)


def test_normalize_shards_none():
    assert _normalize_shards(None) is None


def test_normalize_shard_exclude_axes():
    assert _normalize_shard_exclude_axes(["t", "c"]) == ("t", "c")
    assert _normalize_shard_exclude_axes("t c") == ("t", "c")
    assert _normalize_shard_exclude_axes("none") == ()
    assert _normalize_shard_exclude_axes(None) is None


# --- zarr_convert end-to-end with sharding ---

def test_zarr_convert_with_auto_shards(tmp_path):
    """The CLI conversion helper must thread shards through to the written zarr store."""
    src = tmp_path / "source.zarr"
    out = tmp_path / "out.zarr"

    lvl0 = np.arange(1 * 2 * 32 * 256 * 256, dtype="uint16").reshape(1, 2, 32, 256, 256)
    metadata = {
        "axes": "tczyx",
        "scales": [(1.0, 1.0, 1.0)],
        "time_increment": 1.0,
        "time_increment_unit": "second",
        "channel_names": ["A", "B"],
        "channel_colors": ["FF0000", "00FF00"],
    }
    mm.ArrayManager(lvl0, metadata, chunks=(1, 1, 4, 64, 64)).to_zarr(
        str(src), ngff_version="0.5", zarr_format=3, overwrite=True
    )

    zarr_convert(
        input_path=str(src),
        zarr_path=str(out),
        microscope="zarr",
        chunk_size=[1, 1, 4, 64, 64],
        zarr_format=3,
        num_levels=1,
        shards="auto",
        shard_target_mb=0.5,
        shard_exclude_axes=["t", "c"],
    )

    root = zarr.open_group(str(out), mode="r")
    z = root["0"]
    assert z.chunks == (1, 1, 4, 64, 64)
    assert z.shards is not None
    # t and c must stay unmerged (equal to chunk size on those axes).
    assert z.shards[0] == z.chunks[0]
    assert z.shards[1] == z.chunks[1]
    # at least one of z/y/x must have grown.
    assert any(z.shards[i] != z.chunks[i] for i in (2, 3, 4))


def test_zarr_convert_without_shards_is_unchanged(tmp_path):
    """Omitting shards must produce a plain unsharded store (backward compatible)."""
    src = tmp_path / "source2.zarr"
    out = tmp_path / "out2.zarr"

    lvl0 = np.arange(1 * 2 * 8 * 64 * 64, dtype="uint16").reshape(1, 2, 8, 64, 64)
    metadata = {
        "axes": "tczyx",
        "scales": [(1.0, 1.0, 1.0)],
        "time_increment": 1.0,
        "time_increment_unit": "second",
        "channel_names": ["A", "B"],
        "channel_colors": ["FF0000", "00FF00"],
    }
    mm.ArrayManager(lvl0, metadata, chunks=(1, 1, 8, 64, 64)).to_zarr(
        str(src), ngff_version="0.5", zarr_format=3, overwrite=True
    )

    zarr_convert(
        input_path=str(src),
        zarr_path=str(out),
        microscope="zarr",
        chunk_size=[1, 1, 8, 64, 64],
        zarr_format=3,
        num_levels=1,
    )

    root = zarr.open_group(str(out), mode="r")
    assert root["0"].shards is None
