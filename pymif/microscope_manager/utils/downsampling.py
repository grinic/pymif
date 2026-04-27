# pymif/microscope_manager/utils/downsampling.py

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import Any, TypeAlias

SpatialFactor: TypeAlias = int | float | Sequence[int | float]
SpatialScale: TypeAlias = tuple[float, float, float]
SpatialIntFactor: TypeAlias = tuple[int, int, int]

SPATIAL_AXES = {"z", "y", "x"}


def normalize_spatial_factor(
    value: SpatialFactor,
    *,
    name: str = "downscale_factor",
    allow_float: bool = True,
) -> SpatialScale | SpatialIntFactor:
    """Normalize scalar or ZYX factor to a 3-tuple.

    Scalar 2      -> (2, 2, 2)
    Tuple (1,2,2) -> (1, 2, 2)

    The tuple order is always Z, Y, X.
    """
    if isinstance(value, Real) and not isinstance(value, bool):
        values = (float(value),) * 3
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 3:
            raise ValueError(
                f"`{name}` must be a scalar or a ZYX sequence of length 3, "
                f"got {value!r}."
            )
        values = tuple(float(v) for v in value)
    else:
        raise TypeError(
            f"`{name}` must be a scalar or a ZYX sequence of length 3, "
            f"got {type(value).__name__}."
        )

    if any(v <= 0 for v in values):
        raise ValueError(f"`{name}` values must be > 0, got {values!r}.")

    if not allow_float:
        ints = tuple(int(v) for v in values)
        if any(float(i) != v for i, v in zip(ints, values)):
            raise ValueError(f"`{name}` values must be integers, got {values!r}.")
        return ints

    return values


def spatial_factor_power(factor: SpatialFactor, exponent: int) -> SpatialScale:
    values = normalize_spatial_factor(factor, allow_float=True)

    if exponent >= 0:
        return tuple(v**exponent for v in values)

    return tuple(1.0 / (v ** (-exponent)) for v in values)


def multiply_spatial_scales(
    scales: Sequence[int | float],
    factors: SpatialFactor,
) -> SpatialScale:
    if len(scales) != 3:
        raise ValueError(
            f"`scales` must contain one value for each ZYX axis, got {scales!r}."
        )

    factor_tuple = normalize_spatial_factor(factors, allow_float=True)
    return tuple(float(s) * f for s, f in zip(scales, factor_tuple))


def relative_level_factors(
    total_levels: int,
    ref_level: int,
    downscale_factor: SpatialFactor = 2,
) -> list[SpatialScale]:
    """Return level scale ratios relative to `ref_level`.

    Example with downscale_factor=(1,2,2), ref_level=0:

        level 0 -> (1, 1, 1)
        level 1 -> (1, 2, 2)
        level 2 -> (1, 4, 4)

    Example with ref_level=1:

        level 0 -> (1, 0.5, 0.5)
        level 1 -> (1, 1, 1)
        level 2 -> (1, 2, 2)
    """
    if not 0 <= ref_level < total_levels:
        raise ValueError(
            f"`ref_level` must be in [0, {total_levels}), got {ref_level}."
        )

    return [
        spatial_factor_power(downscale_factor, level - ref_level)
        for level in range(total_levels)
    ]


def axis_names_from_multiscales(multiscales: dict[str, Any]) -> list[str]:
    names: list[str] = []

    for axis in multiscales.get("axes", []) or []:
        if isinstance(axis, dict):
            name = axis.get("name")
        else:
            name = axis

        if name is not None:
            names.append(str(name))

    return names


def _dataset_scale_vector(dataset: dict[str, Any]) -> Sequence[int | float] | None:
    for transform in dataset.get("coordinateTransformations", []) or []:
        if isinstance(transform, dict) and transform.get("type") == "scale":
            return transform.get("scale")

    return None


def level_scale_ratios_from_multiscales(
    multiscales: dict[str, Any],
    datasets: Sequence[dict[str, Any]],
    ref_level: int,
) -> list[SpatialScale] | None:
    """Infer relative spatial level factors from NGFF multiscales metadata.

    This lets region-writing work with existing anisotropic pyramids without
    requiring the caller to pass downscale_factor again.
    """
    if not datasets or not 0 <= ref_level < len(datasets):
        return None

    axis_names = axis_names_from_multiscales(multiscales)
    spatial_positions = [
        i for i, name in enumerate(axis_names)
        if name.lower() in SPATIAL_AXES
    ]

    level_scales: list[SpatialScale] = []

    for dataset in datasets:
        scale_vector = _dataset_scale_vector(dataset)
        if scale_vector is None:
            return None

        if (
            axis_names
            and len(scale_vector) == len(axis_names)
            and len(spatial_positions) == 3
        ):
            spatial = tuple(float(scale_vector[i]) for i in spatial_positions)
        elif len(scale_vector) >= 3:
            spatial = tuple(float(v) for v in scale_vector[-3:])
        else:
            return None

        if len(spatial) != 3:
            return None

        level_scales.append(spatial)

    ref_scale = level_scales[ref_level]

    return [
        tuple(level_scale[i] / ref_scale[i] for i in range(3))
        for level_scale in level_scales
    ]