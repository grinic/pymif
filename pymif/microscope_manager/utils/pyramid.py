import dask.array as da
from typing import List, Tuple, Dict, Any, Sequence, Union


SpatialFactor = Union[int, Sequence[int]]


def normalize_spatial_factor(
    factor: SpatialFactor,
    name: str = "downscale_factor",
) -> Tuple[int, int, int]:
    """Normalize scalar or ZYX downsampling factor.

    Examples
    --------
    2 -> (2, 2, 2)
    (1, 2, 2) -> (1, 2, 2)
    """
    if isinstance(factor, int):
        factors = (factor, factor, factor)
    else:
        factors = tuple(factor)

    if len(factors) != 3:
        raise ValueError(
            f"{name} must be an int or a ZYX sequence of length 3, "
            f"got {factor!r}"
        )

    if any(not isinstance(f, int) for f in factors):
        raise TypeError(f"{name} values must be integers, got {factor!r}")

    if any(f <= 0 for f in factors):
        raise ValueError(f"{name} values must be > 0, got {factor!r}")

    return factors


def factor_power(
    factors: Tuple[int, int, int],
    exponent: int,
) -> Tuple[int, int, int]:
    return tuple(f ** exponent for f in factors)


def multiply_scales(
    scales,
    factors: Tuple[int, int, int],
) -> Tuple[float, float, float]:
    return tuple(float(s) * f for s, f in zip(scales, factors))


def pad_to_divisible(
    array: da.Array,
    factors: SpatialFactor,
) -> da.Array:
    """Pad spatial axes ZYX so they are divisible by downsampling factors."""
    factors = normalize_spatial_factor(factors, name="factors")

    pad_width = [(0, 0)] * array.ndim

    for axis, factor in zip([-3, -2, -1], factors):
        if factor == 1:
            continue

        size = array.shape[axis]
        remainder = size % factor

        if remainder != 0:
            pad_width[axis] = (0, factor - remainder)

    return da.pad(array, pad_width, mode="edge")


def downsample_nn(
    array: da.Array,
    factors: SpatialFactor,
) -> da.Array:
    """Nearest-neighbor downsampling via striding in spatial ZYX axes."""
    factors = normalize_spatial_factor(factors, name="factors")

    slicing = [slice(None)] * array.ndim

    for axis, factor in zip([-3, -2, -1], factors):
        slicing[axis] = slice(0, None, factor)

    return array[tuple(slicing)]


def build_pyramid(
    data_levels: List[da.Array],
    metadata: Dict[str, Any],
    num_levels: int = 3,
    downscale_factor: SpatialFactor = 2,
    start_level: int = 0,
) -> Tuple[List[da.Array], Dict[str, Any]]:
    """
    Generate a multiscale pyramid and updated metadata for NGFF-compatible
    OME-Zarr writing.

    downscale_factor can be either:

    - 2, meaning downsample Z, Y and X by 2
    - (1, 2, 2), meaning keep Z unchanged and downsample only YX
    """
    factors = normalize_spatial_factor(downscale_factor)

    if data_levels[0].ndim != 5:
        raise ValueError("Expected base_level to have shape (T, C, Z, Y, X)")

    print(f"Requested start level {start_level}")

    if start_level < len(data_levels):
        print("Resolution layer already available.")
        pyramid = [data_levels[start_level]]
        new_scales = [tuple(metadata["scales"][start_level])]

    else:
        print("Resolution layer not available: will compute.")

        base_downscale = factor_power(factors, start_level)

        current = pad_to_divisible(data_levels[0], base_downscale)
        down = downsample_nn(current, base_downscale)

        pyramid = [down]
        new_scales = [
            multiply_scales(metadata["scales"][0], base_downscale)
        ]

    print("Creating pyramid.")

    for _ in range(1, num_levels):
        current = pad_to_divisible(pyramid[-1], factors)
        down = downsample_nn(current, factors)
        pyramid.append(down)

    print("Updating metadata.")

    for level in range(1, num_levels):
        scale_factor = factor_power(factors, level)
        new_scales.append(
            multiply_scales(new_scales[0], scale_factor)
        )

    metadata["scales"] = new_scales

    metadata["size"] = [level.shape for level in pyramid]

    return pyramid, metadata