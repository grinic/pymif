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


def get_spatial_axes(metadata: Dict[str, Any]) -> Tuple[int, int, int]:
    """Return axis indices for Z, Y, X from metadata['axes']."""
    axes = metadata.get("axes", "").lower()

    missing = [ax for ax in "zyx" if ax not in axes]
    if missing:
        raise ValueError(
            f"metadata['axes'] must contain z, y and x. "
            f"Got axes={axes!r}; missing {missing}."
        )

    return axes.index("z"), axes.index("y"), axes.index("x")


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
    spatial_axes: Tuple[int, int, int],
) -> da.Array:
    """Pad spatial axes ZYX so they are divisible by downsampling factors."""
    factors = normalize_spatial_factor(factors, name="factors")

    pad_width = [(0, 0)] * array.ndim

    for axis, factor in zip(spatial_axes, factors):
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
    spatial_axes: Tuple[int, int, int],
) -> da.Array:
    """Nearest-neighbor downsampling via striding in spatial ZYX axes."""
    factors = normalize_spatial_factor(factors, name="factors")

    slicing = [slice(None)] * array.ndim

    for axis, factor in zip(spatial_axes, factors):
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

    This function is axis-aware and uses metadata["axes"] to find Z, Y, X.

    downscale_factor can be either:

    - 2, meaning downsample Z, Y and X by 2
    - (1, 2, 2), meaning keep Z unchanged and downsample only YX
    """
    if not data_levels:
        raise ValueError("data_levels cannot be empty.")

    if num_levels is None:
        num_levels = len(data_levels)

    if num_levels < 1:
        raise ValueError("num_levels must be >= 1.")

    if start_level < 0:
        raise ValueError("start_level must be >= 0.")

    axes = metadata.get("axes")

    if axes is None:
        if data_levels[0].ndim == 5:
            axes = "tczyx"
        elif data_levels[0].ndim == 4:
            axes = "tzyx"
        elif data_levels[0].ndim == 3:
            axes = "zyx"
        else:
            raise ValueError(
                "metadata['axes'] is required when data ndim is not 3, 4, or 5."
            )

        metadata = dict(metadata)
        metadata["axes"] = axes

    axes = axes.lower()

    if data_levels[0].ndim != len(axes):
        raise ValueError(
            f"Array ndim ({data_levels[0].ndim}) does not match "
            f"metadata['axes'] ({metadata.get('axes')!r})."
        )

    if "scales" not in metadata or not metadata["scales"]:
        raise ValueError("metadata must contain a non-empty 'scales' list.")

    factors = normalize_spatial_factor(downscale_factor)
    spatial_axes = get_spatial_axes(metadata)

    print(f"Requested start level {start_level}")

    if start_level < len(data_levels):
        print("Resolution layer already available.")
        pyramid = [data_levels[start_level]]
        new_scales = [tuple(metadata["scales"][start_level])]

    else:
        print("Resolution layer not available: will compute.")

        base_downscale = factor_power(factors, start_level)

        current = pad_to_divisible(
            data_levels[0],
            base_downscale,
            spatial_axes=spatial_axes,
        )
        down = downsample_nn(
            current,
            base_downscale,
            spatial_axes=spatial_axes,
        )

        pyramid = [down]
        new_scales = [
            multiply_scales(metadata["scales"][0], base_downscale)
        ]

    print("Creating pyramid.")

    for _ in range(1, num_levels):
        current = pad_to_divisible(
            pyramid[-1],
            factors,
            spatial_axes=spatial_axes,
        )
        down = downsample_nn(
            current,
            factors,
            spatial_axes=spatial_axes,
        )
        pyramid.append(down)

    print("Updating metadata.")

    for level in range(1, num_levels):
        scale_factor = factor_power(factors, level)
        new_scales.append(
            multiply_scales(new_scales[0], scale_factor)
        )

    metadata = dict(metadata)
    metadata["scales"] = new_scales
    metadata["size"] = [tuple(level.shape) for level in pyramid]
    metadata["chunksize"] = [
        tuple(level.chunksize) for level in pyramid
    ]

    return pyramid, metadata