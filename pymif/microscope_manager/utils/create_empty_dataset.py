from typing import Dict, Any
import dask.array as da
import zarr

def create_empty_dataset(
            root: zarr.Group,
            metadata: Dict[str, Any],
):
    """
    Create an empty OME-Zarr dataset using provided metadata.
    """
    if not metadata:
        raise ValueError("Metadata is required to create an empty dataset.")

    sizes = metadata["size"]  # list of shapes per level
    chunks = metadata["chunksize"]
    dtype = metadata.get("dtype", "uint16")

    # Create a pyramid of empty arrays
    pyramid = []
    for shape, chunk in zip( sizes, chunks ):
        arr = da.zeros(shape, chunks=chunk, dtype=dtype)
        pyramid.append(arr)

    from .write_to_zarr import write_to_zarr as _write_to_zarr
    _write_to_zarr(
        root,
        data_levels=pyramid,
        metadata=metadata,
        compressor =None,
        compressor_level =3,
        overwrite =True,
        parallelize =False,
    )

    print(f"[INFO] Created empty OME-Zarr dataset at {root.store.path}.")
