# PyMIF — microscopy I/O, OME-Zarr conversion, and NGFF utilities

**PyMIF** is a Python package for reading microscopy datasets from multiple acquisition systems, building multiscale pyramids, writing **OME-NGFF / OME-Zarr**, and interacting with those datasets from Python, the command line, and napari.

It is developed for users of the [Mesoscopic Imaging Facility (MIF)](https://www.embl.org/groups/mesoscopic-imaging-facility/), but the repository now covers a broader scope than simple vendor import: it includes a reusable manager API, NGFF-aware zarr creation utilities, region writing for images and labels, batch conversion helpers, and napari widgets for conversion and overview generation.

For the rendered docs, see [the documentation page](https://grinic.github.io/pymif/).

> [!NOTE]
> Current PyMIF releases write **NGFF v0.5 / Zarr v3** by default. Existing **NGFF v0.4 / Zarr v2** datasets remain supported through `ZarrManager` and `ZarrV04Manager`.

![Demo](documentation/demo.gif)

*Demonstration of PyMIF usage. Data: near newborn mouse embryo (~1.5 cm long). Fluorescence signal: methylene blue + autofluorescence. Sample processed and imaged by Montserrat Coll at the Mesoscopic Imaging Facility. Video speed: 2.5× real speed.*

---

## Current repository scope

PyMIF currently contains five main pieces:

1. **Microscope managers** for reading datasets and normalizing them to a common API.
2. **OME-Zarr / NGFF writing utilities** for full dataset export, empty dataset creation, subgroup creation, and region updates.
3. **Pyramid and subsetting helpers** for multiscale generation and dataset cropping.
4. **CLI tools** for one-off and batch conversion to zarr.
5. **Napari widgets** for interactive conversion, ROI selection, and overview generation.

### Supported data sources

The main reader classes currently exposed by `pymif.microscope_manager` are:

- `ArrayManager` — wrap an in-memory NumPy or Dask array using PyMIF metadata conventions.
- `LuxendoManager` — Luxendo XML + HDF5 datasets.
- `OperaManager` — Opera Phenix / Opera PE OME-TIFF style datasets.
- `ScapeManager` — Leica SCAPE OME-TIFF + XLIF datasets.
- `ViventisManager` — Viventis LS1 datasets.
- `ZeissManager` — Zeiss CZI datasets.
- `ZarrManager` — NGFF v0.4/v0.5 OME-Zarr datasets.
- `ZarrV04Manager` — compatibility reader for older v0.4-style datasets.

### Core capabilities

- Read vendor-specific microscopy metadata into a shared metadata schema.
- Represent image data lazily with Dask.
- Build multiscale pyramids from a base-resolution dataset.
- Write OME-Zarr in **NGFF v0.4/Zarr v2** or **NGFF v0.5/Zarr v3** form.
- Create empty image groups and label groups inside an existing zarr hierarchy, including 2D/3D/4D/5D axis-aware zarr datasets.
- Write image patches or label patches back into an existing zarr dataset.
- Visualize datasets in napari.
- Convert single datasets or CSV-defined batches from the CLI.

---

## Installation

A clean conda environment is recommended:

```console
conda create -n pymif python=3.12
conda activate pymif
```

Then install from the repository:

```console
git clone https://github.com/grinic/pymif.git
cd pymif
pip install .
```

For development work:

```console
pip install -e .
```

To use the napari widgets as well:

```console
pip install -e .[napari]
```

---

## Quick usage

### Python API

```python
import pymif.microscope_manager as mm

# Read a source dataset
source = mm.ViventisManager("path/to/Position_1")

# Build a pyramid in memory
source.build_pyramid(num_levels=3)

# Export to OME-Zarr (default: NGFF v0.5 / zarr v3)
source.to_zarr("output.zarr")

# Re-open the written zarr dataset
z = mm.ZarrManager("output.zarr")
viewer = z.visualize(start_level=0, in_memory=False)
```

### Create an empty zarr dataset from metadata

```python
import pymif.microscope_manager as mm

z = mm.ZarrManager(
    "empty.zarr",
    mode="a",
    metadata={
        "size": [(1, 2, 16, 256, 256)],
        "chunksize": [(1, 1, 16, 128, 128)],
        "scales": [(2.0, 0.5, 0.5)],
        "units": ("micrometer", "micrometer", "micrometer"),
        "axes": "tczyx",
        "channel_names": ["GFP", "RFP"],
        "channel_colors": ["00FF00", "FF0000"],
        "time_increment": 1.0,
        "time_increment_unit": "second",
        "dtype": "uint16",
    },
)
```

### Axis-aware ZarrManager metadata

Most microscope-specific managers still normalize data to legacy `tczyx`, but `ZarrManager` and `ArrayManager` can now write and read any unique subset of the image axes `t`, `c`, `z`, `y`, and `x`. The axes string must have one label per array dimension and may not contain non-image axes such as tiles, ROIs, scenes, or wells.

For example, a two-dimensional YX intensity image can be written as:

```python
import dask.array as da
import numpy as np
import pymif.microscope_manager as mm

img = np.zeros((512, 512), dtype=np.uint16)
levels = [da.from_array(img, chunks=(256, 256))]
metadata = {
    "axes": "yx",
    "size": [(512, 512)],
    "chunksize": [(256, 256)],
    "scales": [(0.5, 0.5)],       # follows the spatial axes in axes order
    "units": ("micrometer", "micrometer"),
    "dtype": "uint16",
    "data_type": "intensity",    # or "label"
}

mm.ArrayManager(levels, metadata).to_zarr("yx_image.zarr")
```

Use `data_type="label"` for segmentation, mask, or annotation data. Label arrays must use an integer dtype. For NGFF v0.5/Zarr v3, PyMIF stores the semantic type under `attrs["ome"]["data_type"]`, writes array-level `dimension_names`, writes `multiscales[0]["type"] = "label"`, and adds `image-label` metadata. For NGFF v0.4/Zarr v2, the equivalent metadata is stored directly on the group attributes.

Legacy calls that create labels from an intensity image metadata dictionary still work:

```python
z = mm.ZarrManager("image.zarr", mode="a")
z.create_empty_group("nuclei", image_metadata, is_label=True)
```

When `image_metadata["axes"] == "tczyx"`, this legacy `is_label=True` form creates a `tzyx` label group by dropping the intensity channel axis. New callers that really want channelled label data can pass explicit label metadata instead:

```python
label_metadata = {**image_metadata, "data_type": "label"}
z.create_empty_group("classes", label_metadata, data_type="label")
```

### Update a region in an existing zarr image

```python
import numpy as np
import pymif.microscope_manager as mm

z = mm.ZarrManager("output.zarr", mode="a")
patch = np.full((1, 1, 2, 64, 64), 999, dtype=np.uint16)

z.write_image_region(
    patch,
    t=slice(0, 1),
    c=slice(0, 1),
    z=slice(10, 12),
    y=slice(100, 164),
    x=slice(100, 164),
    level=0,
)
```

---

## Examples

The `examples/` directory includes axis-aware zarr examples:

- `examples/example_zarrmanager_axis_aware.py` — script showing YX intensity writing, YX label metadata, subsetting, and empty label-group creation.
- `examples/example_zarrmanager_axis_aware.ipynb` — notebook version of the same workflow.

## CLI

Single conversion:

```console
pymif 2zarr -i INPUT_PATH -m MICROSCOPE -z OUTPUT_ZARR
```

Batch conversion from a CSV manifest:

```console
pymif batch2zarr -i INPUT_FILE.csv
```

Get help:

```console
pymif -h
pymif 2zarr -h
pymif batch2zarr -h
```

---

## Napari plugin

PyMIF provides napari widgets for conversion and overview generation. After installing the napari extras, the conversion widget is available from:

`Plugins > PyMIF > Converter Plugin`

The widget can load data, preview channels, define a 3D ROI, restrict z/time/channel ranges, choose pyramid settings, and export to OME-Zarr. For axis-aware zarr datasets, controls tied to missing axes are disabled; for example, a dataset with `axes="yx"` has no active T slider or channel selector.

![napari-demo](documentation/napari-demo.png)

---

## Documentation strategy in this repository

PyMIF uses **Sphinx + MyST + AutoAPI**. In practice this means:

- user-facing project scope belongs in `README.md` and `doc/README.md`
- API pages are generated automatically from Python docstrings
- documenting classes, methods, and helper functions directly in the source code is the best way to improve the docs

The most important API entry points to document and keep stable are:

- `MicroscopeManager`
- vendor reader classes in `pymif.microscope_manager`
- `ZarrManager`
- zarr-writing helpers in `pymif.microscope_manager.utils`
- CLI entry points in `pymif.cli`
- napari widgets in `pymif.napari`

---

## Contributing and extending PyMIF

New microscope support is typically added by subclassing `MicroscopeManager` and implementing `read()` so that it returns:

```python
Tuple[List[dask.array.Array], Dict[str, Any]]
```

The returned metadata should follow the PyMIF schema used across the repository, including:

```python
{
  "size": [... per pyramid level ...],
  "chunksize": [... per pyramid level ...],
  "scales": [... per pyramid level ...],
  "units": (...),
  "axes": "tczyx",        # ZarrManager/ArrayManager also accept yx, zyx, tzyx, cyx, etc.
  "data_type": "intensity", # or "label" for integer label data
  "channel_names": [...],    # required only when a c axis is present
  "channel_colors": [...],
  "time_increment": ...,     # required only when a t axis is present
  "time_increment_unit": ...,
  "dtype": ...,
}
```

For axis-aware zarr data, `scales` and `units` contain only the spatial axes present in `axes`, in that same order. For example, `axes="yx"` uses two scale values per pyramid level, while `axes="tczyx"` uses three spatial scale values for `z`, `y`, and `x`. Once that contract is respected, the new manager automatically benefits from the common PyMIF tooling such as `build_pyramid()`, `to_zarr()`, `visualize()`, `reorder_channels()`, `update_metadata()`, and `subset_dataset()`.
