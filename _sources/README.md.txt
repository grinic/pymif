# PyMIF — Python code for users of the Mesoscopic Imaging Facility

**PyMIF** (source code [here](https://github.com/grinic/pymif)) is a modular Python package to read, visualize, and write multiscale (pyramidal) microscopy image data from a variety of microscope platforms available at the [Mesoscopic Imaging Facility (MIF)](https://www.embl.org/groups/mesoscopic-imaging-facility/) into the [OME-NGFF (Zarr)](https://ngff.openmicroscopy.org/) format.

**NOTE**: As of v0.3.0, PyMIF follows NGFF v0.5 standards. Datasets created with older version of PyMIF (e.g. 0.2.4) can still be loaded using the manager `ZarrV04Manager` as shown in [examples](https://github.com/grinic/pymif/tree/main/examples).

---

## 📦 Features

- ✅ Read and parse image metadata from multiple microscope vendors and data formats:
  - **Viventis** (`.ome + .tif`)
  - **Luxendo** (`.xml + .h5`)
  - **Opera PE** (`.ome.tiff`)
  - **Zeiss** (`.czi`)
  - **Generic OME-Zarr**
  - **Numpy or Dask array**
- ✅ Abstract base class `MicroscopeManager` ensures uniform interface for all readers
- ✅ Lazy loading via Dask for memory-efficient processing
- ✅ Build pyramidal (multiscale) OME-Zarr archives from raw data or existing pyramids
- ✅ Write OME-Zarr with:
  - Blosc or GZIP compression
  - Nested directory layout
  - Full NGFF + OMERO metadata (channel names, colors, scales, units)
  - Optional parallelization with `dask-distribute`
- ✅ Visualize pyramids in **Napari** using `napari-ome-zarr` plugin:
  - Using lazy loading for fast visualization, or
  - Using *in-memory* loading of any resolution layer for interactivity.
- ✅ Compatible with automated workflows and interactive exploration (Jupyter + scripts)

---

## 🗂️ Project Structure

```
pymif/
├── pymif
│ └── microscope_manager
│   ├── luxendo_manager.py
│   ├── viventis_manager.py
│   ├── opera_manager.py
│   ├── zeiss_manager.py
│   ├── zarr_manager.py
│   ├── zarr_v04_manager.py
│   ├── array_manager.py
│   ├── microscope_manager.py
│   └── utils/
│    ├── pyramid.py
│    ├── visualize.py
│    ├── add_labels.py
│    ├── subset.py
│    ├── to_zarr.py
│    ├── write_image_region.py
│    ├── write_label_region.py
│    ├── create_empty_dataset.py
│    └── create_empty_group.py
│
├── examples/
| ├── example_luxendo.ipynb
| ├── example_viventis.ipynb
| ├── example_opera.ipynb
| ├── example_zeiss.ipynb
| ├── example_zarr.ipynb
| ├── example_array.ipynb
│ └── ...
│
├── tests/
│ └── ...
│
├── requirements.txt
├── setup.py
└── README.md
```


---

## 🚀 Getting Started

### 📥 Installation

It is recommended to install pymif in a clean conda environment:

```bash
conda create -n pymif python=3.12
conda activate pymif
```

Installation is then done by cloning the repository:

```bash
git clone https://github.com/grinic/pymif.git
cd pymif
python -m pip install .
```

**NOTE**: Use the `-e` option if you want to use the download as installation folder.

### 📚 Example Usage

With the following code, we read Viventis image data and parse the corresponding metadata. Next, we build a pyramidal structure of 3 resolution layers and save it into an OME-Zarr format. Finally, we load the new dataset and visualize it in napari.

```python
import pymif.microscope_manager as mm

dataset = mm.ViventisManager("path/to/Position_1")
dataset.build_pyramid(num_levels=3)
dataset.to_zarr("output.zarr")
dataset_zarr = mm.ZarrManager("output.zarr")
viewer = dataset_zarr.visualize(start_level=0, in_memory=False)
```

![Demo](../documentation/demo.gif)
*Demonstration of pymif usage. Data: near newborn mouse embryo (~1.5 cm long). Fluorescence signal: methylene blue + autofluorescence. Sample processed and imaged by Montserrat Coll at the Mesoscopic Imaging Facility. Video speed: 2.5X real speed.*


For more examples, see [examples](https://github.com/grinic/pymif/tree/main/examples).

### 🧪 Running Tests

```bash
pytest tests/
```

### ➕ Adding New Microscope Support and Contributing

Contributions/PRs are welcome! If you would like to help and add a new format:

- Subclass MicroscopeManager

- Implement read() returning:

```python
Tuple[List[dask.array], Dict[str, Any]]
```

- Follow this metadata schema:

```python
{
  "size": [... per level ...],
  "scales": [... per level ...],
  "units": (...),
  "axes": "tczyx",
  "channel_names": [...],
  "channel_colors": [...],
  "time_increment": ...,
  "time_increment_unit": ...,
  ...
}
```

You will automatically inherit all `MicroscopeManager` methods, including:
- `build_pyramid()`, 
- `to_zarr()`, 
- `visualize()`,
- `reorder_channels()`,
- `update_metadata()`,
- ...
