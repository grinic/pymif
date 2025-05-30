# pymif — Python code for users of the Mesoscopic Imaging Facility

**pymif** is a modular Python package to read, visualize, and write multiscale (pyramidal) microscopy image data from a variety of microscope platforms available at the [Mesoscopic Imaging Facility (MIF)](https://www.embl.org/groups/mesoscopic-imaging-facility/) into the [OME-NGFF (Zarr)](https://ngff.openmicroscopy.org/) format.

---

## 📦 Features

- ✅ Read and parse image metadata from multiple microscope vendors:
  - **Viventis** (`.ome + .tif`)
  - **Luxendo** (`.xml + .h5`)
  - **Generic OME-Zarr**
- ✅ Abstract base class `MicroscopeManager` ensures uniform interface for all readers
- ✅ Lazy loading via Dask for memory-efficient processing
- ✅ Build pyramidal (multiscale) OME-Zarr archives from raw data or existing pyramids
- ✅ Write OME-Zarr with:
  - Blosc compression
  - Nested directory layout
  - Full NGFF + OMERO metadata (channel names, colors, scales, units)
- ✅ Visualize pyramids in **Napari** using `napari-ome-zarr` plugin
- ✅ Compatible with automated workflows and interactive exploration (Jupyter + scripts)

---

## 🗂️ Project Structure

```
pymif/
├── microscope_manager/
│ ├── viventis_manager.py
│ ├── luxendo_manager.py
│ ├── zarr_manager.py
│ ├── microscope_manager.py
│ └── utils/
│  ├── pyramid.py
│  ├── visualize.py
│  └── write.py
├── examples/
| ├── example_luxendo.ipynb
│ └── example_viventis.ipynb
├── tests/
│ ├── test_viventis_manager.py
│ ├── test_zarr_manager.py
│ └── test_luxendo_manager.py
├── requirements.txt
├── setup.py
└── README.md
```


---

## 🚀 Getting Started

### 📥 Installation

It is recommended to install pymif in a clean conda environment:

```bash
conda create -n pymif python=3.10
conda activate pymif
```

Installation is then done clining the repository:

```bash
git clone https://github.com/grinic/pymif.git
cd pymif
pip install -e .
```

### 📚 Example Usage

```python
from pymif.microscope_manager.viventis_manager import ViventisManager

dataset = ViventisManager("path/to/Position_1")
dataset.build_pyramid(num_levels=3)
dataset.write("output.zarr")
viewer = dataset.visualize(start_level=0, in_memory=False)
```

For more examples, see [examples](examples/).

### 🧪 Running Tests

```bash
pytest tests/
```

### ➕ Adding New Microscope Support

To add a new format:

- Subclass MicroscopeManager

- Implement read() returning:

```python
Tuple[List[dask.array], Dict[str, Any]]
```

- Follow this metadata schema:

```python
{
  "size": [... per level ...],
  "scales": [...],
  "units": (...),
  "axes": "tczyx",
  "channel_names": [...],
  "channel_colors": [...],
  "time_increment": ...,
  ...
}
```

You will automatically inherit `build_pyramid()`, `write()` and `visualize()`.
