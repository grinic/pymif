# 🧠 pymif — Python Microscope Image Framework

**pymif** is a modular Python package to read, visualize, and write multiscale (pyramidal) microscopy image data from a variety of microscope platforms into the [OME-NGFF (Zarr)](https://ngff.openmicroscopy.org/) format.

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
│ ├── build_pyramid.py
│ ├── visualize.py
│ └── write.py
├── examples/
│ └── example_viventis.ipynb
├── tests/
│ ├── test_viventis_manager.py
│ ├── test_zarr_manager.py
│ └── test_luxendo_manager.py
├── requirements.txt
└── setup.py
```


---

## 🚀 Getting Started

### 📥 Installation

```bash
git clone https://github.com/<your-username>/pymif.git
cd pymif
pip install -e .
```

Install visualization dependencies

```bash
pip install napari[all] napari-ome-zarr
```

### 📚 Example Usage

```python
from microscope_manager.viventis_manager import ViventisManager

manager = ViventisManager("path/to/Position_1")
manager.build_pyramid(num_levels=3)
manager.write("output.zarr")
viewer = manager.visualize(start_level=0, in_memory=False)
viewer.window.show()
```

For more examples, see [examples](https://github.com/grinic/pymif/tree/main/pymif/examples).

### 🧪 Running Tests

```bash
pytest tests/
```

### ➕ Adding New Microscope Support

- To add a new format:

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
