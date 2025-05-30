# 🧪 zarr_converter

A Python package to **convert microscopy image files** (TIFF, CZI, LIF, H5) into **pyramidal OME-Zarr** format with full metadata support, using `aicsimageio`, `ome-types`, and `ome-zarr`.

---

## ✅ Features

- 📂 Input formats: **TIF**, **CZI**, **LIF**, **H5**
- 🧱 Pyramidal OME-Zarr output (multiscale)
- ⚡ Scalable: powered by **Dask**
- 🧬 Metadata: pixel sizes, channel names, display colors
- 🔧 CLI and Python API
- 🖼️ Optional viewer integration with **Napari**

---

## 📦 Installation

Clone the repository and install the package:

```bash
git clone https://github.com/yourname/zarr_converter.git
cd zarr_converter
pip install .
```

Install dependencies directly:

```bash
pip install -r requirements.txt
```

---

## 🖥️ CLI Usage

Run directly from terminal:

```bash
zarr-convert INPUT_FILE OUTPUT_ZARR [OPTIONS]
```

### Example:

```bash
zarr-convert ./data/sample.tif ./output/sample.zarr \
  --pixel-size-xy 0.1625 \
  --pixel-size-z 0.5
```

### CLI Options:

| Option               | Description                                    |
|----------------------|------------------------------------------------|
| `--pixel-size-xy`    | XY pixel size in microns (optional)            |
| `--pixel-size-z`     | Z spacing in microns (optional)                |
| `--downscale-factor` | Downsampling factor per level (default: 2)     |
| `--min-dim`          | Minimum size to stop pyramid creation (default: 64) |

---

## 🧑‍💻 Python API

You can also use `zarr_converter` programmatically:

```python
from zarr_converter import ZarrConverter

converter = ZarrConverter("path/to/image.czi")
converter.save_as_ome_zarr(
    "output/sample.zarr",
    pixel_size_xy=0.25,
    pixel_size_z=1.0,
    channel_names=["DAPI", "FITC", "TRITC"],
    channel_colors=["#0000FF", "#00FF00", "#FF0000"]
)
```

---

## 📁 Example File Structure

```text
zarr_converter/
├── __init__.py
├── core.py         # Main logic for reading and writing
├── cli.py          # CLI wrapper using click
requirements.txt    # List of all dependencies
setup.py            # Python packaging script
README.md           # This file
```

---

## 🧪 Sample Viewing

### Using [Napari](https://napari.org/):

```bash
napari path/to/output.zarr
```

### In Python (Dask/Zarr):

```python
import dask.array as da
import zarr

z = zarr.open("output.zarr", mode="r")
pyramid_levels = [da.from_zarr(z[level]) for level in z.group_keys()]
```

---

## 🔧 Example `requirements.txt`

```text
aicsimageio>=4.12.1
ome-types>=0.6.1
ome-zarr>=0.9.0
dask>=2024.4.1
zarr>=2.16.1
scikit-image>=0.22.0
scikit-learn>=1.4.2
numpy>=1.26.4
pandas>=2.2.2
seaborn>=0.13.2
matplotlib>=3.8.4
tifffile>=2024.5.10
readlif>=0.6.5
keras>=3.3.3
tensorflow>=2.16.1
numba>=0.59.1
openpyxl>=3.1.2
pyimagej>=1.4.1
PyOpenGL>=3.1.7
scipy>=1.13.0
xarray>=2024.5.0
xlsxwriter>=3.2.0
napari[all]>=0.4.18
click>=8.1.7
```

---

## ⚙️ Example `setup.py`

```python
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="zarr_converter",
    version="0.1.0",
    description="Convert microscopy images (TIFF, CZI, LIF, H5) to pyramidal OME-Zarr with metadata.",
    author="Your Name",
    author_email="you@example.com",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "zarr-convert=zarr_converter.cli:convert"
        ]
    },
    python_requires=">=3.8",
)
```

---

## 📄 License

MIT License © Your Name

---

## 🙋‍♀️ Contributing

- Issues and PRs welcome!
- Please include tests and clear commit messages.

---

## 🔗 Useful Links

- [OME-NGFF (OME-Zarr format)](https://ngff.openmicroscopy.org/latest/)
- [AICSImageIO documentation](https://allencell.github.io/aicsimageio/)
- [OME Types](https://pypi.org/project/ome-types/)
- [Napari Viewer](https://napari.org/)