# PyMIF — Python code for users of the Mesoscopic Imaging Facility

**PyMIF** is a modular Python package to read, visualize, and write multiscale (pyramidal) microscopy image data from a variety of microscope platforms available at the [Mesoscopic Imaging Facility (MIF)](https://www.embl.org/groups/mesoscopic-imaging-facility/) into the [OME-NGFF (Zarr)](https://ngff.openmicroscopy.org/) format.

For more information, see [the documentation page](https://grinic.github.io/pymif/).

![Demo](documentation/demo.gif)
*Demonstration of pymif usage. Data: near newborn mouse embryo (~1.5 cm long). Fluorescence signal: methylene blue + autofluorescence. Sample processed and imaged by Montserrat Coll at the Mesoscopic Imaging Facility. Video speed: 2.5X real speed.*

---

## 🚀 Getting Started

### 📥 Installation

It is recommended to install pymif in a clean conda environment:

```bash
conda create -n pymif python=3.10
conda activate pymif
```

Installation is then done by cloning the repository:

```bash
git clone https://github.com/grinic/pymif.git
cd pymif
python -m pip install -e .
```

### 📚 Minimal Example Usage

```python
import pymif.microscope_manager as mm

dataset = mm.ViventisManager("path/to/Position_1")
dataset.build_pyramid(num_levels=3)
dataset.write("output.zarr")
dataset_zarr = mm.ZarrManager("output.zarr")
viewer = dataset_zarr.visualize(start_level=0, in_memory=False)
```

For more examples, see [examples](examples/).
