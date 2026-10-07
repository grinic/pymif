import os, re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Any
import dask.array as da
import h5py
import numpy as np
from .microscope_manager import MicroscopeManager
import itertools

class LuxendoManager(MicroscopeManager):
    """
    Reader for Luxendo microscope data saved as multi-resolution HDF5 (.lux.h5) and XML metadata.

    This class parses Luxendo's XML configuration and builds a lazy Dask array pyramid for downstream processing.
    """
        
    def __init__(self, 
                 path: str,
                 chunks: Tuple[int, ...] = (1, 1, 8, 4096, 4096)):
        """
        Initialize the LuxendoManager.

        Parameters
        ----------
        path : str
            Path to the Luxendo dataset directory.
        chunks : Tuple[int, ...], optional
            Chunk shape for Dask arrays, by default (1, 1, 8, 4096, 4096).
        """
        
        super().__init__()
        self.path = Path(path)
        self.chunks = chunks
        self._open_files = []
        self._h5_handles: Dict[Path, h5py.File] = {}
        self.read()

    def _parse_metadata(self) -> Dict[str, Any]:
        """
        Parse XML metadata from the Luxendo dataset.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing dataset shape, voxel sizes, channel info, and other metadata.
        """
        
        xml_path = next(self.path.glob("*.xml"))
        tree = ET.parse(xml_path)
        root = tree.getroot()

        setups = root.findall(".//ViewSetup")
        timepoints = root.find(".//Timepoints")
        first_tp = int(timepoints.find("first").text)
        last_tp = int(timepoints.find("last").text)
        size_t = last_tp - first_tp + 1

        setup_sizes = {}
        setup_voxels = {}
        channel_names = []
        channel_ids = []

        for setup in setups:
            setup_id = int(setup.find("id").text)
            size = tuple(map(int, setup.find("size").text.split()))
            size = (size[2],size[1],size[0]) # invert XYZ -> ZYX
            voxel = tuple(map(float, setup.find("voxelSize/size").text.split()))
            voxel = (voxel[2],voxel[1],voxel[0]) # invert XYZ -> ZYX
            setup_sizes[setup_id] = size
            setup_voxels[setup_id] = voxel

        # Assume sorted by id
        channels = root.findall(".//Channel")
        for ch in channels:
            channel_ids.append(int(ch.find("id").text))
            channel_names.append(ch.find("name").text)

        size_c = len(channel_ids)
        size_z, size_y, size_x = setup_sizes[0]  # all setups have same size
        scales = [setup_voxels[0]]
        units = ["micrometer"] * 3  # consistent with metadata
        
        # Gather HDF5 files and dataset names
        h5_files = sorted(self.path.glob("*.lux.h5"))
        # Open the reference file once for names and shapes of all levels
        with h5py.File(h5_files[0], "r") as f:
            dataset_names = self._sorted_dataset_names(f)
            size = [(size_t, size_c) + f[ds_name].shape for ds_name in dataset_names]
        
        for name in dataset_names[1:]:
            downscale_factors = list(map(int, re.findall(r'\d+', name)))
            
            scales.append(tuple([
                scales[0][0] * downscale_factors[0],
                scales[0][1] * downscale_factors[1],
                scales[0][2] * downscale_factors[2],
            ]))
            
        palette = ['white', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan']
        channel_colors = list(itertools.islice(itertools.cycle(palette), len(channel_names)))

        return {
            "size": size,
            "scales": scales,
            "units": tuple(units),
            "time_increment": 1.0,
            "time_increment_unit": "s",
            "channel_names": channel_names,
            "channel_colors": channel_colors,  # Example, map from name if needed
            "dtype": "uint16",
            "plane_files": None,
            "axes": "tczyx"
        }
        
    def _read_h5_stack(self, h5_path: Path, 
                       dataset_name: str) -> np.ndarray:
        """
        Load a single resolution dataset lazily as a Dask array.

        Parameters
        ----------
        h5_path : Path
            Path to the .lux.h5 file.
        dataset_name : str
            Internal dataset name (e.g., "Data", "Data444", etc.)

        Returns
        -------
        np.ndarray
            A Dask array loaded from the HDF5 file.
        """
        
        f = self._h5_handles.get(h5_path)
        if f is None:  # one handle per file, shared by all pyramid levels
            f = self._h5_handles[h5_path] = h5py.File(h5_path, "r")
            self._open_files.append(f)
        # return dask array, no readings yet
        return da.from_array(f[dataset_name], chunks=self.chunks[2:]) 

    def _read_h5_shape(self, h5_path: Path, dataset_name: str):
        """
        Read the shape and dtype of a dataset in an HDF5 file.

        Parameters
        ----------
        h5_path : Path
            Path to the .lux.h5 file.
        dataset_name : str
            Internal dataset name.

        Returns
        -------
        Tuple[Tuple[int, ...], np.dtype]
            A tuple containing the dataset shape and dtype.
        """
        
        with h5py.File(h5_path, "r") as f:
            return f[dataset_name].shape, f[dataset_name].dtype
        
    def get_available_datasets(self, h5_file) -> List:
        """
        Extract all dataset names from a .lux.h5 file.

        Parameters
        ----------
        h5_file : Path
            Path to a Luxendo HDF5 file.

        Returns
        -------
        List[str]
            Sorted list of dataset names.
        """
        
        with h5py.File(h5_file, "r") as f:
            return self._sorted_dataset_names(f)

    @staticmethod
    def _sorted_dataset_names(f: h5py.File) -> List[str]:
        """Return the "Data*" dataset names of an open file in natural scale order."""
        return sorted((k for k in f.keys() if k.startswith("Data")), key=lambda s: (len(s), s))

    def _build_dask_array(self) -> List[da.Array]:
        """
        Construct a multiscale image pyramid as Dask arrays.

        Returns
        -------
        List[da.Array]
            A list of Dask arrays representing each resolution level (from highest to lowest).
        """
        
        t, c, z, y, x = self.metadata["size"][0]
        
        def extract_tp_ch_numbers(filename: str) -> tuple[int, int]:
            tp_match = re.search(r'tp-(\d+)', filename)
            ch_match = re.search(r'ch-(\d+)', filename)
            tp = int(tp_match.group(1)) if tp_match else -1
            ch = int(ch_match.group(1)) if ch_match else -1
            return tp, ch
        
        h5_files = sorted(self.path.glob("*.lux.h5"), 
                          key=lambda f: extract_tp_ch_numbers(f.name)
                          )
        assert len(h5_files) == t * c, "Mismatch between expected and found HDF5 files."

        dataset_names = self.get_available_datasets(h5_files[0])

        pyramid = []
        for ds_name in dataset_names:
            lazy_arrays = [
                [self._read_h5_stack(h5_files[ti * c + ci], ds_name) for ci in range(c)]
                for ti in range(t)
            ]
            pyramid.append(da.stack(lazy_arrays, axis=0).rechunk(self.chunks))  # T, C, Z, Y, X

        return pyramid

    def read(self) -> Tuple[List[da.Array], Dict[str, Any]]:
        """
        Read Luxendo image data and metadata.

        Returns
        -------
        Tuple[List[da.Array], Dict[str, Any]]
            A list of Dask arrays (pyramidal levels) and a metadata dictionary.
        """
        
        self.metadata = self._parse_metadata()
        self.data = self._build_dask_array()
        return
    
