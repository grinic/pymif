import os, re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Any
import dask.array as da
from dask import delayed
import h5py
import numpy as np
from .microscope_manager import MicroscopeManager

class LuxendoManager(MicroscopeManager):
    """
    Reader for Luxendo microscope data saved as multi-resolution HDF5 (.lux.h5) and XML metadata.

    This class parses Luxendo's XML configuration and builds a lazy Dask array pyramid for downstream processing.
    """
        
    def __init__(self, 
                 path: str,
                 chunks: Tuple[int, ...] = (1, 1, 16, 256, 256)):
        """
        Initialize the LuxendoManager.

        Parameters
        ----------
        path : str
            Path to the Luxendo dataset directory.
        chunks : Tuple[int, ...], optional
            Chunk shape for Dask arrays, by default (1, 1, 16, 256, 256).
        """
        
        super().__init__()
        self.path = Path(path)
        self.chunks = chunks
        self._open_files = []
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
        dataset_names = self.get_available_datasets(h5_files[0])
        
        size = [(size_t, size_c) + self._read_h5_shape(h5_files[0], ds_name)[0] for ds_name in dataset_names]
        
        for name in dataset_names[1:]:
            downscale_factors = list(map(int, re.findall(r'\d+', name)))
            
            scales.append(tuple([
                scales[0][0] * downscale_factors[0],
                scales[0][1] * downscale_factors[1],
                scales[0][2] * downscale_factors[2],
            ]))

        return {
            "size": size,
            "scales": scales,
            "units": tuple(units),
            "time_increment": 1.0,
            "time_increment_unit": "s",
            "channel_names": channel_names,
            "channel_colors": [0xFF0000, 0x0000FF],  # Example, map from name if needed
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
        
        f = h5py.File(h5_path, "r")
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
            dataset_names = [k for k in f.keys() if k.startswith("Data")]
        dataset_names = sorted(dataset_names, key=lambda s: (len(s), s))  # natural scale order
        return dataset_names

    def _build_dask_array(self) -> List[da.Array]:
        """
        Construct a multiscale image pyramid as Dask arrays.

        Returns
        -------
        List[da.Array]
            A list of Dask arrays representing each resolution level (from highest to lowest).
        """
        
        t, c, z, y, x = self.metadata["size"][0]
        
        h5_files = sorted(self.path.glob("*.lux.h5"))
        assert len(h5_files) == t * c, "Mismatch between expected and found HDF5 files."

        dataset_names = self.get_available_datasets(h5_files[0])

        pyramid = []
        for ds_name in dataset_names:
            lazy_arrays = []
            for ti in range(t):
                row = []
                for ci in range(c):
                    index = ti * c + ci
                    h5_path = h5_files[index]
                    # shape, dtype = self._read_h5_shape(h5_path, ds_name)
                    delayed_arr = self._read_h5_stack(h5_path, ds_name)
                    row.append(delayed_arr)
                lazy_arrays.append(row)

            dask_stack = da.stack([[arr for arr in row] for row in lazy_arrays], axis=0)  # T, C, Z, Y, X
            dask_stack = dask_stack.rechunk(self.chunks)
            pyramid.append(dask_stack)
            
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
        return (self.data, self.metadata)
    
