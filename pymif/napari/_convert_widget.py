import os
import napari
from napari import current_viewer
import numpy as np
from magicgui import magicgui
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import pymif.microscope_manager as mm
from magicgui.widgets import FileEdit
from PyQt5.QtWidgets import QFileDialog, QWidget, QVBoxLayout
from matplotlib import rc
rc('font', size=12)
rc('font', family='Arial')
# rc('font', serif='Times')
rc('pdf', fonttype=42)
# rc('text', usetex=True)

# -------------------------
# MagicGUI controls
# -------------------------

def convert_widget():
    viewer = current_viewer()
    _state = {"dataset": None}

    @magicgui(
        call_button="Convert to zarr",
        output_path={"widget_type": "FileEdit", "mode": "d"},
    )
    def make_convert_widget(
        output_path: FileEdit = None,
    ):
        dataset = _state["dataset"]
        if dataset is None:
            return

        dataset.to_zarr(output_path)

    @magicgui(
        call_button="Visualize in napari",
        input_path={"widget_type": "FileEdit", "mode": "d"},
        scene_index={"widget_type": "SpinBox", "min": 0, "max": 100, "step": 1},
        file_format={"choices": ["viventis", "opera", "luxendo", "zarrV05", "zarrV04", "zeiss"]},
    )
    def make_visualize_widget(
        file_format="zarrV05",
        scene_index=0,
        input_path: FileEdit = None,
    ):
        dataset = _state["dataset"]
        if dataset is None:
            return

        viewer.layers.clear()

        if make_visualize_widget.file_format.value in ["zarrV05", "zarrV04"]:
            viewer.open(make_visualize_widget.input_path.value, plugin='napari-ome-zarr')
        else:
            dataset = _state["dataset"]
            dataset.visualize(
                viewer=viewer
            )

    # -------------------------
    # Input path callback
    # -------------------------
    @make_visualize_widget.file_format.changed.connect
    def _update_mode(file_format):
        if file_format == "zeiss": 
            make_visualize_widget.scene_index.enabled = True 
        else: 
            make_visualize_widget.scene_index.enabled = False
        
        if file_format in ["zarrV05", "zarrV04", "viventis", "luxendo"]:
            make_visualize_widget.input_path.mode = "d" 
        else: 
            make_visualize_widget.input_path.mode = "r"


    @make_visualize_widget.input_path.changed.connect
    def _on_input_path_change(path):
        if not path:
            return

        if make_visualize_widget.file_format.value == "zeiss":
            dataset = mm.ZeissManager(path, scene_index=make_visualize_widget.scene_index.value)
        elif make_visualize_widget.file_format.value == "viventis":
            dataset = mm.ViventisManager(path)
        elif make_visualize_widget.file_format.value == "opera":
            dataset = mm.OperaManager(path)
        elif make_visualize_widget.file_format.value == "luxendo":
            dataset = mm.LuxendoManager(path)
        elif make_visualize_widget.file_format.value == "zarrV05":
            dataset = mm.ZarrManager(path)
        elif make_visualize_widget.file_format.value == "zarrV04":
            dataset = mm.ZarrV04Manager(path)
        else:
            return

        _state["dataset"] = dataset

        suffix = ""
        if path.suffix == ".zarr":
            suffix = "_1"

        if path.is_file():
            name = path.stem          # removes extension
        else:
            name = path.name          # directory name as-is

        default_output = path.parent / f"{name}{suffix}.zarr"
        make_convert_widget.output_path.value = default_output

    make_visualize_widget.scene_index.enabled = False

    # -------------------------
    # Compose single dock widget
    # -------------------------

    container = QWidget()
    layout = QVBoxLayout(container)
    layout.addWidget(make_visualize_widget.native)
    layout.addWidget(make_convert_widget.native)
    # layout.addWidget(reset_roi_widget.native)

    # make_visualize_widget.input_path.tooltip = (
    #     "Select a folder containing OME-Zarr data (.zarr)"
    # )

    return container
