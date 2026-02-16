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

    # ---

    @magicgui(
        call_button="Convert to zarr",
        
        chunk_xy={"label": "Chunk XY (px)", "min": 64, "max": 2048, "step": 1, "value": 512},
        chunk_z={"label": "Chunk Z (px)", "min": 1, "max": 256, "step": 1, "value": 16},
        
        chunk_mb={"label": "Chunk size (MB)", "min": 1, "max": 512, "step": 1, "value": 64},

        n_levels={"label": "Resolution levels", "min": 1, "max": 10, "value": 5},
        
        t_start={"label": "T start", "min": 0, "step": 1, "value": 0},
        t_end={"label": "T end", "min": 0, "step": 1, "value": 0},

        channels={
            "label": "Channels",
            "choices": ["ch0", "ch1", "ch2"],
            "widget_type": "Select",
            "allow_multiple": True,
        },

        z_start={"label": "Z start", "min": 0, "step": 1, "value": 0},
        z_end={"label": "Z end", "min": 0, "step": 1, "value": 0},

        y_start={"label": "Y start", "min": 0, "step": 1, "value": 0},
        y_end={"label": "Y end", "min": 0, "step": 1, "value": 0},

        x_start={"label": "X start", "min": 0, "step": 1, "value": 0},
        x_end={"label": "X end", "min": 0, "step": 1, "value": 0},

        output_path={"widget_type": "FileEdit", "mode": "d"},
    )
    def make_convert_widget(
        chunk_xy=512,
        chunk_z=16,
        chunk_mb=64,
        n_levels=5,
        t_start=0, t_end=0,
        z_start=0, z_end=0,
        y_start=0, y_end=0,
        x_start=0, x_end=0,
        channels=(),
        output_path: FileEdit = None,
    ):
        dataset = _state["dataset"]
        if dataset is None:
            return

        dataset.to_zarr(output_path)

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

        if path.is_file() or path.suffix==".zarr":
            name = path.stem          # removes extension
        else:
            name = path.name          # directory name as-is

        default_output = path.parent / f"{name}{suffix}.zarr"
        make_convert_widget.output_path.value = default_output

        make_convert_widget.channels.choices = dataset.metadata["channel_names"]
        make_convert_widget.channels.value = tuple(dataset.metadata["channel_names"])


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
