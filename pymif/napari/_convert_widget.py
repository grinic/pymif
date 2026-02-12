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

def make_convert_widget():
    viewer = current_viewer()
    _state = {"dataset": None}

    @magicgui(
        call_button="Convert to zarr",
        file_format={"choices": ["viventis", "opera", "luxendo", "zarrV05", "zarrV04", "zeiss"]},
        input_path={"widget_type": "FileEdit", "mode": "d"},
        scene_index={"widget_type": "SpinBox", "min": 0, "max": 100, "step": 1},
        output_path={"widget_type": "FileEdit", "mode": "d"},
    )
    def make_convert_widget_inner(
        input_path: FileEdit = None,
        file_format="zarrV05",
        scene_index=0,
        output_path: FileEdit = None,
    ):
        dataset = _state["dataset"]
        if dataset is None:
            return

        dataset.to_zarr(output_path)

    # -------------------------
    # Input path callback
    # -------------------------
    @make_convert_widget_inner.file_format.changed.connect
    def _update_mode(file_format):
        if file_format == "zeiss": 
            make_convert_widget_inner.scene_index.enabled = True 
        else: 
            make_convert_widget_inner.scene_index.enabled = False
        
        if file_format in ["zarrV05", "zarrV04", "viventis", "luxendo"]:
            make_convert_widget_inner.input_path.mode = "d" 
        else: 
            make_convert_widget_inner.input_path.mode = "rm"


    @make_convert_widget_inner.input_path.changed.connect
    def _on_input_path_change(path):
        if not path:
            return

        if make_convert_widget_inner.file_format.value == "zeiss":
            dataset = mm.ZeissManager(path, scene_index=make_convert_widget_inner.scene_index.value)
        elif make_convert_widget_inner.file_format.value == "viventis":
            dataset = mm.ViventisManager(path)
        elif make_convert_widget_inner.file_format.value == "opera":
            dataset = mm.OperaManager(path)
        elif make_convert_widget_inner.file_format.value == "luxendo":
            dataset = mm.LuxendoManager(path)
        elif make_convert_widget_inner.file_format.value == "zarrV05":
            dataset = mm.ZarrManager(path)
        elif make_convert_widget_inner.file_format.value == "zarrV04":
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
        make_convert_widget_inner.output_path.value = default_output

        # viewer.layers.clear()
        # viewer.open(path, plugin="napari-ome-zarr")

        # if "ROI" not in viewer.layers:
        #     add_roi_layer(viewer, dataset)

        # if "Zrange" not in viewer.layers:
        #     add_zrange_layer(viewer, dataset)

        # if "box" not in viewer.layers:
        #     viewer.add_shapes(
        #         data=[],
        #         shape_type="polygon",
        #         name="box",
        #         edge_color="coral",
        #         face_color="cyan",
        #         ndim=3,
        #         scale=dataset.metadata["scales"][0],
        #     )

    # @make_overview_widget_inner.input_path.changed.connect
    # def _enable_buttons(path):
    #     enabled = bool(path)
    #     make_overview_widget_inner.call_button.enabled = enabled
    #     reset_roi_widget.call_button.enabled = enabled

    # make_overview_widget_inner.call_button.enabled = False
    # reset_roi_widget.call_button.enabled = False

    make_convert_widget_inner.scene_index.enabled = False

    # -------------------------
    # Compose single dock widget
    # -------------------------

    container = QWidget()
    layout = QVBoxLayout(container)
    layout.addWidget(make_convert_widget_inner.native)
    # layout.addWidget(reset_roi_widget.native)

    # make_convert_widget_inner.input_path.tooltip = (
    #     "Select a folder containing OME-Zarr data (.zarr)"
    # )

    return container
