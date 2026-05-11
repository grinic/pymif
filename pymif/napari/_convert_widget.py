import os
import napari
from napari import current_viewer
from datetime import datetime
from napari.qt.threading import thread_worker
import numpy as np
from magicgui import magicgui
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import pymif.microscope_manager as mm
from magicgui.widgets import CheckBox, FileEdit, CheckBox
import sys
from qtpy.QtWidgets import QTextEdit
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtGui import QTextCursor
from qtpy.QtWidgets import QFileDialog, QWidget, QVBoxLayout, QPushButton, QLabel, QToolButton
from matplotlib import rc
rc('font', size=12)
rc('font', family='Arial')
# rc('font', serif='Times')
rc('pdf', fonttype=42)
# rc('text', usetex=True)

# -------------------------
# MagicGUI controls
# -------------------------

class EmittingStream(QObject):
    """Qt-compatible text stream used to forward console output to the widget."""

    text_written = Signal(str)

    def write(self, text):
        """Emit written text to the Qt signal."""
        self.text_written.emit(str(text))

    def flush(self):
        """Provide a file-like ``flush`` method for compatibility."""
        pass

    def isatty(self):
        """Report that this stream is not an interactive terminal."""
        return False
    
def dataset_reader(microscope):
    """Return the PyMIF manager class matching the selected microscope key."""
    if microscope == "zeiss":
        reader = mm.ZeissManager
    elif microscope == "viventis":
        reader = mm.ViventisManager
    elif microscope == "opera":
        reader = mm.OperaManager
    elif microscope == "luxendo":
        reader = mm.LuxendoManager
    elif microscope == "zarr":
        reader = mm.ZarrManager
    elif microscope == "scape":
        reader = mm.ScapeManager
    else:
        raise ValueError(f"Unsupported microscope: {microscope}")
    
    return reader


def _dataset_axes(dataset):
    """Return normalized dataset axes, defaulting to legacy TCZYX metadata."""
    return str(dataset.metadata.get("axes", "tczyx")).lower()


def _axis_index(dataset, axis):
    axes = _dataset_axes(dataset)
    return axes.index(axis) if axis in axes else None


def _axis_size(dataset, axis, default=1):
    idx = _axis_index(dataset, axis)
    if idx is None:
        return default
    return int(dataset.metadata["size"][0][idx])


def _scale_for_axes(dataset, requested_axes):
    """Return a napari scale tuple for the requested spatial axis labels."""
    axes = _dataset_axes(dataset)
    spatial_axes = [ax for ax in axes if ax in "zyx"]
    scale_map = dict(zip(spatial_axes, dataset.metadata.get("scales", [(1,) * len(spatial_axes)])[0]))
    return tuple(scale_map.get(ax, 1) for ax in requested_axes if ax in axes)


def _chunk_shape_for_dataset(dataset, chunk_z=16, chunk_y=512, chunk_x=512):
    """Map widget Z/Y/X chunk values onto whatever axes the dataset has."""
    chunk_map = {"t": 1, "c": 1, "z": chunk_z, "y": chunk_y, "x": chunk_x}
    axes = _dataset_axes(dataset)
    return tuple(int(max(1, min(_axis_size(dataset, ax), chunk_map[ax]))) for ax in axes)


def _present_range(dataset, axis, requested_range):
    """Return a clipped inclusive range tuple for a present axis, else None."""
    if _axis_index(dataset, axis) is None:
        return None
    max_idx = _axis_size(dataset, axis) - 1
    start, end = requested_range
    return int(max(0, min(start, max_idx))), int(max(0, min(end, max_idx)))


def _set_range_widget(widget, size, enabled=True):
    max_idx = max(0, int(size) - 1)
    widget.max = max_idx
    widget.value = (0, max_idx)
    widget.enabled = bool(enabled and size > 1)

def get_chunk_size(dataset_size, max_size_mb=100, axes="tczyx"):
    """Estimate a chunk shape for any dataset axis subset.

    The returned tuple follows ``axes``.  Legacy callers that pass a five-value
    TCZYX size still receive a five-value chunk tuple.
    """
    axes = str(axes).lower()
    size_map = dict(zip(axes, dataset_size))
    z = max(1, int(size_map.get("z", 1)))
    y = max(1, int(size_map.get("y", 1)))
    x = max(1, int(size_map.get("x", 1)))

    n_chunks = {"z": 4, "y": 1, "x": 1}
    chunk_map = {"t": 1, "c": 1}

    while True:
        chunk_map.update(
            {
                "z": int(z / n_chunks["z"] + 1),
                "y": int(y / n_chunks["y"] + 1),
                "x": int(x / n_chunks["x"] + 1),
            }
        )
        size_mb = 2 * chunk_map["z"] * chunk_map["y"] * chunk_map["x"] / 1024 / 1024
        if size_mb <= max_size_mb:
            break
        n_chunks = {k: v * 2 for k, v in n_chunks.items()}

    return tuple(int(max(1, min(size_map.get(ax, 1), chunk_map[ax]))) for ax in axes)


def get_n_levels(dataset_size, axes="tczyx"):
    """Estimate a default number of pyramid levels from present spatial axes."""
    axes = str(axes).lower()
    size_map = dict(zip(axes, dataset_size))
    shape = [max(1, int(size_map[ax])) for ax in axes if ax in "zyx"]
    if not shape:
        return 1

    n = 1
    while any(s > 2048 for s in shape):
        n += 1
        shape = [max(1, s // 2) for s in shape]

    return max(3, n)


def _run_conversion(
    reader,
    path,
    scene_index,
    chunks,
    t_range,
    single_t,
    z_range,
    single_z,
    y_range,
    x_range,
    channels,
    n_levels,
    downscale_factor,
    output_path,
    file_format,
    zarr_format,
    ):
    """Run the conversion pipeline used by the napari worker thread."""
    print("Starting conversion in background thread...")

    if file_format == "zeiss":
        dataset = reader(path, scene_index=scene_index, chunks=chunks)
    else:
        dataset = reader(path, chunks=chunks)

    axes = _dataset_axes(dataset)
    print("Dataset axes:", axes)

    if len(channels) == 0 or "c" not in axes:
        channels_index = None
    else:
        channels_index = [
            dataset.metadata["channel_names"].index(ch)
            for ch in channels
            if ch in dataset.metadata.get("channel_names", [])
        ]

    subset_kwargs = {}
    tr = _present_range(dataset, "t", t_range)
    zr = _present_range(dataset, "z", z_range)
    yr = _present_range(dataset, "y", y_range)
    xr = _present_range(dataset, "x", x_range)

    if tr is not None and _axis_size(dataset, "t") > 1:
        subset_kwargs["T"] = list(range(tr[0], tr[1] + 1))
    if channels_index is not None and _axis_size(dataset, "c") > 1:
        subset_kwargs["C"] = channels_index
    if zr is not None and _axis_size(dataset, "z") > 1:
        subset_kwargs["Z"] = list(range(zr[0], zr[1] + 1))
    if yr is not None:
        subset_kwargs["Y"] = list(range(yr[0], yr[1] + 1))
    if xr is not None:
        subset_kwargs["X"] = list(range(xr[0], xr[1] + 1))

    if subset_kwargs:
        dataset.subset_dataset(**subset_kwargs)

    chunks = _chunk_shape_for_dataset(
        dataset,
        chunk_z=chunks[2] if len(chunks) > 2 else 1,
        chunk_y=chunks[3] if len(chunks) > 3 else 512,
        chunk_x=chunks[4] if len(chunks) > 4 else 512,
    )

    print("Requested input chunks:", chunks)
    print("Chunks before pyramid:", [arr.chunksize for arr in dataset.data])

    dataset.build_pyramid(num_levels=n_levels, downscale_factor=downscale_factor)

    dataset.data = [
        arr.rechunk(chunks) if arr.ndim == len(chunks) else arr
        for arr in dataset.data
    ]

    print("Chunks after pyramid:", [arr.chunksize for arr in dataset.data])

    ngff_version = "0.4" if zarr_format == 2 else "0.5"
    dataset.to_zarr(output_path, zarr_format=zarr_format, ngff_version=ngff_version)    

    return output_path

@thread_worker
def convert_worker(**kwargs):
    """Background worker wrapper used by the napari conversion widget."""
    return _run_conversion(**kwargs)


def convert_widget():
    """Create the main PyMIF napari conversion widget."""
    viewer = current_viewer()
    
    def lock_roi_in_3d(event=None):
        if "ROI" not in viewer.layers:
            return
        if viewer.dims.ndisplay == 3:
            viewer.layers["ROI"].editable = False
            viewer.layers["ROI"].selectable = False
            make_convert_widget.y_range.enabled = False
            make_convert_widget.x_range.enabled = False
        else:
            viewer.layers["ROI"].editable = True
            viewer.layers["ROI"].selectable = True
            dataset = _state.get("dataset")
            make_convert_widget.y_range.enabled = bool(dataset and _axis_index(dataset, "y") is not None)
            make_convert_widget.x_range.enabled = bool(dataset and _axis_index(dataset, "x") is not None)

    def ensure_crop_layers(dataset):
        axes = _dataset_axes(dataset)
        if "ROI" not in viewer.layers and {"y", "x"}.issubset(axes):
            ymax = _axis_size(dataset, "y") - 1
            xmax = _axis_size(dataset, "x") - 1
            rect = np.array([
                [0, 0],
                [0, xmax],
                [ymax, xmax],
                [ymax, 0],
            ])
            layer = viewer.add_shapes(
                data=rect,
                shape_type="rectangle",
                name="ROI",
                edge_color="orange",
                face_color="transparent",
                edge_width=10,
                ndim=2,
                scale=_scale_for_axes(dataset, "yx"),
            )
            layer.mode = "add_rectangle"

        if "Zrange" not in viewer.layers and {"z", "y", "x"}.issubset(axes):
            points = np.array([
                [0, _axis_size(dataset, "y") // 2, _axis_size(dataset, "x") // 2],
                [_axis_size(dataset, "z") - 1, _axis_size(dataset, "y") // 2, _axis_size(dataset, "x") // 2],
            ])
            layer = viewer.add_points(
                data=points,
                name="Zrange",
                face_color="lime",
                size=20,
                ndim=3,
                out_of_slice_display=True,
                scale=_scale_for_axes(dataset, "zyx"),
            )
            layer.mode = "add"

        if "CropBox" not in viewer.layers and {"z", "y", "x"}.issubset(axes):
            layer = viewer.add_shapes(
                data=np.empty((0, 2, 3)),
                shape_type="polygon",
                name="CropBox",
                edge_color="cyan",
                face_color="transparent",
                ndim=3,
                scale=_scale_for_axes(dataset, "zyx"),
            )
            layer.editable = False          # prevents adding/moving/editing shapes
            layer.selectable = False        # prevents selecting the shape
            layer.mode = "pan_zoom"         # ensures no shape tool is active
            layer.opacity = 0.7
            layer.edge_width = 2
            layer.blending = "additive"

    def update_3d_box():
        if any(name not in viewer.layers for name in ("ROI", "Zrange", "CropBox")):
            return
        roi_layer = viewer.layers["ROI"]
        zrange_layer = viewer.layers["Zrange"]
        if len(roi_layer.data) == 0 or len(zrange_layer.data) < 2:
            return
        zvals = [d[0] for d in zrange_layer.data]

        zrange = list(zvals[:2])

        roi = [list(r) for r in roi_layer.data[0]]

        polygon = [
            [[zrange[0]] + roi[0], [zrange[0]] + roi[1], [zrange[0]] + roi[2], [zrange[0]] + roi[3]],
            [[zrange[1]] + roi[0], [zrange[1]] + roi[1], [zrange[1]] + roi[2], [zrange[1]] + roi[3]],
            [[zrange[0]] + roi[0], [zrange[1]] + roi[0], [zrange[1]] + roi[1], [zrange[0]] + roi[1]],
            [[zrange[0]] + roi[0], [zrange[1]] + roi[0], [zrange[1]] + roi[3], [zrange[0]] + roi[3]],
            [[zrange[0]] + roi[1], [zrange[1]] + roi[1], [zrange[1]] + roi[2], [zrange[0]] + roi[2]],
            [[zrange[0]] + roi[3], [zrange[1]] + roi[3], [zrange[1]] + roi[2], [zrange[0]] + roi[2]],
        ]

        viewer.layers["CropBox"].data = polygon
        viewer.layers["CropBox"].refresh()

    def clamp(v, vmin, vmax):
        return int(max(vmin, min(v, vmax)))


    def roi_to_widget(event=None):
        if _syncing["roi"]:
            return
        _syncing["roi"] = True
    
        try:
            if "ROI" not in viewer.layers:
                return
            layer = viewer.layers["ROI"]
            if layer is None or len(layer.data) == 0:
                return

            (y0, x0), (_, _), (y1, x1), (_, _) = layer.data[0]

            y0, y1 = sorted([int(y0), int(y1)])
            x0, x1 = sorted([int(x0), int(x1)])

            dataset = _state["dataset"]
            b = {
                "y_max": _axis_size(dataset, "y") - 1,
                "x_max": _axis_size(dataset, "x") - 1,
            }

            y0 = clamp(y0, 0, b["y_max"])
            y1 = clamp(y1, 0, b["y_max"])
            x0 = clamp(x0, 0, b["x_max"])
            x1 = clamp(x1, 0, b["x_max"])

            layer.data = [np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]])]

            make_convert_widget.y_range.value = (y0, y1)
            make_convert_widget.x_range.value = (x0, x1)

            update_3d_box()
        finally:
            _syncing["roi"] = False

    def zpoints_to_widget(event=None):
        if _syncing["z"]:
            return
        _syncing["z"] = True
        
        try:
            if "Zrange" not in viewer.layers:
                return
            layer = viewer.layers["Zrange"]
            if layer is None or len(layer.data) < 2:
                return

            zs = sorted(int(p[0]) for p in layer.data[:2])

            make_convert_widget.z_range.value = (zs[0], zs[1])

            update_3d_box()
        finally:
            _syncing["z"] = False

    def widget_to_roi(event=None):
        if _syncing["roi"]:
            return
        _syncing["roi"] = True

        try:    
            if "ROI" not in viewer.layers:
                return
            layer = viewer.layers["ROI"]

            dataset = _state["dataset"]
            b = {
                "y_max": _axis_size(dataset, "y") - 1,
                "x_max": _axis_size(dataset, "x") - 1,
            }

            y0 = clamp(make_convert_widget.y_range.value[0], 0, b["y_max"])
            y1 = clamp(make_convert_widget.y_range.value[1], 0, b["y_max"])
            x0 = clamp(make_convert_widget.x_range.value[0], 0, b["x_max"])
            x1 = clamp(make_convert_widget.x_range.value[1], 0, b["x_max"])

            y0, y1 = sorted([y0, y1])
            x0, x1 = sorted([x0, x1])

            rect = np.array([
                [y0, x0],
                [y0, x1],
                [y1, x1],
                [y1, x0],
            ])

            layer.data = [rect]

            update_3d_box()
        finally:
            _syncing["roi"] = False
    
    def widget_to_zpoints(event=None):
        if _syncing["z"]:
            return
        _syncing["z"] = True
        
        try:
            if "Zrange" not in viewer.layers:
                return
            layer = viewer.layers["Zrange"]

            z0 = make_convert_widget.z_range.value[0]
            z1 = make_convert_widget.z_range.value[1]

            prev_points = layer.data

            points = np.array([
                [z0, prev_points[0,1], prev_points[0,2]],
                [z1, prev_points[1,1], prev_points[1,2]],
            ])

            layer.data = points

            update_3d_box()
        finally:
            _syncing["z"] = False
        
    def keep_single_roi(event=None):
        if _syncing["roi"]:
            return
        _syncing["roi"] = True

        try:    
            if "ROI" not in viewer.layers:
                return
            layer = viewer.layers["ROI"]
            if len(layer.data) > 1:
                layer.data = [layer.data[-1]]

        finally:
            _syncing["roi"] = False

    def keep_two_zpoints(event=None):
        if _syncing["z"]:
            return
        _syncing["z"] = True
        
        try:
            if "Zrange" not in viewer.layers:
                return
            layer = viewer.layers["Zrange"]
            if len(layer.data) > 2:
                layer.data = layer.data[-2:]
        finally:
            _syncing["z"] = False

    _state = {"dataset": None}
    _syncing = {"roi": False, "z": False}

    @magicgui(
        call_button="Visualize in napari",
        input_path={"widget_type": "FileEdit", "mode": "d"},
        scene_index={"widget_type": "SpinBox", "min": 0, "max": 100, "step": 1},
        file_format={"choices": ["viventis", "opera", "luxendo", "zarr", "zeiss","scape"]},
    )
    def make_visualize_widget(
        file_format="zarr",
        scene_index=0,
        input_path: FileEdit = None,
    ):
        dataset = _state["dataset"]
        if dataset is None:
            return

        viewer.layers.clear()

        if make_visualize_widget.file_format.value in ["zarr"]:
            viewer.open(make_visualize_widget.input_path.value, plugin='napari-ome-zarr')
        else:
            dataset = _state["dataset"]
            dataset.visualize(
                viewer=viewer
            )

        # initialize ROI to full image when spatial axes are available
        ensure_crop_layers(dataset)

        widget_to_roi()
        widget_to_zpoints()
        update_3d_box()

        if "ROI" in viewer.layers:
            viewer.layers["ROI"].mode = "add_rectangle"
            viewer.layers["ROI"].events.data.connect(keep_single_roi)
            viewer.layers["ROI"].events.data.connect(roi_to_widget)
            make_convert_widget.y_range.changed.connect(widget_to_roi)
            make_convert_widget.x_range.changed.connect(widget_to_roi)

        if "Zrange" in viewer.layers:
            viewer.layers["Zrange"].events.data.connect(keep_two_zpoints)
            viewer.layers["Zrange"].events.data.connect(zpoints_to_widget)
            make_convert_widget.z_range.changed.connect(widget_to_zpoints)

    # ---

    @magicgui(
        call_button="Convert to zarr",

        chunk_x={"label": "Chunk X", "min": 8, "max": 2**16, "step": 1, "value": 512},
        chunk_y={"label": "Chunk Y", "min": 8, "max": 2**16, "step": 1, "value": 512},
        chunk_z={"label": "Chunk Z", "min": 1, "max": 2**16, "step": 1, "value": 16},
        
        n_levels={"label": "Resolution levels", "min": 1, "max": 10, "value": 5},
        downscale_z={"label": "Downscale Z", "min": 1, "max": 64, "step": 1, "value": 2},
        downscale_y={"label": "Downscale Y", "min": 1, "max": 64, "step": 1, "value": 2},
        downscale_x={"label": "Downscale X", "min": 1, "max": 64, "step": 1, "value": 2},
        zarr_format={"label": "Zarr format", "choices": [2, 3], "value": 3},
        
        t_range={"label": "T range", "widget_type": "RangeSlider", "min": 0, "max": 2**16, "step": 1, "value": (0, 2**16)},
        single_t = {"label": "Single T frame", "widget_type": "CheckBox", "value": False},
        z_range={"label": "Z range", "widget_type": "RangeSlider", "min": 0, "max": 2**16, "step": 1, "value": (0, 2**16)},
        single_z = {"label": "Single Z plane", "widget_type": "CheckBox", "value": False},
        y_range={"label": "Y range", "widget_type": "RangeSlider", "min": 0, "max": 2**16, "step": 1, "value": (0, 2**16)},
        x_range={"label": "X range", "widget_type": "RangeSlider", "min": 0, "max": 2**16, "step": 1, "value": (0, 2**16)},

        channels={
            "label": "Channels",
            "choices": ["ch0", "ch1", "ch2"],
            "widget_type": "Select",
            "allow_multiple": True,
        },

        output_path={"widget_type": "FileEdit", "mode": "d"},
    )
    def make_convert_widget(
        t_range=(0,1000),
        single_t=False,
        z_range=(0,1000),
        single_z=False,
        y_range=(0,1000),
        x_range=(0,1000),
        channels=(),
        chunk_x=512,
        chunk_y=512,
        chunk_z=16,
        n_levels=5,
        downscale_z=2,
        downscale_y=2,
        downscale_x=2,
        zarr_format=3,
        output_path: FileEdit = None,
    ):

        reader = dataset_reader(make_visualize_widget.file_format.value)
        path = make_visualize_widget.input_path.value
        scene_index = make_visualize_widget.scene_index.value
        file_format = make_visualize_widget.file_format.value

        chunks = (1, 1, chunk_z, chunk_y, chunk_x)
        downscale_factor = (downscale_z, downscale_y, downscale_x)
        t_start, t_end = t_range
        z_start, z_end = z_range
        y_start, y_end = y_range
        x_start, x_end = x_range

        if make_visualize_widget.file_format.value == "zeiss":
            dataset = reader(path, scene_index = scene_index, chunks = chunks)
        else:
            dataset = reader(path, chunks = chunks)  

        worker = convert_worker(
                reader=reader,
                path=path,
                scene_index=scene_index,
                chunks=chunks,
                t_range=t_range,
                single_t=single_t,
                z_range=z_range,
                single_z=single_z,
                y_range=y_range,
                x_range=x_range,
                channels=channels,
                n_levels=n_levels,
                downscale_factor=downscale_factor,
                output_path=output_path,
                file_format=file_format,
                zarr_format=zarr_format,
            )
        
        make_convert_widget.enabled = False
        make_visualize_widget.enabled = False

        @worker.returned.connect
        def _on_done(result_path):
            print(f"Conversion completed: {result_path}")
            make_convert_widget.enabled = True
            make_visualize_widget.enabled = True

        @worker.errored.connect
        def _on_error(err):
            print("Conversion failed:", err)
            make_convert_widget.enabled = True
            make_visualize_widget.enabled = True

        worker.start()

    # -------------------------
    # Input path callback
    # -------------------------

    @make_visualize_widget.file_format.changed.connect
    def _update_mode(file_format):
        if file_format == "zeiss": 
            make_visualize_widget.scene_index.enabled = True 
        else: 
            make_visualize_widget.scene_index.enabled = False
        
        if file_format in ["zarr", "viventis", "luxendo"]:
            make_visualize_widget.input_path.mode = "d" 
        else: 
            make_visualize_widget.input_path.mode = "r"

    @make_visualize_widget.input_path.changed.connect
    def _on_input_path_change(path):
        if not path:
            return
        
        reader = dataset_reader(make_visualize_widget.file_format.value)

        if make_visualize_widget.file_format.value == "zeiss":
            dataset = reader(path, scene_index=make_visualize_widget.scene_index.value)
        else:
            dataset = reader(path)

        for i in dataset.metadata:
            print(f"{i.upper()}: {dataset.metadata[i]}")

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

        make_convert_widget.enabled = True

        axes = _dataset_axes(dataset)
        chunk_size = get_chunk_size(dataset.metadata["size"][0], max_size_mb=100, axes=axes)
        chunk_map = dict(zip(axes, chunk_size))
        num_levels = get_n_levels(dataset.metadata["size"][0], axes=axes)

        make_convert_widget.chunk_x.max = _axis_size(dataset, "x") + 1
        make_convert_widget.chunk_x.value = chunk_map.get("x", 1)
        make_convert_widget.chunk_x.enabled = "x" in axes

        make_convert_widget.chunk_y.max = _axis_size(dataset, "y") + 1
        make_convert_widget.chunk_y.value = chunk_map.get("y", 1)
        make_convert_widget.chunk_y.enabled = "y" in axes

        make_convert_widget.chunk_z.max = _axis_size(dataset, "z") + 1
        make_convert_widget.chunk_z.value = chunk_map.get("z", 1)
        make_convert_widget.chunk_z.enabled = "z" in axes

        make_convert_widget.n_levels.value = num_levels
        make_convert_widget.downscale_z.enabled = "z" in axes
        make_convert_widget.downscale_y.enabled = "y" in axes
        make_convert_widget.downscale_x.enabled = "x" in axes

        _set_range_widget(make_convert_widget.t_range, _axis_size(dataset, "t"), enabled="t" in axes)
        _set_range_widget(make_convert_widget.z_range, _axis_size(dataset, "z"), enabled="z" in axes)
        _set_range_widget(make_convert_widget.y_range, _axis_size(dataset, "y"), enabled="y" in axes)
        _set_range_widget(make_convert_widget.x_range, _axis_size(dataset, "x"), enabled="x" in axes)
        make_convert_widget.single_t.enabled = "t" in axes
        make_convert_widget.single_z.enabled = "z" in axes

        channel_names = dataset.metadata.get("channel_names", []) if "c" in axes else []
        n_channels = len(channel_names)
        row_height = 20  # approx height per item in pixels
        channels_widget = make_convert_widget.channels.native
        channels_widget.setMaximumHeight(row_height * max(1, n_channels))
        make_convert_widget.channels.choices = channel_names
        make_convert_widget.channels.value = tuple(channel_names)
        make_convert_widget.channels.enabled = n_channels > 0

        make_convert_widget.output_path.value = default_output

        viewer.layers.clear()

    make_visualize_widget.scene_index.enabled = False
    make_convert_widget.enabled = False
    viewer.dims.events.ndisplay.connect(lock_roi_in_3d)
    # lock_roi_in_3d()

    def sync_single_z(checked):
        dataset = _state.get("dataset")
        has_z = bool(dataset and _axis_index(dataset, "z") is not None and _axis_size(dataset, "z") > 1)
        if checked and has_z:
            z_val = make_convert_widget.z_range.value
            # clamp to single plane
            make_convert_widget.z_range.value = (z_val[0], z_val[0])
            make_convert_widget.z_range.enabled = False
        else:
            make_convert_widget.z_range.enabled = has_z

    make_convert_widget.single_z.changed.connect(sync_single_z)

    def sync_single_t(checked):
        dataset = _state.get("dataset")
        has_t = bool(dataset and _axis_index(dataset, "t") is not None and _axis_size(dataset, "t") > 1)
        if checked and has_t:
            t_val = make_convert_widget.t_range.value
            # clamp to single frame
            make_convert_widget.t_range.value = (t_val[0], t_val[0])
            make_convert_widget.t_range.enabled = False
        else:
            make_convert_widget.t_range.enabled = has_t

    make_convert_widget.single_t.changed.connect(sync_single_t)
    # -------------------------
    # Compose single dock widget
    # -------------------------

    container = QWidget()
    layout = QVBoxLayout(container)
    
    title1_label = QLabel("Dataset loading:")
    title1_label.setStyleSheet("""
        QLabel {
            font-weight: bold;
            font-size: 15px;
            padding: 2px 0px;
        }
    """)

    title2_label = QLabel("Dataset conversion:")
    title2_label.setStyleSheet("""
        QLabel {
            font-weight: bold;
            font-size: 15px;
            padding: 2px 0px;
        }
    """)
    layout.addWidget(title1_label)
    layout.addWidget(make_visualize_widget.native)
    layout.addWidget(title2_label)

    advanced_btn = QToolButton()
    advanced_btn.setText("Additional parameters ▸")
    advanced_btn.setStyleSheet("""
        QToolButton {
            font-size: 13px;
            font-weight: 600;
            color: #cccccc;
            padding: 4px 0px;
        }
        QToolButton:hover {
            color: white;
        }
        """)
    advanced_btn.setCheckable(True)
    advanced_btn.setChecked(False)
    advanced_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

    advanced_widgets = [
        make_convert_widget.chunk_x.native.parent(),
        make_convert_widget.chunk_y.native.parent(),
        make_convert_widget.chunk_z.native.parent(),
        make_convert_widget.n_levels.native.parent(),
        make_convert_widget.downscale_z.native.parent(),
        make_convert_widget.downscale_y.native.parent(),
        make_convert_widget.downscale_x.native.parent(),
        make_convert_widget.zarr_format.native.parent(),
    ]

    for w in advanced_widgets:
        w.setVisible(False)

    def _toggle_advanced(checked):
        for w in advanced_widgets:
            w.setVisible(checked)
        advanced_btn.setText("Additional parameters ▾" if checked else "Additional parameters ▸")
        advanced_btn.setStyleSheet("""
            QToolButton {
                font-size: 13px;
                font-weight: 600;
                color: %s;
            }
        """ % ("#ffffff" if checked else "#aaaaaa"))
        
    advanced_btn.toggled.connect(_toggle_advanced)

    make_convert_widget.native.layout().insertWidget(7, advanced_btn)

    layout.addWidget(make_convert_widget.native)
    # layout.addWidget(reset_roi_widget.native)

    # make_visualize_widget.input_path.tooltip = (
    #     "Select a folder containing OME-Zarr data (.zarr)"
    # )

    log_widget = QTextEdit()
    log_widget.setReadOnly(True)
    log_widget.setMinimumHeight(150)
    log_widget.setStyleSheet("""
        QTextEdit {
            background-color: #111;
            color: #ddd;
            font-family: Consolas, Menlo, monospace;
            font-size: 12px;
        }
    """)

    stdout_stream = EmittingStream()
    stderr_stream = EmittingStream()

    sys.stdout = stdout_stream
    sys.stderr = stderr_stream

    def append_with_time(text):
        if not text.strip():
            return

        ts = datetime.now().strftime("[%H:%M:%S] ")
        html = f"<span style='color:#ffffff;'>{ts}{text.rstrip()}</span>"

        # insertHtml does NOT add an automatic newline like append()
        log_widget.insertHtml(html + "<br>")  # we add exactly one <br> per line

        # auto-scroll
        cursor = log_widget.textCursor()
        cursor.movePosition(QTextCursor.End)
        log_widget.setTextCursor(cursor)
        log_widget.ensureCursorVisible()

    stdout_stream.text_written.connect(append_with_time)
    stderr_stream.text_written.connect(append_with_time)

    title3_label = QLabel("PyMIF Log:")
    title3_label.setStyleSheet("""
        QLabel {
            font-weight: bold;
            font-size: 15px;
            padding: 2px 0px;
        }
    """)

    clear_btn = QPushButton("Clear log")
    clear_btn.clicked.connect(log_widget.clear)

    layout.addWidget(title3_label)
    layout.addWidget(log_widget)
    layout.addWidget(clear_btn)



    return container
