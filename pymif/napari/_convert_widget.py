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
from PyQt5.QtWidgets import QFileDialog, QWidget, QVBoxLayout, QPushButton, QLabel, QToolButton
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
    text_written = Signal(str)

    def write(self, text):
        self.text_written.emit(str(text))

    def flush(self):
        pass

def dataset_reader(microscope):
    if microscope == "zeiss":
        reader = mm.ZeissManager
    elif microscope == "viventis":
        reader = mm.ViventisManager
    elif microscope == "opera":
        reader = mm.OperaManager
    elif microscope == "luxendo":
        reader = mm.LuxendoManager
    elif microscope == "zarrV05":
        reader = mm.ZarrManager
    elif microscope == "zarrV04":
        reader = mm.ZarrV04Manager
    elif microscope == "scape":
        reader = mm.ScapeManager
    else:
        raise ValueError(f"Unsupported microscope: {microscope}")
    
    return reader

def get_chunk_size(dataset_size, max_size_mb=100):
    # --- Select chunk size ---
    n_chunks = [4,1,1]
    chunk_size = [
        1, 1, # T, C
        int((dataset_size[2]/n_chunks[0])+1),  # Z
        int((dataset_size[3]/n_chunks[1])+1),  # Y
        int((dataset_size[4]/n_chunks[2])+1)   # X
    ]
    size_mb = 2*chunk_size[2]*chunk_size[3]*chunk_size[4]/1024/1024
    while size_mb > max_size_mb:
        n_chunks = [n_chunks[0]*2,n_chunks[1]*2,n_chunks[2]*2]
        chunk_size = [
            1, 1, # T, C
            int((dataset_size[2]/n_chunks[0])+1),  # Z
            int((dataset_size[3]/n_chunks[1])+1),  # Y
            int((dataset_size[4]/n_chunks[2])+1)   # X
        ]
        size_mb = 2*chunk_size[2]*chunk_size[3]*chunk_size[4]/1024/1024

    return chunk_size

def get_n_levels(dataset_size):
    n = 1
    shape = [dataset_size[2], dataset_size[3], dataset_size[4]] # [Y, X]
    while (shape[0]>2048) or (shape[1]>2048) or (shape[2]>2048):
        n+=1
        shape = [shape[0]//2, shape[1]//2, shape[2]//2]

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
    output_path,
    file_format,
):
    print("Starting conversion in background thread...")

    if file_format == "zeiss":
        dataset = reader(path, scene_index=scene_index, chunks=chunks)
    else:
        dataset = reader(path, chunks=chunks)

    t_start, t_end = t_range
    z_start, z_end = z_range
    y_start, y_end = y_range
    x_start, x_end = x_range

    print(t_start,t_end)

    if len(channels) == 0:
        channels_index = None
    else:
        channels_index = [
            dataset.metadata["channel_names"].index(ch)
            for ch in channels
        ]

    if dataset.metadata["size"][0][0] > 1:
        dataset.subset_dataset(T=list(range(t_start, t_end + 1)))

    if dataset.metadata["size"][0][1] > 1:
        dataset.subset_dataset(C=channels_index)

    if dataset.metadata["size"][0][2] > 1:
        dataset.subset_dataset(Z=list(range(z_start, z_end + 1)))

    dataset.subset_dataset(
        Y=list(range(y_start, y_end + 1)),
        X=list(range(x_start, x_end + 1)),
    )

    dataset.build_pyramid(num_levels=n_levels)
    dataset.to_zarr(output_path)

    return output_path

@thread_worker
def convert_worker(**kwargs):
    return _run_conversion(**kwargs)

def convert_widget():
    viewer = current_viewer()
    
    def lock_roi_in_3d(event=None):
        if viewer.dims.ndisplay == 3:
            viewer.layers["ROI"].editable = False
            viewer.layers["ROI"].selectable = False
            make_convert_widget.y_range.enabled = False
            make_convert_widget.x_range.enabled = False
        else:
            viewer.layers["ROI"].editable = True
            viewer.layers["ROI"].selectable = True
            make_convert_widget.y_range.enabled = True
            make_convert_widget.x_range.enabled = True

    def ensure_crop_layers(dataset):
        if "ROI" not in viewer.layers:
            rect = np.array([
                [0, 0],
                [0, dataset.metadata["size"][0][4]-1],
                [dataset.metadata["size"][0][3]-1, dataset.metadata["size"][0][4]-1],
                [dataset.metadata["size"][0][3]-1, 0],
            ])
            layer = viewer.add_shapes(
                data=rect,
                shape_type="rectangle",
                name="ROI",
                edge_color="orange",
                face_color="transparent",
                edge_width=10,
                ndim=2,
                scale=dataset.metadata["scales"][0][1:],  # YX
            )
            layer.mode = "add_rectangle"

        if "Zrange" not in viewer.layers:
            points = np.array([
                [0, dataset.metadata["size"][0][3]//2, dataset.metadata["size"][0][4]//2],
                [dataset.metadata["size"][0][2]-1, dataset.metadata["size"][0][3]//2, dataset.metadata["size"][0][4]//2],
            ])
            layer = viewer.add_points(
                data=points,
                name="Zrange",
                face_color="lime",
                size=20,
                ndim=3,  # (z, y, x) but we'll only use z
                out_of_slice_display=True,
                scale=dataset.metadata["scales"][0],
            )
            layer.mode = "add"

        if "CropBox" not in viewer.layers:
            layer = viewer.add_shapes(
                data=np.empty((0, 2, 3)),
                shape_type="polygon",
                name="CropBox",
                edge_color="cyan",
                face_color="transparent",
                ndim=3,
                scale=dataset.metadata["scales"][0],
            )
            layer.editable = False          # prevents adding/moving/editing shapes
            layer.selectable = False        # prevents selecting the shape
            layer.mode = "pan_zoom"         # ensures no shape tool is active
            layer.opacity = 0.7
            layer.edge_width = 2
            layer.blending = "additive"

    def update_3d_box():
        roi_layer = viewer.layers["ROI"]
        zrange_layer = viewer.layers["Zrange"]
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
                "y_max": dataset.metadata["size"][0][3] - 1,
                "x_max": dataset.metadata["size"][0][4] - 1,
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
                "y_max": dataset.metadata["size"][0][3] - 1,
                "x_max": dataset.metadata["size"][0][4] - 1,
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
        file_format={"choices": ["viventis", "opera", "luxendo", "zarrV05", "zarrV04", "zeiss","scape"]},
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

        # initialize ROI to full image
        ensure_crop_layers(dataset)

        widget_to_roi()
        widget_to_zpoints()
        update_3d_box()

        viewer.layers["ROI"].mode = "add_rectangle"

        viewer.layers["ROI"].events.data.connect(keep_single_roi)
        viewer.layers["ROI"].events.data.connect(roi_to_widget)

        viewer.layers["Zrange"].events.data.connect(keep_two_zpoints)
        viewer.layers["Zrange"].events.data.connect(zpoints_to_widget)

        make_convert_widget.y_range.changed.connect(widget_to_roi)
        make_convert_widget.x_range.changed.connect(widget_to_roi)
        make_convert_widget.z_range.changed.connect(widget_to_zpoints)

    # ---

    @magicgui(
        call_button="Convert to zarr",

        chunk_x={"label": "Chunk X", "min": 8, "max": 2**16, "step": 1, "value": 512},
        chunk_y={"label": "Chunk Y", "min": 8, "max": 2**16, "step": 1, "value": 512},
        chunk_z={"label": "Chunk Z", "min": 1, "max": 2**16, "step": 1, "value": 16},
        
        n_levels={"label": "Resolution levels", "min": 1, "max": 10, "value": 5},
        
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
        output_path: FileEdit = None,
    ):

        reader = dataset_reader(make_visualize_widget.file_format.value)
        path = make_visualize_widget.input_path.value
        scene_index = make_visualize_widget.scene_index.value
        file_format = make_visualize_widget.file_format.value

        chunks = (1, 1, chunk_z, chunk_y, chunk_x)
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
                output_path=output_path,
                file_format=file_format,
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
        
        if file_format in ["zarrV05", "zarrV04", "viventis", "luxendo"]:
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

        chunk_size = get_chunk_size(dataset.metadata["size"][0], max_size_mb=100)
        num_levels = get_n_levels(dataset.metadata["size"][0])

        make_convert_widget.chunk_x.max = dataset.metadata["size"][0][4]+1
        make_convert_widget.chunk_x.value = chunk_size[4]

        make_convert_widget.chunk_y.max = dataset.metadata["size"][0][3]+1
        make_convert_widget.chunk_y.value = chunk_size[3]

        make_convert_widget.chunk_z.max = dataset.metadata["size"][0][2]+1
        make_convert_widget.chunk_z.value = chunk_size[2]

        make_convert_widget.n_levels.value = num_levels

        make_convert_widget.t_range.max = dataset.metadata["size"][0][0] - 1
        make_convert_widget.t_range.value = (0, dataset.metadata["size"][0][0] - 1)

        make_convert_widget.z_range.max = dataset.metadata["size"][0][2] - 1
        make_convert_widget.z_range.value = (0, dataset.metadata["size"][0][2] - 1)

        make_convert_widget.y_range.max = dataset.metadata["size"][0][3] - 1
        make_convert_widget.y_range.value = (0, dataset.metadata["size"][0][3] - 1)

        make_convert_widget.x_range.max = dataset.metadata["size"][0][4] - 1
        make_convert_widget.x_range.value = (0, dataset.metadata["size"][0][4] - 1)

        n_channels = len(dataset.metadata["channel_names"])
        row_height = 20  # approx height per item in pixels
        channels_widget = make_convert_widget.channels.native
        channels_widget.setMaximumHeight(row_height * n_channels)
        make_convert_widget.channels.choices = dataset.metadata["channel_names"]
        make_convert_widget.channels.value = tuple(dataset.metadata["channel_names"])

        make_convert_widget.output_path.value = default_output

        viewer.layers.clear()

    make_visualize_widget.scene_index.enabled = False
    make_convert_widget.enabled = False
    viewer.dims.events.ndisplay.connect(lock_roi_in_3d)
    # lock_roi_in_3d()

    def sync_single_z(checked):
        if checked:
            z_val = make_convert_widget.z_range.value
            # clamp to single plane
            make_convert_widget.z_range.value = (z_val[0], z_val[0])
            make_convert_widget.z_range.enabled = False
        else:
            make_convert_widget.z_range.enabled = True

    make_convert_widget.single_z.changed.connect(sync_single_z)

    def sync_single_t(checked):
        if checked:
            t_val = make_convert_widget.t_range.value
            # clamp to single frame
            make_convert_widget.t_range.value = (t_val[0], t_val[0])
            make_convert_widget.t_range.enabled = False
        else:
            make_convert_widget.t_range.enabled = True

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

    make_convert_widget.native.layout().insertWidget(5, advanced_btn)

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
