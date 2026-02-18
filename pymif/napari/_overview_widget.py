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
# Your existing functions
# -------------------------

def draw_3d_polygon(viewer, dataset):
    roi_layer = viewer.layers["ROI"]
    zrange_layer = viewer.layers["Zrange"]
    zvals = [d[0] for d in zrange_layer.data]

    if len(zvals) == 0:
        zrange = [0, dataset.data[0].shape[2]]
    elif len(zvals) == 1:
        zrange = [zvals[0], dataset.data[0].shape[2]]
    else:
        zrange = list(zvals[:2])

    if len(roi_layer.data) == 0:
        roi = [
            [0, 0],
            [0, dataset.data[0].shape[4]],
            [dataset.data[0].shape[3], dataset.data[0].shape[4]],
            [dataset.data[0].shape[3], 0],
        ]
    else:
        roi = [list(r) for r in roi_layer.data[0]]

    polygon = [
        [[zrange[0]] + roi[0], [zrange[0]] + roi[1], [zrange[0]] + roi[2], [zrange[0]] + roi[3]],
        [[zrange[1]] + roi[0], [zrange[1]] + roi[1], [zrange[1]] + roi[2], [zrange[1]] + roi[3]],
        [[zrange[0]] + roi[0], [zrange[1]] + roi[0], [zrange[1]] + roi[1], [zrange[0]] + roi[1]],
        [[zrange[0]] + roi[0], [zrange[1]] + roi[0], [zrange[1]] + roi[3], [zrange[0]] + roi[3]],
        [[zrange[0]] + roi[1], [zrange[1]] + roi[1], [zrange[1]] + roi[2], [zrange[0]] + roi[2]],
        [[zrange[0]] + roi[3], [zrange[1]] + roi[3], [zrange[1]] + roi[2], [zrange[0]] + roi[2]],
    ]

    viewer.layers["box"].data = polygon
    viewer.layers["box"].refresh()

def add_roi_layer(viewer, dataset):
    shapes_layer = viewer.add_shapes(
        data=np.empty((0, 4, 2)),
        shape_type="rectangle",
        name="ROI",
        face_color="transparent",
        edge_color="white",
        edge_width=10,
        ndim=2,
        scale=dataset.metadata["scales"][0][1:],
    )
    shapes_layer.mode = "add_rectangle"

    def check_shape_count(event=None):

        if len(shapes_layer.data) > 0:
            shapes_layer.mode = 'select'
            draw_3d_polygon(viewer, dataset)
        else:
            viewer.layers["box"].data = []
            shapes_layer.mode = 'add_rectangle'

        # shapes_layer.refresh()

    # Add/delete shapes
    shapes_layer.events.data.connect(check_shape_count)

    # Mode changes
    shapes_layer.events.mode.connect(check_shape_count)

    return shapes_layer

def add_zrange_layer(viewer, dataset):
    points_layer = viewer.add_points(
        data=np.empty((0, 2, 3)),
        name="Zrange",
        face_color="white",
        ndim=3,
        size=20,
        out_of_slice_display=True,
        scale=dataset.metadata["scales"][0],
    )
    points_layer.mode = "add"

    def check_points_count(event=None):

        if len(points_layer.data) == 1:
            if not points_layer.mode == 'select':
                points_layer.mode = 'add'
        elif len(points_layer.data) > 1:
            points_layer.mode = 'select'
            viewer.layers.selection.active = viewer.layers['box']
        else:
            points_layer.mode = 'add'
        draw_3d_polygon(viewer, dataset)

    # Add/delete shapes
    points_layer.events.data.connect(check_points_count)

    # Mode changes
    points_layer.events.mode.connect(check_points_count)
    
    return points_layer

def make_overview(dataset, viewer, output_path, overview_filename="overview", overview_format="pdf"):
    roi_layer = viewer.layers["ROI"]
    zrange_layer = viewer.layers["Zrange"]

    ymin = int(roi_layer.data[0][:, 0].min())
    xmin = int(roi_layer.data[0][:, 1].min())
    ymax = int(roi_layer.data[0][:, 0].max())
    xmax = int(roi_layer.data[0][:, 1].max())

    if len(zrange_layer.data)>0:
        zmin = int(min([d[0] for d in zrange_layer.data]))
        zmax = int(max([d[0] for d in zrange_layer.data]))
    else:
        zmin = 0
        zmax = dataset.metadata["size"][0][2]

    crop = dataset.data[0][0, :, zmin:zmax, ymin:ymax, xmin:xmax].compute()

    channel_names = dataset.metadata["channel_names"]
    fig, ax = plt.subplots(
        nrows=1,
        ncols=len(channel_names) + 1,
        figsize=(3 * (len(channel_names) + 1), 3),
        tight_layout=True,
    )

    merged = None

    for i, channel_name in enumerate(channel_names):
        colors = [list(c) for c in viewer.layers[channel_name].colormap.colors]
        pos = [j / (len(colors) - 1) for j in range(len(colors))]

        cdict = {
            "red": [[p, c[0], c[0]] for p, c in zip(pos, colors)],
            "green": [[p, c[1], c[1]] for p, c in zip(pos, colors)],
            "blue": [[p, c[2], c[2]] for p, c in zip(pos, colors)],
        }

        newcmp = LinearSegmentedColormap("testCmap", segmentdata=cdict, N=256)
        clims = [int(v) for v in viewer.layers[channel_name].contrast_limits]

        img = np.max(crop[i], 0)
        ax[i].imshow(img, cmap=newcmp, vmin=clims[0], vmax=clims[1])
        ax[i].axis("off")

        norm = Normalize(vmin=clims[0], vmax=clims[1], clip=True)
        img_norm = norm(img)
        img_rgb = newcmp(img_norm)[..., :3]

        if merged is None:
            merged = img_rgb.copy()
        else:
            merged += img_rgb
            merged = np.clip(merged, 0, 1)

    ax[-1].imshow(merged)
    ax[-1].axis("off")
    plt.show()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    fig.savefig(output_path / f"{overview_filename}.{overview_format}", dpi=300)
    # plt.close(fig)

# -------------------------
# MagicGUI controls
# -------------------------

def overview_widget():
    viewer = current_viewer()
    _state = {"dataset": None}

    @magicgui(
        call_button="Make Overview",
        input_path={"widget_type": "FileEdit", "mode": "d"},
        output_path={"widget_type": "FileEdit", "mode": "d"},
        overview_format={"choices": ["tiff", "png", "jpeg", "pdf"]},
    )
    def make_overview_widget_inner(
        input_path: FileEdit = None,
        output_path: FileEdit = None,
        overview_filename: str = "overview",
        overview_format: str = "pdf",
    ):
        dataset = _state["dataset"]
        if dataset is None:
            return

        make_overview(
            dataset,
            viewer,
            output_path=output_path,
            overview_filename=overview_filename,
            overview_format=overview_format,
        )

    @magicgui(call_button="Reset ROI and Zrange")
    def reset_roi_widget():
        if "ROI" in viewer.layers:
            viewer.layers["ROI"].data = np.empty((0, 4, 2))
        if "Zrange" in viewer.layers:
            viewer.layers["Zrange"].data = np.empty((0, 2, 3))
        if "box" in viewer.layers:
            viewer.layers["box"].data = []

    # -------------------------
    # Input path callback
    # -------------------------

    @make_overview_widget_inner.input_path.changed.connect
    def _on_input_path_change(path):
        if not path:
            return

        dataset = mm.ZarrManager(path)
        _state["dataset"] = dataset

        default_output = path.parent / "overview"
        make_overview_widget_inner.output_path.value = default_output

        viewer.layers.clear()
        viewer.open(path, plugin="napari-ome-zarr")

        if "ROI" not in viewer.layers:
            add_roi_layer(viewer, dataset)

        if "Zrange" not in viewer.layers:
            add_zrange_layer(viewer, dataset)

        if "box" not in viewer.layers:
            viewer.add_shapes(
                data=[],
                shape_type="polygon",
                name="box",
                edge_color="coral",
                face_color="cyan",
                ndim=3,
                scale=dataset.metadata["scales"][0],
            )

    @make_overview_widget_inner.input_path.changed.connect
    def _enable_buttons(path):
        enabled = bool(path)
        make_overview_widget_inner.call_button.enabled = enabled
        reset_roi_widget.call_button.enabled = enabled

    make_overview_widget_inner.call_button.enabled = False
    reset_roi_widget.call_button.enabled = False

    # -------------------------
    # Compose single dock widget
    # -------------------------

    container = QWidget()
    layout = QVBoxLayout(container)
    layout.addWidget(make_overview_widget_inner.native)
    layout.addWidget(reset_roi_widget.native)

    make_overview_widget_inner.input_path.tooltip = (
        "Select a folder containing OME-Zarr data (.zarr)"
    )

    return container
