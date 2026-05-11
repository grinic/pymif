import os
from napari import current_viewer
import numpy as np
from magicgui import magicgui
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import pymif.microscope_manager as mm
from magicgui.widgets import FileEdit
from qtpy.QtWidgets import QWidget, QVBoxLayout
from matplotlib import rc
rc('font', size=12)
rc('font', family='Arial')
# rc('font', serif='Times')
rc('pdf', fonttype=42)
# rc('text', usetex=True)

# -------------------------
# Axis-aware helpers
# -------------------------

def _dataset_axes(dataset):
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
    axes = _dataset_axes(dataset)
    spatial_axes = [ax for ax in axes if ax in "zyx"]
    scales = dataset.metadata.get("scales", [(1,) * len(spatial_axes)])
    scale_map = dict(zip(spatial_axes, scales[0]))
    return tuple(scale_map.get(ax, 1) for ax in requested_axes if ax in axes)



def draw_3d_polygon(viewer, dataset):
    """Update the 3D crop box layer from the current 2D ROI and z-range markers."""
    if any(name not in viewer.layers for name in ("ROI", "Zrange", "box")):
        return
    axes = _dataset_axes(dataset)
    if not {"z", "y", "x"}.issubset(axes):
        return

    roi_layer = viewer.layers["ROI"]
    zrange_layer = viewer.layers["Zrange"]
    zvals = [d[0] for d in zrange_layer.data]

    if len(zvals) == 0:
        zrange = [0, _axis_size(dataset, "z") - 1]
    elif len(zvals) == 1:
        zrange = [zvals[0], _axis_size(dataset, "z") - 1]
    else:
        zrange = list(zvals[:2])

    if len(roi_layer.data) == 0:
        roi = [
            [0, 0],
            [0, _axis_size(dataset, "x") - 1],
            [_axis_size(dataset, "y") - 1, _axis_size(dataset, "x") - 1],
            [_axis_size(dataset, "y") - 1, 0],
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
    """Add the editable 2D ROI rectangle layer used by the overview widget."""
    if not {"y", "x"}.issubset(_dataset_axes(dataset)):
        return None

    ymax = _axis_size(dataset, "y") - 1
    xmax = _axis_size(dataset, "x") - 1
    shapes_layer = viewer.add_shapes(
        data=np.array([[0, 0], [0, xmax], [ymax, xmax], [ymax, 0]]),
        shape_type="rectangle",
        name="ROI",
        face_color="transparent",
        edge_color="white",
        edge_width=10,
        ndim=2,
        scale=_scale_for_axes(dataset, "yx"),
    )
    shapes_layer.mode = "add_rectangle"

    def check_shape_count(event=None):
        if len(shapes_layer.data) > 0:
            shapes_layer.mode = 'select'
            draw_3d_polygon(viewer, dataset)
        else:
            if "box" in viewer.layers:
                viewer.layers["box"].data = []
            shapes_layer.mode = 'add_rectangle'

    shapes_layer.events.data.connect(check_shape_count)
    shapes_layer.events.mode.connect(check_shape_count)

    return shapes_layer


def add_zrange_layer(viewer, dataset):
    """Add the point layer used to mark the z-range for overview generation."""
    if not {"z", "y", "x"}.issubset(_dataset_axes(dataset)):
        return None

    points_layer = viewer.add_points(
        data=np.array([[0, _axis_size(dataset, "y") // 2, _axis_size(dataset, "x") // 2],
                       [_axis_size(dataset, "z") - 1, _axis_size(dataset, "y") // 2, _axis_size(dataset, "x") // 2]]),
        name="Zrange",
        face_color="white",
        ndim=3,
        size=20,
        out_of_slice_display=True,
        scale=_scale_for_axes(dataset, "zyx"),
    )
    points_layer.mode = "add"

    def check_points_count(event=None):
        if len(points_layer.data) == 1:
            if not points_layer.mode == 'select':
                points_layer.mode = 'add'
        elif len(points_layer.data) > 1:
            points_layer.mode = 'select'
            if 'box' in viewer.layers:
                viewer.layers.selection.active = viewer.layers['box']
        else:
            points_layer.mode = 'add'
        draw_3d_polygon(viewer, dataset)

    points_layer.events.data.connect(check_points_count)
    points_layer.events.mode.connect(check_points_count)
    
    return points_layer


def make_overview(dataset, viewer, output_path, overview_filename="overview", overview_format="pdf"):
    """Render per-channel and merged maximum projections for the selected ROI."""
    axes = _dataset_axes(dataset)
    if not {"y", "x"}.issubset(axes):
        raise ValueError("Overview export requires Y and X axes.")

    if "ROI" in viewer.layers and len(viewer.layers["ROI"].data) > 0:
        roi_layer = viewer.layers["ROI"]
        ymin = int(roi_layer.data[0][:, 0].min())
        xmin = int(roi_layer.data[0][:, 1].min())
        ymax = int(roi_layer.data[0][:, 0].max())
        xmax = int(roi_layer.data[0][:, 1].max())
    else:
        ymin, xmin = 0, 0
        ymax, xmax = _axis_size(dataset, "y") - 1, _axis_size(dataset, "x") - 1

    if "z" in axes and "Zrange" in viewer.layers and len(viewer.layers["Zrange"].data) > 0:
        zrange_layer = viewer.layers["Zrange"]
        zmin = int(min([d[0] for d in zrange_layer.data]))
        zmax = int(max([d[0] for d in zrange_layer.data]))
    else:
        zmin = 0
        zmax = _axis_size(dataset, "z") - 1

    slicer = []
    kept_axes = []
    for ax_name in axes:
        if ax_name == "t":
            slicer.append(0)
        elif ax_name == "c":
            slicer.append(slice(None))
            kept_axes.append("c")
        elif ax_name == "z":
            slicer.append(slice(zmin, zmax + 1))
            kept_axes.append("z")
        elif ax_name == "y":
            slicer.append(slice(ymin, ymax + 1))
            kept_axes.append("y")
        elif ax_name == "x":
            slicer.append(slice(xmin, xmax + 1))
            kept_axes.append("x")

    crop = dataset.data[0][tuple(slicer)].compute()

    # Normalize to CZYX for plotting.
    order = [kept_axes.index(ax) for ax in kept_axes if ax in "czyx"]
    crop = np.moveaxis(crop, order, range(len(order))) if order else np.asarray(crop)
    kept_axes = [ax for ax in kept_axes if ax in "czyx"]

    if "c" not in kept_axes:
        crop = np.expand_dims(crop, axis=0)
        kept_axes.insert(0, "c")
    if "z" not in kept_axes:
        c_axis = kept_axes.index("c")
        if c_axis != 0:
            crop = np.moveaxis(crop, c_axis, 0)
            kept_axes.insert(0, kept_axes.pop(c_axis))
        crop = np.expand_dims(crop, axis=1)
        kept_axes.insert(1, "z")

    # Move to CZYX explicitly if a non-standard axis order was used.
    target = [kept_axes.index(ax) for ax in "czyx" if ax in kept_axes]
    crop = np.moveaxis(crop, target, range(len(target)))

    channel_names = dataset.metadata.get("channel_names") or ["image"]
    if len(channel_names) != crop.shape[0]:
        channel_names = [f"Channel {i}" for i in range(crop.shape[0])]

    fig, ax = plt.subplots(
        nrows=1,
        ncols=len(channel_names) + 1,
        figsize=(3 * (len(channel_names) + 1), 3),
        tight_layout=True,
    )
    ax = np.atleast_1d(ax)

    merged = None

    for i, channel_name in enumerate(channel_names):
        if channel_name in viewer.layers and hasattr(viewer.layers[channel_name], "colormap"):
            colors = [list(c) for c in viewer.layers[channel_name].colormap.colors]
            pos = [j / (len(colors) - 1) for j in range(len(colors))]
            cdict = {
                "red": [[p, c[0], c[0]] for p, c in zip(pos, colors)],
                "green": [[p, c[1], c[1]] for p, c in zip(pos, colors)],
                "blue": [[p, c[2], c[2]] for p, c in zip(pos, colors)],
            }
            newcmp = LinearSegmentedColormap("pymif_channel_colormap", segmentdata=cdict, N=256)
            clims = [int(v) for v in viewer.layers[channel_name].contrast_limits]
        else:
            newcmp = "gray"
            clims = [int(np.min(crop[i])), int(np.max(crop[i])) or 1]

        img = np.max(crop[i], 0)
        ax[i].imshow(img, cmap=newcmp, vmin=clims[0], vmax=clims[1])
        ax[i].axis("off")

        norm = Normalize(vmin=clims[0], vmax=clims[1], clip=True)
        img_norm = norm(img)
        img_rgb = plt.get_cmap(newcmp)(img_norm)[..., :3] if isinstance(newcmp, str) else newcmp(img_norm)[..., :3]

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

# -------------------------
# MagicGUI controls
# -------------------------

def overview_widget():
    """Create the napari widget for selecting an ROI and exporting overview figures."""
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

        if "box" not in viewer.layers and {"z", "y", "x"}.issubset(_dataset_axes(dataset)):
            viewer.add_shapes(
                data=[],
                shape_type="polygon",
                name="box",
                edge_color="coral",
                face_color="cyan",
                ndim=3,
                scale=_scale_for_axes(dataset, "zyx"),
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
