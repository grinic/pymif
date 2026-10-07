from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("napari")

from napari.components import ViewerModel
from qtpy.QtWidgets import QApplication, QPushButton, QScrollArea, QSplitter, QTextEdit, QWidget

import pymif.napari._convert_widget as convert_module
import pymif.napari._export_widget as export_module
import pymif.napari._overview_widget as overview_module
from pymif.napari._layout import MIN_LOG_HEIGHT, compose_dock


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def converter(app, monkeypatch):
    # The plugin redirects stdout/stderr into its log box; undo that after the test.
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    monkeypatch.setattr(convert_module, "current_viewer", lambda: ViewerModel())
    widget = convert_module.convert_widget()
    widget._log_streams = (sys.stdout, sys.stderr)  # the plugin's log-box streams
    yield widget
    widget.close()


def convert_button(widget):
    return next(b for b in widget.findChildren(QPushButton) if b.text() == "Convert to zarr")


def inside_scroll_area(widget):
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QScrollArea):
            return True
        parent = parent.parentWidget()
    return False


@pytest.mark.parametrize("size", [(300, 340), (420, 760), (900, 300)])
def test_convert_button_always_visible(converter, app, size):
    converter.resize(*size)
    converter.show()
    app.processEvents()
    button = convert_button(converter)
    assert button.isVisible()
    assert not inside_scroll_area(button)  # fixed footer: never scrolls away
    bottom = button.mapTo(converter, button.rect().bottomLeft()).y()
    assert 0 < bottom <= size[1]


def test_log_is_tall_and_resizable(converter, app):
    log = converter.findChild(QTextEdit)
    assert log.maximumHeight() > 10_000  # no fixed cap any more
    assert log.minimumHeight() >= 100

    splitter = converter.findChild(QSplitter)
    assert splitter is not None and splitter.count() == 2
    assert splitter.widget(1).minimumHeight() >= MIN_LOG_HEIGHT
    converter.resize(420, 760)
    converter.show()
    app.processEvents()
    assert log.height() >= 100  # several lines, not two or three


def test_parameters_scroll_instead_of_being_cut_off(converter):
    scroll = converter.findChild(QScrollArea)
    assert scroll is not None and scroll.widgetResizable()
    assert converter.minimumWidth() <= 300  # fits a narrow side dock


def test_other_widgets_are_scrollable_docks(app, monkeypatch):
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    for module, factory in ((overview_module, "overview_widget"), (export_module, "export_widget")):
        monkeypatch.setattr(module, "current_viewer", lambda: ViewerModel(), raising=False)
        widget = getattr(module, factory)()
        assert widget.findChild(QScrollArea) is not None, factory


def test_compose_dock_without_footer_or_bottom(app):
    dock = compose_dock(QWidget())
    assert dock.findChild(QScrollArea) is not None and dock.findChild(QSplitter) is None


def test_convert_button_starts_disabled_until_a_dataset_is_loaded(converter):
    # Same as the panel it was moved out of: nothing to convert yet.
    assert not convert_button(converter).isEnabled()


def test_footer_button_still_runs_the_conversion_and_locks_while_running(converter, app, monkeypatch):
    """The button was moved out of the panel: it must still trigger the call and be locked while it runs."""
    import time

    calls = []

    def fake_conversion(**kwargs):
        calls.append(kwargs)
        time.sleep(0.3)  # long enough to observe the locked state
        return "converted.zarr"

    monkeypatch.setattr(convert_module, "_run_conversion", fake_conversion)

    button = convert_button(converter)
    log = converter.findChild(QTextEdit)
    # pytest swaps sys.stdout between setup and test body; point it back to the log box
    monkeypatch.setattr(sys, "stdout", converter._log_streams[0])
    monkeypatch.setattr(sys, "stderr", converter._log_streams[1])
    button.parentWidget().setEnabled(True)  # the footer; unlocked as after loading a dataset
    button.click()
    assert not button.isEnabled()  # locked at once: a second click must not start a second job
    button.click()  # ignored while disabled

    deadline = time.time() + 20
    while not button.isEnabled() and time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()
    assert button.isEnabled(), "button must be re-enabled when the conversion ends"
    assert len(calls) == 1  # exactly one conversion was started
    assert "Conversion completed: converted.zarr" in log.toPlainText()


def conversion_box(button):
    """The bordered 'Dataset conversion' frame that contains ``button``, or ``None``."""
    parent = button.parentWidget()
    while parent is not None:
        if parent.objectName() == "pymifSection":
            return parent
        parent = parent.parentWidget()
    return None


def conversion_toggle(widget):
    from qtpy.QtWidgets import QToolButton

    return next(b for b in widget.findChildren(QToolButton) if b.text().startswith("Dataset conversion"))


@pytest.mark.parametrize("expanded", [False, True])
def test_convert_button_stays_inside_the_dataset_conversion_box(converter, app, expanded):
    toggle = conversion_toggle(converter)
    toggle.setChecked(expanded)
    converter.resize(420, 760)
    converter.show()
    app.processEvents()

    button = convert_button(converter)
    box = conversion_box(button)
    assert box is not None, "the Convert button must live in a section frame"
    assert toggle in box.findChildren(type(toggle)), "...and that frame is the 'Dataset conversion' one"
    assert button.isVisible()  # also while the parameters are collapsed
    assert box.rect().contains(button.mapTo(box, button.rect().center()))
    bottom = button.mapTo(converter, button.rect().bottomLeft()).y()
    assert 0 < bottom <= converter.height()


def test_collapsing_hides_the_parameters_but_not_the_title_or_button(converter, app):
    toggle = conversion_toggle(converter)
    button = convert_button(converter)
    converter.resize(420, 760)
    converter.show()

    toggle.setChecked(True)
    app.processEvents()
    params = [w for w in conversion_box(button).findChildren(QScrollArea)]
    assert params and all(p.isVisible() for p in params)

    toggle.setChecked(False)
    app.processEvents()
    assert all(not p.isVisible() for p in params)
    assert toggle.isVisible() and button.isVisible()
