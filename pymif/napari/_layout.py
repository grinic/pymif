"""Layout helpers that keep the PyMIF napari widgets usable at any dock size."""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

#: Narrow enough for a side dock; the content scrolls horizontally below this.
MIN_WIDTH = 260
#: Minimum heights so that the log always shows several lines and the
#: scrollable part always shows something.
MIN_LOG_HEIGHT = 140
MIN_CONTENT_HEIGHT = 120


def make_section(*widgets, margins=(6, 4, 6, 6), spacing=3):
    """Wrap widgets in a rounded, bordered frame to visually separate a block.

    The border uses a translucent grey so it reads on both light and dark
    napari themes. More widgets can be added later via ``section.layout()``.
    """
    frame = QFrame()
    frame.setObjectName("pymifSection")
    frame.setStyleSheet(
        "QFrame#pymifSection { border: 1px solid rgba(128, 128, 128, 0.55);"
        " border-radius: 6px; }"
    )
    inner = QVBoxLayout(frame)
    inner.setContentsMargins(*margins)
    inner.setSpacing(spacing)
    for w in widgets:
        inner.addWidget(w)
    return frame


def make_footer(button: QPushButton) -> QWidget:
    """A bar holding the main action button, kept outside of any scroll area."""
    footer = QWidget()
    layout = QHBoxLayout(footer)
    layout.setContentsMargins(0, 2, 0, 2)
    button.setMinimumHeight(30)
    button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    layout.addWidget(button)
    return footer


def compose_dock(content: QWidget, footer: QWidget | None = None, bottom: QWidget | None = None) -> QWidget:
    """Build the dock widget: scrollable content, an always-visible footer, a resizable bottom pane.

    * ``content`` goes in a scroll area, so nothing is ever cut off, whatever the
      size or position of the dock (side or bottom).
    * ``footer`` (typically the main action button) stays visible below it.
    * ``bottom`` (typically the log) sits under a splitter handle, so it can be
      dragged taller or shorter.
    """
    container = QWidget()
    container.setMinimumWidth(MIN_WIDTH)
    container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    outer = QVBoxLayout(container)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setWidget(content)
    scroll.setMinimumHeight(MIN_CONTENT_HEIGHT)

    top = QWidget()
    top_layout = QVBoxLayout(top)
    top_layout.setContentsMargins(0, 0, 0, 0)
    top_layout.setSpacing(2)
    top_layout.addWidget(scroll, 1)
    if footer is not None:
        top_layout.addWidget(footer, 0)

    if bottom is None:
        outer.addWidget(top)
        return container

    bottom.setMinimumHeight(MIN_LOG_HEIGHT)
    splitter = QSplitter(Qt.Vertical)
    splitter.setChildrenCollapsible(False)
    splitter.addWidget(top)
    splitter.addWidget(bottom)
    splitter.setStretchFactor(0, 3)
    splitter.setStretchFactor(1, 2)
    splitter.setSizes([480, 260])
    outer.addWidget(splitter)
    container.splitter = splitter  # exposed for tests / callers
    return container
