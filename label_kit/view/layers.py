"""Non-destructive overlay layers.

A layer draws *on top of* the frame without ever touching its pixels. This is
the direct replacement for the old ``cv2.putText`` that burned the frame counter
into the decoded array — an approach that made the overlay impossible to toggle,
impossible to click through, blurry at any zoom, and permanently baked into
anything cropped out of that frame and sent to a model.

Two flavours, distinguished by which coordinate space they draw in:

* :class:`SceneLayer` — image pixel coordinates. Pans and zooms with the frame.
  Use for anything anchored to image content: guide lines, grids, ROI markers.
* :class:`OverlayLayer` — widget coordinates. Fixed to the viewport. Use for
  HUD: counters, readouts, status text.

Both are painted above the annotation items, ordered by :attr:`Layer.z`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetricsF, QPainter, QPen

from ..core.events import Event
from ..core.frames import Frame
from ..core.geometry import Point, Size, ViewTransform
from .theme import Theme

__all__ = [
    "BusyLayer",
    "CrosshairLayer",
    "FrameInfoLayer",
    "Layer",
    "LayerContext",
    "OverlayLayer",
    "SceneLayer",
    "draw_hud_text",
]

_HUD_MARGIN = 10.0
_HUD_PAD = 5.0
_HUD_FONT_PX = 12


@dataclass(frozen=True)
class LayerContext:
    """Everything a layer may need, handed over at paint time.

    Passing state in rather than letting layers hold a canvas reference keeps
    them independently testable and stops a layer from mutating the view it is
    drawing into.
    """

    frame: Frame | None
    transform: ViewTransform
    theme: Theme
    viewport: QRectF
    image_size: Size
    cursor: Point | None = None

    @property
    def has_frame(self) -> bool:
        return self.frame is not None


def draw_hud_text(
    painter: QPainter,
    origin: QPointF,
    text: str,
    theme: Theme,
    *,
    align_right: bool = False,
) -> QRectF:
    """Draw a readable text chip in widget coordinates. Returns its rect.

    A translucent plate behind the text is what makes an overlay legible over
    both a blown-out sky and a black tunnel, which plain drawn text is not.
    """
    font = QFont()
    font.setPixelSize(_HUD_FONT_PX)
    painter.setFont(font)
    metrics = QFontMetricsF(font)

    width = metrics.horizontalAdvance(text) + 2 * _HUD_PAD
    height = metrics.height() + _HUD_PAD
    left = origin.x() - width if align_right else origin.x()
    chip = QRectF(left, origin.y(), width, height)

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(QColor(theme.hud_bg)))
    painter.drawRoundedRect(chip, 3, 3)

    painter.setPen(QPen(QColor(theme.hud_fg)))
    painter.drawText(
        chip.adjusted(_HUD_PAD, 0, -_HUD_PAD, 0),
        int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
        text,
    )
    return chip


class Layer(ABC):
    """Base class for everything drawn over the frame.

    Subclass :class:`SceneLayer` or :class:`OverlayLayer` rather than this.
    """

    #: Unique id. Namespace it with your plugin id.
    id: str = ""
    #: Human-readable name, used for the View ▸ Overlays toggle.
    title: str = ""
    #: Paint order. Higher draws later, i.e. on top.
    z: float = 0.0
    #: Whether this layer appears in the overlay toggle menu.
    user_toggleable: bool = True
    #: Optional shortcut for the generated visibility toggle.
    shortcut: str | None = None
    #: Plugin that contributed this layer, filled in by ``PluginContext``.
    owner: str | None = None

    def __init__(self, layer_id: str = "", title: str = "", z: float | None = None) -> None:
        if layer_id:
            self.id = layer_id
        if title:
            self.title = title
        if z is not None:
            self.z = z
        self._visible = True
        self.changed: Event[str] = Event(f"layer.{self.id}.changed")

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        value = bool(value)
        if value != self._visible:
            self._visible = value
            self.changed.emit(self.id)

    def toggle(self) -> bool:
        self.visible = not self._visible
        return self._visible

    def invalidate(self) -> None:
        """Ask the canvas to repaint. Call after changing layer parameters."""
        self.changed.emit(self.id)

    def on_frame(self, frame: Frame | None) -> None:
        """Hook: a new frame arrived. Override to precompute per-frame state."""

    def on_attach(self) -> None:
        """Hook: the layer was added to a canvas."""

    def on_detach(self) -> None:
        """Hook: the layer was removed. Release any resources here."""

    @abstractmethod
    def paint(self, painter: QPainter, ctx: LayerContext) -> None: ...


class SceneLayer(Layer):
    """Draws in image pixel coordinates. Moves and scales with the frame.

    The painter arrives with the view transform already applied, so drawing at
    ``(100, 50)`` lands on image pixel ``(100, 50)`` at any zoom. Use cosmetic
    pens for lines that should keep a constant on-screen width.
    """

    space = "scene"


class OverlayLayer(Layer):
    """Draws in widget coordinates. Pinned to the viewport, ignores pan/zoom."""

    space = "widget"


# ── built-in layers ─────────────────────────────────────────────────────


class FrameInfoLayer(OverlayLayer):
    """Frame counter and timestamp, top-left.

    The non-destructive replacement for burning the counter into the pixels.
    """

    id = "core.frame_info"
    title = "Frame Counter"
    z = 100.0

    def __init__(self, show_time: bool = True) -> None:
        super().__init__()
        self.show_time = show_time
        self.frame_count = 0
        self.fps = 0.0
        self.visible = False

    def set_media(self, frame_count: int, fps: float) -> None:
        self.frame_count = frame_count
        self.fps = fps
        self.invalidate()

    def paint(self, painter: QPainter, ctx: LayerContext) -> None:
        if ctx.frame is None:
            return
        index = ctx.frame.index
        text = f"{index} / {max(self.frame_count - 1, 0)}" if self.frame_count else str(index)
        if self.show_time and self.fps > 0:
            text += f"   {_fmt_time(index / self.fps)}"
        draw_hud_text(painter, QPointF(_HUD_MARGIN, _HUD_MARGIN), text, ctx.theme)


class CrosshairLayer(OverlayLayer):
    """Full-viewport crosshair plus a pixel-coordinate readout.

    Precise box placement is guesswork without this; it is the difference
    between "roughly on the traffic light" and a reproducible pixel coordinate.
    """

    id = "core.crosshair"
    title = "Crosshair"
    z = 90.0

    def __init__(self) -> None:
        super().__init__()
        self.visible = False

    def paint(self, painter: QPainter, ctx: LayerContext) -> None:
        if ctx.cursor is None:
            return
        widget_pos = ctx.transform.image_to_widget(ctx.cursor)
        x, y = widget_pos.x, widget_pos.y

        pen = QPen(QColor(ctx.theme.crosshair_fg))
        pen.setWidthF(1.0)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(QPointF(ctx.viewport.left(), y), QPointF(ctx.viewport.right(), y))
        painter.drawLine(QPointF(x, ctx.viewport.top()), QPointF(x, ctx.viewport.bottom()))

        readout = f"{int(ctx.cursor.x)}, {int(ctx.cursor.y)}"
        if ctx.frame is not None and _inside(ctx.cursor, ctx.image_size):
            r, g, b = ctx.frame.image[int(ctx.cursor.y), int(ctx.cursor.x)]
            readout += f"   rgb({r}, {g}, {b})"
        draw_hud_text(
            painter,
            QPointF(ctx.viewport.right() - _HUD_MARGIN, ctx.viewport.bottom() - 28),
            readout,
            ctx.theme,
            align_right=True,
        )


class BusyLayer(OverlayLayer):
    """Shows which models are currently running.

    Inference on CPU takes seconds. Without visible feedback the app looks
    frozen and users click the same button again, queueing more work.
    """

    id = "core.busy"
    title = "Inference Status"
    z = 200.0
    user_toggleable = False

    def __init__(self) -> None:
        super().__init__()
        self._running: dict[str, str] = {}

    def set_running(self, key: str, label: str) -> None:
        self._running[key] = label
        self.invalidate()

    def clear_running(self, key: str) -> None:
        if self._running.pop(key, None) is not None:
            self.invalidate()

    @property
    def is_busy(self) -> bool:
        return bool(self._running)

    def paint(self, painter: QPainter, ctx: LayerContext) -> None:
        if not self._running:
            return
        y = _HUD_MARGIN
        for label in self._running.values():
            chip = draw_hud_text(
                painter,
                QPointF(ctx.viewport.right() - _HUD_MARGIN, y),
                f"⏳ {label}",
                ctx.theme,
                align_right=True,
            )
            y += chip.height() + 4


def _inside(p: Point, size: Size) -> bool:
    return 0 <= p.x < size.width and 0 <= p.y < size.height


def _fmt_time(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"
