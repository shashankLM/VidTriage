"""Interaction tools — how a mouse gesture becomes a geometry.

A tool receives events whose coordinates are **already in image pixels**, and
publishes a :class:`ToolResult` when the user completes a gesture. It knows
nothing about models, annotations or the store; it just turns a drag into a
:class:`~vidtriage.core.geometry.Rect`.

That single ``produced`` channel is the seam the whole prompt workflow hangs
off. A plugin wanting "drag a box, run SAM inside it" subscribes to the canvas's
``tool_result`` signal and filters on ``tool_id`` — it does not touch the canvas,
and adding a brand-new gesture means registering a new ``Tool`` subclass and
nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF

from ..core.events import Event
from ..core.geometry import Geometry, Point, Polygon, Rect
from .layers import LayerContext

__all__ = [
    "BoxTool",
    "PanTool",
    "PointTool",
    "PolygonTool",
    "SelectTool",
    "Tool",
    "ToolEvent",
    "ToolResult",
    "builtin_tools",
]

_MIN_DRAG_PX = 3.0
_CLOSE_POLYGON_PX = 10.0


@dataclass(frozen=True)
class ToolEvent:
    """A mouse event, pre-translated into image space."""

    image_pos: Point
    widget_pos: Point
    button: Qt.MouseButton = Qt.MouseButton.NoButton
    buttons: Qt.MouseButton = Qt.MouseButton.NoButton
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier
    view_scale: float = 1.0

    @property
    def is_shift(self) -> bool:
        return bool(self.modifiers & Qt.KeyboardModifier.ShiftModifier)

    @property
    def is_ctrl(self) -> bool:
        return bool(self.modifiers & Qt.KeyboardModifier.ControlModifier)

    @property
    def is_alt(self) -> bool:
        return bool(self.modifiers & Qt.KeyboardModifier.AltModifier)

    def image_distance(self, screen_pixels: float) -> float:
        return screen_pixels / max(self.view_scale, 1e-9)


@dataclass(frozen=True)
class ToolResult:
    """A completed gesture.

    Args:
        tool_id: Which tool produced it — subscribers filter on this.
        geometry: The shape, in image pixel coordinates.
        positive: For point prompts, whether this is an include (``True``) or
            exclude (``False``) point. SAM-style models use both.
        additive: The user held Shift, meaning "add to what is already there"
            rather than "replace it".
    """

    tool_id: str
    geometry: Geometry
    positive: bool = True
    additive: bool = False


class Tool:
    """Base for every interaction mode.

    Deliberately concrete rather than abstract: every hook is optional, and
    :class:`SelectTool` and :class:`PanTool` are useful with none of them
    overridden — they work by *letting events through* to the graphics items and
    the view's own drag mode.

    Mouse handlers return ``True`` when they consume the event. Returning
    ``False`` lets it fall through to the graphics items, which is how
    :class:`SelectTool` gets item dragging for free.
    """

    #: Unique id. Namespace it with your plugin id.
    id: str = ""
    #: Menu / toolbar label.
    title: str = ""
    #: Optional one-line hint shown in the status bar while active.
    hint: str = ""
    #: Cursor shown over the canvas while this tool is active.
    cursor: Qt.CursorShape = Qt.CursorShape.ArrowCursor
    #: Optional keyboard shortcut. The shell turns it into a command.
    shortcut: str | None = None
    #: Whether graphics items should stay interactive while this tool is active.
    items_interactive: bool = False
    #: Plugin that contributed this tool, filled in by ``PluginContext``.
    owner: str | None = None

    def __init__(self) -> None:
        self.produced: Event[ToolResult] = Event(f"tool.{self.id}.produced")
        self.preview_changed: Event[str] = Event(f"tool.{self.id}.preview")

    # ── lifecycle ───────────────────────────────────────────────────────

    def activate(self) -> None:
        """Called when the tool becomes active."""

    def deactivate(self) -> None:
        """Called when another tool takes over. Must drop in-progress state."""
        self.reset()

    def reset(self) -> None:
        """Abandon any partial gesture."""

    # ── events ──────────────────────────────────────────────────────────

    def mouse_press(self, event: ToolEvent) -> bool:
        return False

    def mouse_move(self, event: ToolEvent) -> bool:
        return False

    def mouse_release(self, event: ToolEvent) -> bool:
        return False

    def mouse_double_click(self, event: ToolEvent) -> bool:
        return False

    def key_press(self, key: int, modifiers: Qt.KeyboardModifier) -> bool:
        return False

    # ── preview ─────────────────────────────────────────────────────────

    def paint_preview(self, painter: QPainter, ctx: LayerContext) -> None:
        """Draw the in-progress gesture, in scene (image) coordinates."""

    def _emit(self, geometry: Geometry, *, positive: bool = True, additive: bool = False) -> None:
        self.produced.emit(ToolResult(self.id, geometry, positive, additive))

    @staticmethod
    def _preview_pen(ctx: LayerContext, dashed: bool = True) -> QPen:
        pen = QPen(ctx.theme.color("selection_fg"))
        pen.setCosmetic(True)
        pen.setWidthF(2.0)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        return pen


# ── built-ins ───────────────────────────────────────────────────────────


class SelectTool(Tool):
    """Default mode: click annotations to select, drag them to edit."""

    id = "select"
    title = "Select"
    hint = "Click to select · drag to move · Delete to remove"
    cursor = Qt.CursorShape.ArrowCursor
    shortcut = "V"
    items_interactive = True


class PanTool(Tool):
    """Drag to pan. Also reachable at any time with the middle mouse button."""

    id = "pan"
    title = "Pan"
    hint = "Drag to pan · scroll to zoom"
    cursor = Qt.CursorShape.OpenHandCursor
    shortcut = "H"
    items_interactive = False


class PointTool(Tool):
    """Click to drop a point prompt.

    Left click is an include point, right click (or Ctrl+left) an exclude point
    — the interaction SAM-family models expect.
    """

    id = "point"
    title = "Point"
    hint = "Click to add a point · right-click for a negative point · Shift to accumulate"
    cursor = Qt.CursorShape.CrossCursor
    shortcut = "P"

    def mouse_press(self, event: ToolEvent) -> bool:
        if event.button not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            return False
        positive = event.button == Qt.MouseButton.LeftButton and not event.is_ctrl
        self._emit(event.image_pos, positive=positive, additive=event.is_shift)
        return True


class BoxTool(Tool):
    """Drag a rectangle.

    Feeds both hand-drawn boxes and box prompts for detection/segmentation
    models. Drags shorter than a few pixels are treated as a stray click and
    dropped, so a slightly-shaky click never creates a degenerate 1px box.
    """

    id = "box"
    title = "Box"
    hint = "Drag a rectangle · Shift to accumulate · Esc to cancel"
    cursor = Qt.CursorShape.CrossCursor
    shortcut = "B"

    def __init__(self) -> None:
        super().__init__()
        self._origin: Point | None = None
        self._current: Point | None = None
        self._additive = False

    def reset(self) -> None:
        if self._origin is not None:
            self._origin = None
            self._current = None
            self.preview_changed.emit(self.id)

    def mouse_press(self, event: ToolEvent) -> bool:
        if event.button != Qt.MouseButton.LeftButton:
            return False
        self._origin = event.image_pos
        self._current = event.image_pos
        self._additive = event.is_shift
        return True

    def mouse_move(self, event: ToolEvent) -> bool:
        if self._origin is None:
            return False
        self._current = event.image_pos
        self.preview_changed.emit(self.id)
        return True

    def mouse_release(self, event: ToolEvent) -> bool:
        if self._origin is None or event.button != Qt.MouseButton.LeftButton:
            return False
        origin, self._origin = self._origin, None
        self._current = None
        self.preview_changed.emit(self.id)

        rect = Rect.from_corners(origin, event.image_pos)
        minimum = event.image_distance(_MIN_DRAG_PX)
        if rect.width < minimum and rect.height < minimum:
            return True
        self._emit(rect, additive=self._additive)
        return True

    def key_press(self, key: int, modifiers: Qt.KeyboardModifier) -> bool:
        if key == Qt.Key.Key_Escape and self._origin is not None:
            self.reset()
            return True
        return False

    def paint_preview(self, painter: QPainter, ctx: LayerContext) -> None:
        if self._origin is None or self._current is None:
            return
        rect = Rect.from_corners(self._origin, self._current)
        painter.setPen(self._preview_pen(ctx))
        fill = QColor(ctx.theme.color("selection_fg"))
        fill.setAlpha(40)
        painter.setBrush(QBrush(fill))
        painter.drawRect(rect.x1, rect.y1, rect.width, rect.height)


class PolygonTool(Tool):
    """Click vertices; double-click, Enter, or click the first vertex to close."""

    id = "polygon"
    title = "Polygon"
    hint = "Click to add points · double-click or Enter to close · Backspace to undo · Esc to cancel"
    cursor = Qt.CursorShape.CrossCursor
    shortcut = "G"

    def __init__(self) -> None:
        super().__init__()
        self._points: list[Point] = []
        self._cursor: Point | None = None
        self._additive = False

    def reset(self) -> None:
        if self._points:
            self._points = []
            self._cursor = None
            self.preview_changed.emit(self.id)

    def mouse_press(self, event: ToolEvent) -> bool:
        if event.button == Qt.MouseButton.RightButton:
            self._undo_point()
            return True
        if event.button != Qt.MouseButton.LeftButton:
            return False

        if self._points:
            snap = event.image_distance(_CLOSE_POLYGON_PX)
            if event.image_pos.distance_to(self._points[0]) <= snap:
                self._close()
                return True
        else:
            self._additive = event.is_shift

        self._points.append(event.image_pos)
        self.preview_changed.emit(self.id)
        return True

    def mouse_move(self, event: ToolEvent) -> bool:
        if not self._points:
            return False
        self._cursor = event.image_pos
        self.preview_changed.emit(self.id)
        return True

    def mouse_double_click(self, event: ToolEvent) -> bool:
        if self._points:
            self._close()
            return True
        return False

    def key_press(self, key: int, modifiers: Qt.KeyboardModifier) -> bool:
        if key == Qt.Key.Key_Escape:
            self.reset()
            return True
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._close()
            return True
        if key == Qt.Key.Key_Backspace:
            self._undo_point()
            return True
        return False

    def _undo_point(self) -> None:
        if self._points:
            self._points.pop()
            self.preview_changed.emit(self.id)

    def _close(self) -> None:
        polygon = Polygon(tuple(self._points))
        self._points = []
        self._cursor = None
        self.preview_changed.emit(self.id)
        if polygon.is_valid:
            self._emit(polygon, additive=self._additive)

    def paint_preview(self, painter: QPainter, ctx: LayerContext) -> None:
        if not self._points:
            return
        points = [QPointF(p.x, p.y) for p in self._points]
        if self._cursor is not None:
            points.append(QPointF(self._cursor.x, self._cursor.y))

        pen = self._preview_pen(ctx)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolyline(QPolygonF(points))

        # Emphasise the first vertex: clicking it closes the shape.
        radius = ctx.transform.widget_length_to_image(4.0)
        painter.setBrush(QBrush(ctx.theme.color("selection_fg")))
        painter.drawEllipse(points[0], radius * 1.5, radius * 1.5)
        for point in points[1:len(self._points)]:
            painter.drawEllipse(point, radius, radius)


def builtin_tools() -> list[Tool]:
    """The tools the app always provides, in menu order."""
    return [SelectTool(), PanTool(), BoxTool(), PointTool(), PolygonTool()]
