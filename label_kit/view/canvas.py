"""ImageCanvas — the robust image widget.

Everything visual happens here, and everything here is composable:

* **The frame is never modified.** Pixels go on screen as-is; overlays are
  painted on top. Anything cropped and handed to a model is the true frame.
* **Coordinates round-trip.** :meth:`widget_to_image` and
  :meth:`image_to_widget` are exact inverses at any zoom or pan, so a click
  resolves to a real image pixel — the thing that makes "point at it and run
  the model on that patch" possible at all.
* **Layers stack.** Anything can draw over the frame, in image space or in
  widget space, toggled independently, ordered by ``z``.
* **Tools are pluggable.** A gesture arrives as a ``ToolResult`` on one signal.
  A new interaction is a new ``Tool`` subclass and no canvas edits.
* **Annotations are live items.** Selection, hover, dragging and resizing come
  from Qt's graphics framework rather than hand-rolled hit-testing.
* **Resizing is free.** The last frame is retained, so a window resize is a
  repaint. The old widget re-seeked and re-decoded a frame to redraw.

Sources of pixels are irrelevant to this class: hand it a
:class:`~label_kit.core.frames.Frame` from a video, a still image, or a
synthesised array, and everything else behaves identically.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPixmap,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from ..core.annotations import Annotation, AnnotationChange, AnnotationStore
from ..core.frames import Frame
from ..core.geometry import Point, Rect, Size, ViewTransform
from ..core.logging import get_logger
from ..core.registry import Registry
from .items import AnnotationItem, CanvasStyle, make_annotation_item
from .layers import BusyLayer, CrosshairLayer, FrameInfoLayer, Layer, LayerContext, OverlayLayer
from .palette import LabelPalette
from .theme import Theme, ThemedMixin, current_theme
from .tools import SelectTool, Tool, ToolEvent, ToolResult, builtin_tools

__all__ = ["ImageCanvas"]

_log = get_logger(__name__)

_MIN_ZOOM = 0.02
_MAX_ZOOM = 60.0
_WHEEL_STEP = 1.15
_FRAME_Z = -1000.0


class ImageCanvas(QGraphicsView, ThemedMixin):
    """A pannable, zoomable, layerable, annotatable image view."""

    # ── outward signals ─────────────────────────────────────────────────
    frame_displayed = Signal(object)        # Frame | None
    view_changed = Signal(object)           # ViewTransform
    cursor_moved = Signal(object)           # Point | None, in image coordinates
    tool_changed = Signal(str)              # tool id
    tool_result = Signal(object)            # ToolResult — the prompt channel
    selection_changed = Signal(object)      # tuple[str, ...] of annotation ids
    annotation_edited = Signal(object)      # Annotation, after a drag
    annotation_context_menu = Signal(str, object)  # annotation id, global QPoint
    canvas_context_menu = Signal(object, object)   # image Point, global QPoint
    delete_requested = Signal(object)       # tuple[str, ...] of annotation ids

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform,
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._frame_item = QGraphicsPixmapItem()
        self._frame_item.setZValue(_FRAME_Z)
        self._frame_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._frame_item)

        self._frame: Frame | None = None
        self._frame_buffer: np.ndarray | None = None
        self._image_size = Size(0, 0)
        self._error_text: str = ""
        self._cursor_image_pos: Point | None = None
        self._fit_pending = True
        self._auto_fit = True
        self._middle_pan_from: QPoint | None = None

        self.palette_map = LabelPalette()
        self._style = CanvasStyle(palette=self.palette_map, theme=current_theme())

        self.layers: Registry[Layer] = Registry("layer", owner_key=lambda layer: layer.owner)
        self.tools: Registry[Tool] = Registry("tool", owner_key=lambda tool: tool.owner)
        self._active_tool: Tool = SelectTool()

        self._store: AnnotationStore | None = None
        self._store_subscription = None
        self._items: dict[str, AnnotationItem] = {}

        self.frame_info_layer = FrameInfoLayer()
        self.crosshair_layer = CrosshairLayer()
        self.busy_layer = BusyLayer()
        for layer in (self.frame_info_layer, self.crosshair_layer, self.busy_layer):
            self.add_layer(layer)

        for tool in builtin_tools():
            self.register_tool(tool)
        self.set_tool(SelectTool.id)

        self._scene.selectionChanged.connect(self._on_scene_selection_changed)
        self.init_theme()

    # ── theming ─────────────────────────────────────────────────────────

    def apply_theme(self, theme: Theme) -> None:
        self._style = CanvasStyle(
            palette=self.palette_map,
            theme=theme,
            show_labels=self._style.show_labels,
            show_scores=self._style.show_scores,
            fill_shapes=self._style.fill_shapes,
        )
        self.setBackgroundBrush(QColor(theme.canvas_bg))
        for item in self._items.values():
            item.set_style(self._style)
        self.viewport().update()

    @property
    def style(self) -> CanvasStyle:
        return self._style

    def set_style_option(self, **options: bool) -> None:
        """Toggle label / score / fill rendering, e.g. ``show_labels=False``."""
        for key, value in options.items():
            if not hasattr(self._style, key):
                raise KeyError(f"unknown style option {key!r}")
            setattr(self._style, key, bool(value))
        for item in self._items.values():
            item.set_style(self._style)
        self.viewport().update()

    # ── frame ───────────────────────────────────────────────────────────

    @property
    def frame(self) -> Frame | None:
        return self._frame

    @property
    def image_size(self) -> Size:
        return self._image_size

    @property
    def has_frame(self) -> bool:
        return self._frame is not None

    def set_frame(self, frame: Frame | None) -> None:
        """Display ``frame``. Passing ``None`` clears the canvas."""
        self._error_text = ""
        if frame is None:
            self.clear()
            return

        previous_size = self._image_size
        self._frame = frame
        # Hold a reference for as long as the pixmap derives from it.
        self._frame_buffer = np.ascontiguousarray(frame.image)
        self._frame_item.setPixmap(_to_pixmap(self._frame_buffer))
        self._image_size = frame.size

        if previous_size != self._image_size:
            self._scene.setSceneRect(QRectF(0, 0, self._image_size.width, self._image_size.height))
            self._fit_pending = True

        if self._fit_pending and self._auto_fit:
            self.fit_to_window()

        for layer in self.layers:
            layer.on_frame(frame)
        self._rebuild_items()
        self.viewport().update()
        self.frame_displayed.emit(frame)

    def clear(self) -> None:
        self._frame = None
        self._frame_buffer = None
        self._frame_item.setPixmap(QPixmap())
        self._image_size = Size(0, 0)
        self._clear_items()
        for layer in self.layers:
            layer.on_frame(None)
        self.viewport().update()
        self.frame_displayed.emit(None)

    def show_error(self, message: str) -> None:
        """Replace the frame with an error message."""
        self.clear()
        self._error_text = message
        self.viewport().update()

    # ── layers ──────────────────────────────────────────────────────────

    def add_layer(self, layer: Layer, *, replace: bool = False) -> Layer:
        self.layers.register(layer.id, layer, replace=replace)
        layer.changed.connect(self._on_layer_changed)
        layer.on_attach()
        if self._frame is not None:
            layer.on_frame(self._frame)
        self.viewport().update()
        return layer

    def remove_layer(self, layer_id: str) -> Layer | None:
        layer = self.layers.unregister(layer_id)
        if layer is not None:
            layer.on_detach()
            self.viewport().update()
        return layer

    def layer(self, layer_id: str) -> Layer | None:
        return self.layers.get(layer_id)

    def set_layer_visible(self, layer_id: str, visible: bool) -> None:
        layer = self.layers.get(layer_id)
        if layer is not None:
            layer.visible = visible

    def _on_layer_changed(self, _layer_id: str) -> None:
        self.viewport().update()

    # ── tools ───────────────────────────────────────────────────────────

    def register_tool(self, tool: Tool, *, replace: bool = False) -> Tool:
        self.tools.register(tool.id, tool, replace=replace)
        tool.produced.connect(self._on_tool_produced)
        tool.preview_changed.connect(self._on_layer_changed)
        return tool

    def unregister_tool(self, tool_id: str) -> Tool | None:
        if self._active_tool.id == tool_id:
            self.set_tool(SelectTool.id)
        return self.tools.unregister(tool_id)

    @property
    def active_tool(self) -> Tool:
        return self._active_tool

    def set_tool(self, tool: Tool | str) -> Tool:
        resolved = self.tools.require(tool) if isinstance(tool, str) else tool
        if resolved is self._active_tool:
            return resolved

        self._active_tool.deactivate()
        self._active_tool = resolved
        resolved.activate()

        if isinstance(resolved, SelectTool):
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        elif resolved.id == "pan":
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

        self.viewport().setCursor(resolved.cursor)
        self.viewport().update()
        self.tool_changed.emit(resolved.id)
        return resolved

    def _on_tool_produced(self, result: ToolResult) -> None:
        self.tool_result.emit(result)

    # ── annotation store ────────────────────────────────────────────────

    def bind_store(self, store: AnnotationStore | None) -> None:
        """Render ``store``'s annotations for whichever frame is displayed."""
        if self._store_subscription is not None:
            self._store_subscription.disconnect()
            self._store_subscription = None
        self._store = store
        if store is not None:
            self._store_subscription = store.changed.connect(self._on_store_changed)
        self._rebuild_items()

    @property
    def store(self) -> AnnotationStore | None:
        return self._store

    def selected_ids(self) -> tuple[str, ...]:
        return tuple(
            item.annotation_id
            for item in self._scene.selectedItems()
            if isinstance(item, AnnotationItem)
        )

    def select_annotations(self, ids: tuple[str, ...] | list[str]) -> None:
        wanted = set(ids)
        for annotation_id, item in self._items.items():
            item.setSelected(annotation_id in wanted)

    def clear_selection(self) -> None:
        self._scene.clearSelection()

    def _current_annotations(self) -> list[Annotation]:
        if self._store is None or self._frame is None:
            return []
        return self._store.for_frame(self._frame.ref)

    def _on_store_changed(self, change: AnnotationChange) -> None:
        if self._frame is None:
            return
        # Only rebuild when the change touches the frame on screen.
        if change.action != "reset" and self._frame.index not in {
            a.frame.index for a in change.annotations
        }:
            return
        self._rebuild_items()

    def _clear_items(self) -> None:
        for item in self._items.values():
            self._scene.removeItem(item)
        self._items.clear()

    def _rebuild_items(self) -> None:
        wanted = {a.id: a for a in self._current_annotations()}
        selected = set(self.selected_ids())

        for annotation_id in list(self._items):
            if annotation_id not in wanted:
                self._scene.removeItem(self._items.pop(annotation_id))

        scale = self.view_transform.scale
        for annotation_id, annotation in wanted.items():
            item = self._items.get(annotation_id)
            if item is None:
                item = make_annotation_item(annotation, self._style)
                item.geometry_committed.connect(self._on_item_geometry_committed)
                item.context_menu_requested.connect(self._on_item_context_menu)
                item.set_view_scale(scale)
                self._scene.addItem(item)
                self._items[annotation_id] = item
                if annotation_id in selected:
                    item.setSelected(True)
            elif item.annotation is not annotation:
                item.set_annotation(annotation)

    def _on_item_geometry_committed(self, annotation_id: str, geometry: object) -> None:
        if self._store is None:
            return
        existing = self._store.by_id(annotation_id)
        if existing is not None:
            self.annotation_edited.emit(existing.with_geometry(geometry))

    def _on_item_context_menu(self, annotation_id: str, scene_pos: QPointF) -> None:
        global_pos = self.viewport().mapToGlobal(self.mapFromScene(scene_pos))
        self.annotation_context_menu.emit(annotation_id, global_pos)

    def _on_scene_selection_changed(self) -> None:
        self.selection_changed.emit(self.selected_ids())

    # ── coordinates ─────────────────────────────────────────────────────

    @property
    def view_transform(self) -> ViewTransform:
        """Scene→viewport mapping, as a plain data object layers and tools can use."""
        t = self.viewportTransform()
        scale = t.m11() if t.m11() > 0 else 1.0
        return ViewTransform(scale=scale, offset_x=t.dx(), offset_y=t.dy())

    def widget_to_image(self, pos: QPoint | QPointF | Point) -> Point:
        """Exact inverse of :meth:`image_to_widget`, sub-pixel accurate.

        Qt's own ``mapToScene``/``mapFromScene`` only accept and return integer
        ``QPoint``, which silently quantises to whole *widget* pixels — at 0.25×
        zoom that is a four-image-pixel error, enough to put a model prompt on
        the wrong object. Applying the viewport transform directly keeps full
        float precision.
        """
        inverted, ok = self.viewportTransform().inverted()
        if not ok:
            return Point(0.0, 0.0)
        scene_pos = inverted.map(_to_qpointf(pos))
        return Point(scene_pos.x(), scene_pos.y())

    def image_to_widget(self, point: Point) -> Point:
        widget_pos = self.viewportTransform().map(QPointF(point.x, point.y))
        return Point(widget_pos.x(), widget_pos.y())

    def is_inside_image(self, point: Point) -> bool:
        return (
            0 <= point.x < self._image_size.width and 0 <= point.y < self._image_size.height
        )

    # ── zoom / pan ──────────────────────────────────────────────────────

    @property
    def zoom(self) -> float:
        return self.transform().m11()

    @property
    def auto_fit(self) -> bool:
        """Whether a newly loaded video is fitted to the window automatically."""
        return self._auto_fit

    @auto_fit.setter
    def auto_fit(self, value: bool) -> None:
        self._auto_fit = bool(value)

    def fit_to_window(self) -> None:
        if self._image_size.is_empty:
            return
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._fit_pending = False
        self._after_view_change()

    def zoom_to(self, factor: float) -> None:
        target = min(max(factor, _MIN_ZOOM), _MAX_ZOOM)
        self.setTransform(QTransform.fromScale(target, target))
        self._fit_pending = False
        self._after_view_change()

    def zoom_by(self, factor: float) -> None:
        target = min(max(self.zoom * factor, _MIN_ZOOM), _MAX_ZOOM)
        if abs(target - self.zoom) < 1e-9:
            return
        self.scale(target / self.zoom, target / self.zoom)
        self._fit_pending = False
        self._after_view_change()

    def zoom_in(self) -> None:
        self.zoom_by(_WHEEL_STEP)

    def zoom_out(self) -> None:
        self.zoom_by(1 / _WHEEL_STEP)

    def zoom_reset(self) -> None:
        """Back to 1 image pixel = 1 screen pixel."""
        self.zoom_to(1.0)

    def center_on_image_point(self, point: Point) -> None:
        self.centerOn(QPointF(point.x, point.y))
        self._after_view_change()

    def _after_view_change(self) -> None:
        transform = self.view_transform
        for item in self._items.values():
            item.set_view_scale(transform.scale)
        self.view_changed.emit(transform)
        self.viewport().update()

    # ── painting ────────────────────────────────────────────────────────

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        """Paint layers and the active tool's preview above the items."""
        ctx = LayerContext(
            frame=self._frame,
            transform=self.view_transform,
            theme=self._style.theme,
            viewport=QRectF(self.viewport().rect()),
            image_size=self._image_size,
            cursor=self._cursor_image_pos,
        )

        for layer in sorted(self.layers, key=lambda item: item.z):
            if not layer.visible:
                continue
            painter.save()
            try:
                if isinstance(layer, OverlayLayer):
                    painter.resetTransform()
                layer.paint(painter, ctx)
            except Exception:  # noqa: BLE001 - a paint that raises every frame would
                # spin forever; disable the layer and keep the canvas usable.
                _log.exception("Layer %r raised while painting; hiding it", layer.id)
                layer.visible = False
            finally:
                painter.restore()

        painter.save()
        try:
            self._active_tool.paint_preview(painter, ctx)
        except Exception:  # noqa: BLE001 - a plugin tool must not break rendering
            _log.exception("Tool %r raised while painting its preview", self._active_tool.id)
        finally:
            painter.restore()

        if self._error_text:
            painter.save()
            painter.resetTransform()
            painter.setPen(QColor(self._style.theme.error_fg))
            font = painter.font()
            font.setPixelSize(16)
            painter.setFont(font)
            painter.drawText(
                QRectF(self.viewport().rect()),
                int(Qt.AlignmentFlag.AlignCenter),
                self._error_text,
            )
            painter.restore()

    # ── events ──────────────────────────────────────────────────────────

    def _tool_event(self, event) -> ToolEvent:
        widget_pos = event.position() if hasattr(event, "position") else QPointF(event.pos())
        return ToolEvent(
            image_pos=self.widget_to_image(widget_pos),
            widget_pos=Point(widget_pos.x(), widget_pos.y()),
            button=event.button() if hasattr(event, "button") else Qt.MouseButton.NoButton,
            buttons=event.buttons(),
            modifiers=event.modifiers(),
            view_scale=self.view_transform.scale,
        )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._middle_pan_from = event.position().toPoint()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        tool_event = self._tool_event(event)
        if self._active_tool.mouse_press(tool_event):
            event.accept()
            return
        if self._active_tool.items_interactive:
            super().mousePressEvent(event)
            if not event.isAccepted() and event.button() == Qt.MouseButton.RightButton:
                self._emit_canvas_context_menu(tool_event, event)
            return
        if event.button() == Qt.MouseButton.RightButton:
            self._emit_canvas_context_menu(tool_event, event)
            return
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._middle_pan_from is not None:
            self._pan_by_screen_delta(event.position().toPoint())
            event.accept()
            return

        tool_event = self._tool_event(event)
        self._update_cursor_readout(tool_event.image_pos)
        if self._active_tool.mouse_move(tool_event):
            event.accept()
            return
        if self._active_tool.items_interactive or self.dragMode() != QGraphicsView.DragMode.NoDrag:
            super().mouseMoveEvent(event)
            return
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._middle_pan_from is not None:
            self._middle_pan_from = None
            self.viewport().setCursor(self._active_tool.cursor)
            event.accept()
            return

        if self._active_tool.mouse_release(self._tool_event(event)):
            event.accept()
            return
        if self._active_tool.items_interactive or self.dragMode() != QGraphicsView.DragMode.NoDrag:
            super().mouseReleaseEvent(event)
            return
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        if self._active_tool.mouse_double_click(self._tool_event(event)):
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def leaveEvent(self, event) -> None:
        self._update_cursor_readout(None)
        super().leaveEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        self.zoom_by(_WHEEL_STEP if delta > 0 else 1 / _WHEEL_STEP)
        event.accept()

    def keyPressEvent(self, event) -> None:
        if self._active_tool.key_press(event.key(), event.modifiers()):
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            selected = self.selected_ids()
            if selected:
                self.delete_requested.emit(selected)
                event.accept()
                return
        super().keyPressEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # A retained frame means resize is a repaint, never a re-decode.
        if self._fit_pending and self._auto_fit:
            self.fit_to_window()
        else:
            self._after_view_change()

    def _pan_by_screen_delta(self, position: QPoint) -> None:
        if self._middle_pan_from is None:
            return
        delta = position - self._middle_pan_from
        self._middle_pan_from = position
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        self._after_view_change()

    def _update_cursor_readout(self, point: Point | None) -> None:
        if point == self._cursor_image_pos:
            return
        self._cursor_image_pos = point
        self.cursor_moved.emit(point)
        if self.crosshair_layer.visible:
            self.viewport().update()

    def _emit_canvas_context_menu(self, tool_event: ToolEvent, event) -> None:
        global_pos = self.viewport().mapToGlobal(event.position().toPoint())
        self.canvas_context_menu.emit(tool_event.image_pos, global_pos)
        event.accept()

    # ── export ──────────────────────────────────────────────────────────

    def snapshot(self, include_overlays: bool = True) -> QImage:
        """Render what is on screen to an image — for reports and bug attachments."""
        if self._frame is None:
            return QImage()
        if not include_overlays:
            return _to_qimage(self._frame_buffer).copy()

        width, height = self._image_size.as_int()
        image = QImage(width, height, QImage.Format.Format_RGB888)
        image.fill(QColor(self._style.theme.canvas_bg))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        try:
            self._scene.render(
                painter,
                QRectF(0, 0, width, height),
                self._scene.sceneRect(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
            )
        finally:
            painter.end()
        return image

    def visible_image_rect(self) -> Rect:
        """The portion of the image currently on screen, in image coordinates."""
        scene_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        return Rect(
            scene_rect.left(), scene_rect.top(), scene_rect.right(), scene_rect.bottom(),
        ).clamped_to(self._image_size)


# ── numpy → Qt ──────────────────────────────────────────────────────────


def _to_qimage(array: np.ndarray) -> QImage:
    """Wrap a contiguous RGB uint8 array as a QImage.

    The QImage does **not** own the buffer, so the caller must keep ``array``
    alive for as long as the image is used, or call ``.copy()``.
    """
    height, width, channels = array.shape
    if channels != 3:
        raise ValueError(f"expected an RGB array, got {channels} channels")
    return QImage(array.data, width, height, array.strides[0], QImage.Format.Format_RGB888)


def _to_pixmap(array: np.ndarray) -> QPixmap:
    """Convert an RGB array to a pixmap. ``fromImage`` deep-copies the pixels."""
    return QPixmap.fromImage(_to_qimage(array))


def _to_qpointf(pos: QPoint | QPointF | Point) -> QPointF:
    if isinstance(pos, Point):
        return QPointF(pos.x, pos.y)
    if isinstance(pos, QPoint):
        return QPointF(pos)
    return pos
