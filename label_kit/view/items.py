"""Graphics items that render and edit annotations on the canvas.

One item per annotation, living in scene (= image pixel) coordinates. Qt then
handles selection, hover and z-ordering for free, and the geometry an item
reports back after a drag is already in image space — no manual unprojection.

**Zoom-invariant decoration.** Outlines use cosmetic pens; grab handles and
label chips are drawn through a ``1/scale`` painter transform so they occupy a
fixed number of screen pixels at any zoom. Everything is still emitted inside
:meth:`AnnotationItem.boundingRect`, which is what keeps Qt's damage tracking
correct — painting outside it leaves stale pixels on screen when the view
scrolls. The canvas pushes the current scale into each item rather than having
items reach into ``painter`` mid-paint, so the bounding rect and the drawing
always agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject

from ..core.annotations import Annotation, AnnotationKind
from ..core.geometry import Geometry, Mask, Point, Polygon, Rect
from .palette import LabelPalette
from .theme import Theme, current_theme

__all__ = ["NO_HANDLE", "AnnotationItem", "CanvasStyle", "Handle", "make_annotation_item"]

_HANDLE_PX = 8.0
_GRAB_SLOP_PX = 4.0
_OUTLINE_PX = 2.0
_POINT_RADIUS_PX = 6.0
_LABEL_PAD_PX = 4.0
_LABEL_FONT_PX = 11
_MASK_ALPHA = 110
_MIN_BOX_PX = 2.0
_MAX_VERTEX_HANDLES = 64

#: Sentinel meaning "the cursor is not over any grab handle".
NO_HANDLE = -1


@dataclass
class CanvasStyle:
    """Everything an item needs to draw itself, owned by the canvas."""

    palette: LabelPalette
    theme: Theme
    show_labels: bool = True
    show_scores: bool = True
    fill_shapes: bool = True

    def color_for(self, annotation: Annotation, alpha: int = 255) -> QColor:
        return self.palette.color(annotation.label, alpha)


class Handle(IntEnum):
    """Named grips on a rectangle, ordered clockwise from the top-left.

    Handles are addressed as plain ``int`` throughout, because polygons use the
    same channel to report a *vertex index* — which has no upper bound and so
    cannot be an enum member.
    """

    TOP_LEFT = 0
    TOP = 1
    TOP_RIGHT = 2
    RIGHT = 3
    BOTTOM_RIGHT = 4
    BOTTOM = 5
    BOTTOM_LEFT = 6
    LEFT = 7


_BOX_HANDLE_CURSORS: dict[int, Qt.CursorShape] = {
    Handle.TOP_LEFT: Qt.CursorShape.SizeFDiagCursor,
    Handle.TOP: Qt.CursorShape.SizeVerCursor,
    Handle.TOP_RIGHT: Qt.CursorShape.SizeBDiagCursor,
    Handle.RIGHT: Qt.CursorShape.SizeHorCursor,
    Handle.BOTTOM_RIGHT: Qt.CursorShape.SizeFDiagCursor,
    Handle.BOTTOM: Qt.CursorShape.SizeVerCursor,
    Handle.BOTTOM_LEFT: Qt.CursorShape.SizeBDiagCursor,
    Handle.LEFT: Qt.CursorShape.SizeHorCursor,
}


class AnnotationItem(QGraphicsObject):
    """Base for every on-canvas annotation.

    Subclasses implement :meth:`local_bounds`, :meth:`paint_shape` and
    :meth:`geometry_after_move`. Labels, selection styling, zoom-invariant
    sizing and drag bookkeeping all live here.
    """

    geometry_committed = Signal(str, object)      # annotation id, Geometry
    context_menu_requested = Signal(str, object)  # annotation id, scene QPointF

    #: Whether the user can drag this kind of shape around.
    editable: bool = True
    #: Cursor shown over a grab handle, keyed by handle id.
    handle_cursors: dict[int, Qt.CursorShape] = {}

    def __init__(self, annotation: Annotation, style: CanvasStyle) -> None:
        super().__init__()
        self._annotation = annotation
        self._style = style
        self._view_scale = 1.0
        self._hovered = False
        self._drag_origin: QPointF | None = None
        self._drag_start_geometry: Geometry | None = None
        self._active_handle: int = NO_HANDLE

        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsFocusable, True)
        # Masks sit under outlines so a box drawn over a mask stays readable.
        self.setZValue(10.0 if annotation.kind is AnnotationKind.MASK else 20.0)

    # ── state ───────────────────────────────────────────────────────────

    @property
    def annotation(self) -> Annotation:
        return self._annotation

    @property
    def annotation_id(self) -> str:
        return self._annotation.id

    def set_annotation(self, annotation: Annotation) -> None:
        self.prepareGeometryChange()
        self._annotation = annotation
        self.update()

    def set_style(self, style: CanvasStyle) -> None:
        self.prepareGeometryChange()
        self._style = style
        self.update()

    def set_view_scale(self, scale: float) -> None:
        """Told by the canvas whenever the zoom changes."""
        if scale <= 0 or abs(scale - self._view_scale) < 1e-9:
            return
        self.prepareGeometryChange()
        self._view_scale = scale
        self.update()

    def _px(self, screen_pixels: float) -> float:
        """Screen pixels expressed in scene (image) units at the current zoom."""
        return screen_pixels / self._view_scale

    # ── geometry hooks ──────────────────────────────────────────────────

    def local_bounds(self) -> QRectF:
        """The shape's own extent in scene units, before decoration."""
        r = self._annotation.bounding_rect
        return QRectF(r.x1, r.y1, r.width, r.height)

    def geometry_after_move(self, dx: float, dy: float) -> Geometry:
        raise NotImplementedError

    def geometry_after_resize(self, handle: int, dx: float, dy: float) -> Geometry:
        """Only rectangles and polygons resize; others fall back to a move."""
        return self.geometry_after_move(dx, dy)

    def paint_shape(self, painter: QPainter, color: QColor, selected: bool) -> None:
        raise NotImplementedError

    def handle_positions(self) -> dict[int, QPointF]:
        return {}

    # ── QGraphicsItem ───────────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        margin = self._px(_HANDLE_PX / 2 + _OUTLINE_PX + _GRAB_SLOP_PX)
        rect = self.local_bounds().adjusted(-margin, -margin, margin, margin)

        label = self._label_text()
        if self._style.show_labels and label:
            width, height = self._label_size_px(label)
            # The chip hangs above the shape's top-left corner.
            rect = rect.united(
                QRectF(
                    self.local_bounds().left(),
                    self.local_bounds().top() - self._px(height),
                    self._px(width),
                    self._px(height),
                ),
            )
        return rect

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        slop = self._px(_GRAB_SLOP_PX + _OUTLINE_PX)
        path.addRect(self.local_bounds().adjusted(-slop, -slop, slop, slop))
        return path

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        selected = self.isSelected()
        color = self._style.color_for(self._annotation)

        self.paint_shape(painter, color, selected)

        if selected and self.editable:
            self._paint_handles(painter)
        if self._style.show_labels:
            self._paint_label(painter, color)

    # ── styling helpers ─────────────────────────────────────────────────

    def _outline_pen(self, color: QColor, selected: bool) -> QPen:
        """Cosmetic pen — width is in device pixels, so it survives any zoom."""
        pen = QPen(self._style.theme.color("selection_fg") if selected else color)
        pen.setCosmetic(True)
        pen.setWidthF(_OUTLINE_PX + (1.0 if selected or self._hovered else 0.0))
        if self._annotation.is_prediction and not selected:
            pen.setStyle(Qt.PenStyle.DashLine)
        return pen

    def _label_text(self) -> str:
        parts = []
        if self._annotation.label:
            parts.append(self._annotation.label)
        if self._style.show_scores and self._annotation.score is not None:
            parts.append(f"{self._annotation.score:.2f}")
        return " ".join(parts)

    @staticmethod
    def _label_font() -> QFont:
        font = QFont()
        font.setPixelSize(_LABEL_FONT_PX)
        return font

    @classmethod
    def _label_size_px(cls, text: str) -> tuple[float, float]:
        metrics = QFontMetricsF(cls._label_font())
        return (
            metrics.horizontalAdvance(text) + 2 * _LABEL_PAD_PX,
            metrics.height() + _LABEL_PAD_PX,
        )

    def _paint_label(self, painter: QPainter, color: QColor) -> None:
        text = self._label_text()
        if not text:
            return
        width, height = self._label_size_px(text)
        anchor = self.local_bounds().topLeft()

        painter.save()
        # Scale so one painter unit equals one screen pixel: the chip and its
        # text keep a constant size while staying anchored in scene space.
        painter.translate(anchor)
        painter.scale(1.0 / self._view_scale, 1.0 / self._view_scale)

        chip = QRectF(0, -height, width, height)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawRect(chip)

        painter.setFont(self._label_font())
        painter.setPen(QPen(self._contrasting(color)))
        painter.drawText(
            chip.adjusted(_LABEL_PAD_PX, 0, 0, 0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            text,
        )
        painter.restore()

    @staticmethod
    def _contrasting(color: QColor) -> QColor:
        luminance = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
        return QColor("#101010") if luminance > 140 else QColor("#f5f5f5")

    # ── handles ─────────────────────────────────────────────────────────

    def _paint_handles(self, painter: QPainter) -> None:
        positions = self.handle_positions()
        if not positions:
            return
        theme = self._style.theme
        pen = QPen(theme.color("handle_fg"))
        pen.setCosmetic(True)
        pen.setWidthF(1.0)
        brush = QBrush(theme.color("handle_bg"))
        half = _HANDLE_PX / 2

        for pos in positions.values():
            painter.save()
            painter.translate(pos)
            painter.scale(1.0 / self._view_scale, 1.0 / self._view_scale)
            painter.setPen(pen)
            painter.setBrush(brush)
            painter.drawRect(QRectF(-half, -half, _HANDLE_PX, _HANDLE_PX))
            painter.restore()

    def handle_at(self, pos: QPointF) -> int:
        reach = self._px(_HANDLE_PX / 2 + _GRAB_SLOP_PX)
        for handle, centre in self.handle_positions().items():
            if abs(centre.x() - pos.x()) <= reach and abs(centre.y() - pos.y()) <= reach:
                return handle
        return NO_HANDLE

    def cursor_for_handle(self, handle: int) -> Qt.CursorShape:
        if handle == NO_HANDLE:
            return Qt.CursorShape.SizeAllCursor if self.editable else Qt.CursorShape.ArrowCursor
        return self.handle_cursors.get(handle, Qt.CursorShape.CrossCursor)

    # ── interaction ─────────────────────────────────────────────────────

    def hoverEnterEvent(self, event) -> None:
        self._hovered = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self._hovered = False
        self.unsetCursor()
        self.update()
        super().hoverLeaveEvent(event)

    def hoverMoveEvent(self, event) -> None:
        if self.isSelected() and self.editable:
            self.setCursor(self.cursor_for_handle(self.handle_at(event.pos())))
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            self.context_menu_requested.emit(self.annotation_id, event.scenePos())
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton or not self.editable:
            super().mousePressEvent(event)
            return

        self.setSelected(True)
        self._drag_origin = event.scenePos()
        self._drag_start_geometry = self._annotation.geometry
        self._active_handle = self.handle_at(event.pos())
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_origin is None or self._drag_start_geometry is None:
            super().mouseMoveEvent(event)
            return
        delta = event.scenePos() - self._drag_origin
        if self._active_handle == NO_HANDLE:
            geometry = self.geometry_after_move(delta.x(), delta.y())
        else:
            geometry = self.geometry_after_resize(self._active_handle, delta.x(), delta.y())
        self.set_annotation(self._annotation.with_geometry(geometry))
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_origin is None:
            super().mouseReleaseEvent(event)
            return
        changed = self._annotation.geometry is not self._drag_start_geometry
        self._drag_origin = None
        self._drag_start_geometry = None
        self._active_handle = NO_HANDLE
        if changed:
            self.geometry_committed.emit(self.annotation_id, self._annotation.geometry)
        event.accept()

    def _start_geometry(self) -> Geometry:
        return self._drag_start_geometry if self._drag_start_geometry is not None else self._annotation.geometry


# ── concrete items ──────────────────────────────────────────────────────


class BoxItem(AnnotationItem):
    """A resizable, movable rectangle."""

    handle_cursors = _BOX_HANDLE_CURSORS

    def _start_rect(self) -> Rect:
        geometry = self._start_geometry()
        return geometry if isinstance(geometry, Rect) else self._annotation.bounding_rect

    def paint_shape(self, painter: QPainter, color: QColor, selected: bool) -> None:
        painter.setPen(self._outline_pen(color, selected))
        if self._style.fill_shapes:
            fill = QColor(color)
            fill.setAlpha(64 if selected else 40)
            painter.setBrush(QBrush(fill))
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(self.local_bounds())

    def handle_positions(self) -> dict[int, QPointF]:
        r = self.local_bounds()
        cx, cy = r.center().x(), r.center().y()
        return {
            Handle.TOP_LEFT: QPointF(r.left(), r.top()),
            Handle.TOP: QPointF(cx, r.top()),
            Handle.TOP_RIGHT: QPointF(r.right(), r.top()),
            Handle.RIGHT: QPointF(r.right(), cy),
            Handle.BOTTOM_RIGHT: QPointF(r.right(), r.bottom()),
            Handle.BOTTOM: QPointF(cx, r.bottom()),
            Handle.BOTTOM_LEFT: QPointF(r.left(), r.bottom()),
            Handle.LEFT: QPointF(r.left(), cy),
        }

    def geometry_after_move(self, dx: float, dy: float) -> Rect:
        return self._start_rect().translated(dx, dy)

    def geometry_after_resize(self, handle: int, dx: float, dy: float) -> Rect:
        r = self._start_rect()
        x1, y1, x2, y2 = r.x1, r.y1, r.x2, r.y2
        if handle in (Handle.TOP_LEFT, Handle.LEFT, Handle.BOTTOM_LEFT):
            x1 += dx
        if handle in (Handle.TOP_RIGHT, Handle.RIGHT, Handle.BOTTOM_RIGHT):
            x2 += dx
        if handle in (Handle.TOP_LEFT, Handle.TOP, Handle.TOP_RIGHT):
            y1 += dy
        if handle in (Handle.BOTTOM_LEFT, Handle.BOTTOM, Handle.BOTTOM_RIGHT):
            y2 += dy

        # Rect normalises inverted corners, so dragging a handle past the far
        # edge flips the box instead of collapsing it. The minimum keeps a
        # flipped-through box from becoming too small to grab again.
        minimum = self._px(_MIN_BOX_PX)
        if abs(x2 - x1) < minimum:
            x2 = x1 + minimum
        if abs(y2 - y1) < minimum:
            y2 = y1 + minimum
        return Rect(x1, y1, x2, y2)


class PointItem(AnnotationItem):
    """A movable marker drawn at constant screen size."""

    def _start_point(self) -> Point:
        geometry = self._start_geometry()
        return geometry if isinstance(geometry, Point) else self._annotation.bounding_rect.center

    def _point(self) -> Point:
        geometry = self._annotation.geometry
        return geometry if isinstance(geometry, Point) else self._annotation.bounding_rect.center

    def local_bounds(self) -> QRectF:
        p = self._point()
        r = self._px(_POINT_RADIUS_PX * 1.8)
        return QRectF(p.x - r, p.y - r, 2 * r, 2 * r)

    def paint_shape(self, painter: QPainter, color: QColor, selected: bool) -> None:
        p = self._point()
        centre = QPointF(p.x, p.y)
        radius = self._px(_POINT_RADIUS_PX)

        pen = self._outline_pen(color, selected)
        pen.setStyle(Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        fill = QColor(color)
        fill.setAlpha(200)
        painter.setBrush(QBrush(fill))
        painter.drawEllipse(centre, radius, radius)

        # Cross-hair arms make the exact pixel unambiguous when zoomed in.
        painter.setBrush(Qt.BrushStyle.NoBrush)
        arm = radius * 1.8
        painter.drawLine(QPointF(centre.x() - arm, centre.y()), QPointF(centre.x() + arm, centre.y()))
        painter.drawLine(QPointF(centre.x(), centre.y() - arm), QPointF(centre.x(), centre.y() + arm))

    def geometry_after_move(self, dx: float, dy: float) -> Point:
        return self._start_point().translated(dx, dy)


class PolygonItem(AnnotationItem):
    """A polygon whose vertices can be dragged individually.

    A handle id here is a vertex index, not a :class:`Handle` member.
    """

    def _start_polygon(self) -> Polygon:
        geometry = self._start_geometry()
        return geometry if isinstance(geometry, Polygon) else Polygon(())

    def _polygon(self) -> Polygon:
        geometry = self._annotation.geometry
        return geometry if isinstance(geometry, Polygon) else Polygon(())

    def paint_shape(self, painter: QPainter, color: QColor, selected: bool) -> None:
        poly = QPolygonF([QPointF(p.x, p.y) for p in self._polygon().points])
        if poly.isEmpty():
            return
        painter.setPen(self._outline_pen(color, selected))
        if self._style.fill_shapes:
            fill = QColor(color)
            fill.setAlpha(75 if selected else 50)
            painter.setBrush(QBrush(fill))
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolygon(poly)

    def handle_positions(self) -> dict[int, QPointF]:
        # Cap the grip count: a 500-vertex outline traced from a mask would
        # otherwise render as an unusable wall of squares.
        points = self._polygon().points[:_MAX_VERTEX_HANDLES]
        return {i: QPointF(p.x, p.y) for i, p in enumerate(points)}

    def cursor_for_handle(self, handle: int) -> Qt.CursorShape:
        if handle == NO_HANDLE:
            return Qt.CursorShape.SizeAllCursor
        return Qt.CursorShape.CrossCursor

    def geometry_after_move(self, dx: float, dy: float) -> Polygon:
        return self._start_polygon().translated(dx, dy)

    def geometry_after_resize(self, handle: int, dx: float, dy: float) -> Polygon:
        start = self._start_polygon()
        if not 0 <= handle < len(start.points):
            return start
        points = list(start.points)
        points[handle] = points[handle].translated(dx, dy)
        return Polygon(tuple(points))


class MaskItem(AnnotationItem):
    """A segmentation mask, tinted and composited over the frame.

    Display-only: masks come from a model, and the way to change one is to
    re-prompt, not to push pixels around by hand.
    """

    editable = False

    def __init__(self, annotation: Annotation, style: CanvasStyle) -> None:
        super().__init__(annotation, style)
        self._pixmap: QPixmap | None = None
        self._pixmap_color: QColor | None = None

    def set_annotation(self, annotation: Annotation) -> None:
        self._pixmap = None
        super().set_annotation(annotation)

    def set_style(self, style: CanvasStyle) -> None:
        self._pixmap = None
        super().set_style(style)

    def _mask(self) -> Mask | None:
        geometry = self._annotation.geometry
        return geometry if isinstance(geometry, Mask) else None

    @staticmethod
    def _build_pixmap(mask: Mask, color: QColor) -> QPixmap | None:
        height, width = mask.data.shape
        if height == 0 or width == 0:
            return None

        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[..., 0] = color.red()
        rgba[..., 1] = color.green()
        rgba[..., 2] = color.blue()
        rgba[..., 3] = mask.data.astype(np.uint8) * _MASK_ALPHA

        # QImage wraps the buffer without owning it; copy() detaches before the
        # numpy array goes out of scope and the pixels become garbage.
        image = QImage(
            rgba.data, width, height, 4 * width, QImage.Format.Format_RGBA8888,
        ).copy()
        return QPixmap.fromImage(image)

    def paint_shape(self, painter: QPainter, color: QColor, selected: bool) -> None:
        mask = self._mask()
        if mask is None or mask.is_empty:
            return
        if self._pixmap is None or self._pixmap_color != color:
            self._pixmap = self._build_pixmap(mask, color)
            self._pixmap_color = QColor(color)
        if self._pixmap is None:
            return

        bounds = mask.bounds
        target = QRectF(bounds.x1, bounds.y1, bounds.width, bounds.height)
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))
        if selected:
            painter.setPen(self._outline_pen(color, True))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(target)

    def geometry_after_move(self, dx: float, dy: float) -> Geometry:
        return self._annotation.geometry


_ITEM_TYPES: dict[AnnotationKind, type[AnnotationItem]] = {
    AnnotationKind.BOX: BoxItem,
    AnnotationKind.POINT: PointItem,
    AnnotationKind.POLYGON: PolygonItem,
    AnnotationKind.MASK: MaskItem,
}


def make_annotation_item(annotation: Annotation, style: CanvasStyle | None = None) -> AnnotationItem:
    """Build the right item subclass for an annotation's geometry."""
    resolved = style or CanvasStyle(palette=LabelPalette(), theme=current_theme())
    return _ITEM_TYPES[annotation.kind](annotation, resolved)
