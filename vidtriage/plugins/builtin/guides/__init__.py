"""Ratio guide lines — horizon and framing references drawn over the video.

This is the non-destructive version of the common workflow of re-encoding a
whole directory of clips with ``cv2.line`` just to see where ``h/2`` falls. Here
the lines are an overlay: toggled instantly, adjustable while watching, correct
at any zoom, and never written into the pixels — so the frames handed to a model
or exported for training stay clean.

It also stands as the smallest complete example of the plugin API: one layer,
a few commands, and persisted settings, in about a hundred lines.
"""

from __future__ import annotations

from fractions import Fraction

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QInputDialog

from ....core.logging import get_logger
from ....view.layers import LayerContext, SceneLayer
from ...api import Plugin, PluginContext

__all__ = ["PLUGIN", "GuideLinesLayer", "GuidesPlugin"]

_log = get_logger(__name__)

#: Fractions of the frame height, as ``"n/d"`` strings.
DEFAULT_HORIZONTAL = ("1/4", "1/3", "1/2", "2/3", "3/4")
DEFAULT_VERTICAL = ("1/2",)

_LABEL_FONT_PX = 11


def _parse_ratios(text: str) -> tuple[list[str], list[str]]:
    """Parse ``"1/4, 1/3, 0.5"`` into normalised ratio strings, plus errors."""
    ratios: list[str] = []
    errors: list[str] = []
    for token in text.replace("\n", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = Fraction(token).limit_denominator(1000)
        except (ValueError, ZeroDivisionError):
            errors.append(f"{token!r} is not a number or fraction")
            continue
        if not 0 < value < 1:
            errors.append(f"{token!r} must be between 0 and 1 (exclusive)")
            continue
        ratios.append(str(value))
    return ratios, errors


class GuideLinesLayer(SceneLayer):
    """Lines at fixed fractions of the frame, labelled with ratio and pixel row.

    Drawn in scene coordinates so the lines track the image under pan and zoom,
    with a cosmetic pen so they stay one pixel wide however far you zoom in.
    """

    id = "guides.lines"
    title = "Ratio Guides"
    z = 50.0
    # The shell generates the View ▸ Overlays toggle from the layer registry;
    # declaring the shortcut here means this plugin registers no toggle command
    # of its own, and so cannot produce a duplicate menu entry.
    shortcut = "Ctrl+G"

    def __init__(
        self,
        horizontal: tuple[str, ...] = DEFAULT_HORIZONTAL,
        vertical: tuple[str, ...] = DEFAULT_VERTICAL,
    ) -> None:
        super().__init__()
        self.horizontal = list(horizontal)
        self.vertical = list(vertical)
        self.show_labels = True
        self.color = "#ff5252"
        self.visible = False

    def set_horizontal(self, ratios: list[str]) -> None:
        self.horizontal = list(ratios)
        self.invalidate()

    def set_vertical(self, ratios: list[str]) -> None:
        self.vertical = list(ratios)
        self.invalidate()

    def paint(self, painter: QPainter, ctx: LayerContext) -> None:
        if ctx.image_size.is_empty:
            return

        pen = QPen(QColor(self.color or ctx.theme.guide_fg))
        pen.setCosmetic(True)
        pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        width, height = ctx.image_size.width, ctx.image_size.height

        for ratio in self.horizontal:
            y = float(Fraction(ratio)) * height
            painter.drawLine(QPointF(0, y), QPointF(width, y))
            if self.show_labels:
                self._label(painter, ctx, QPointF(4, y), f"h·{ratio} = {int(y)}")

        for ratio in self.vertical:
            x = float(Fraction(ratio)) * width
            painter.drawLine(QPointF(x, 0), QPointF(x, height))
            if self.show_labels:
                self._label(painter, ctx, QPointF(x, 14), f"w·{ratio} = {int(x)}")

    def _label(self, painter: QPainter, ctx: LayerContext, anchor: QPointF, text: str) -> None:
        """Draw text at constant screen size, anchored in image space."""
        scale = max(ctx.transform.scale, 1e-9)
        painter.save()
        painter.translate(anchor)
        painter.scale(1.0 / scale, 1.0 / scale)
        font = QFont()
        font.setPixelSize(_LABEL_FONT_PX)
        painter.setFont(font)
        painter.setPen(QPen(QColor(self.color or ctx.theme.guide_fg)))
        painter.drawText(QPointF(2, -3), text)
        painter.restore()


class GuidesPlugin(Plugin):
    id = "guides"
    name = "Ratio Guides"
    description = "Horizontal and vertical framing guides at fractions of the frame"
    default_enabled = True

    def __init__(self) -> None:
        super().__init__()
        self._layer: GuideLinesLayer | None = None

    def activate(self, ctx: PluginContext) -> None:
        self._layer = GuideLinesLayer(
            horizontal=tuple(ctx.settings.get("horizontal", DEFAULT_HORIZONTAL)),
            vertical=tuple(ctx.settings.get("vertical", DEFAULT_VERTICAL)),
        )
        self._layer.visible = bool(ctx.settings.get("visible", False))
        self._layer.show_labels = bool(ctx.settings.get("show_labels", True))
        ctx.add_layer(self._layer)
        # Remember visibility however it was changed — menu, shortcut, or code.
        ctx.subscribe(self._layer.changed, lambda _id: self._persist_visible(ctx))

        ctx.add_command(
            id="guides.edit_horizontal", title="Set Horizontal Guides…",
            menu="View/Overlays", section="9", order=10,
            handler=lambda: self._edit(ctx, horizontal=True),
        )
        ctx.add_command(
            id="guides.edit_vertical", title="Set Vertical Guides…",
            menu="View/Overlays", section="9", order=20,
            handler=lambda: self._edit(ctx, horizontal=False),
        )
        ctx.add_command(
            id="guides.labels", title="Guide Labels", menu="View/Overlays", section="9",
            order=30, checkable=True,
            is_checked=lambda: bool(self._layer and self._layer.show_labels),
            handler=lambda checked: self._set_labels(ctx, checked),
        )

    def _persist_visible(self, ctx: PluginContext) -> None:
        if self._layer is not None:
            ctx.settings["visible"] = self._layer.visible

    def _set_labels(self, ctx: PluginContext, show: bool) -> None:
        if self._layer is None:
            return
        self._layer.show_labels = show
        self._layer.invalidate()
        ctx.settings["show_labels"] = show

    def _edit(self, ctx: PluginContext, horizontal: bool) -> None:
        if self._layer is None:
            return
        axis = "horizontal" if horizontal else "vertical"
        current = self._layer.horizontal if horizontal else self._layer.vertical

        text, accepted = QInputDialog.getText(
            ctx.app.window,
            f"VidTriage — {axis.title()} Guides",
            f"Fractions of the frame {'height' if horizontal else 'width'}, "
            "comma separated.\nFractions or decimals, e.g. 1/3, 0.5, 2/3:",
            text=", ".join(current),
        )
        if not accepted:
            return

        ratios, errors = _parse_ratios(text)
        if errors:
            ctx.app.status("; ".join(errors), 8000)
            return

        if horizontal:
            self._layer.set_horizontal(ratios)
        else:
            self._layer.set_vertical(ratios)
        ctx.settings[axis] = ratios
        self._layer.visible = bool(ratios)
        ctx.app.status(f"{len(ratios)} {axis} guide(s)", 3000)


PLUGIN = GuidesPlugin
