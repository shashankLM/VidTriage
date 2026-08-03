"""Qt view layer: the canvas, its layers, its tools, and theming."""

from __future__ import annotations

from .canvas import ImageCanvas
from .items import AnnotationItem, CanvasStyle, Handle, make_annotation_item
from .layers import (
    BusyLayer,
    CrosshairLayer,
    FrameInfoLayer,
    Layer,
    LayerContext,
    OverlayLayer,
    SceneLayer,
    draw_hud_text,
)
from .palette import LabelPalette, label_color
from .theme import THEMES, Theme, ThemedMixin, ThemeManager, current_theme, set_theme, theme_manager
from .tools import (
    BoxTool,
    PanTool,
    PointTool,
    PolygonTool,
    SelectTool,
    Tool,
    ToolEvent,
    ToolResult,
    builtin_tools,
)

__all__ = [
    "THEMES",
    "AnnotationItem",
    "BoxTool",
    "BusyLayer",
    "CanvasStyle",
    "CrosshairLayer",
    "FrameInfoLayer",
    "Handle",
    "ImageCanvas",
    "LabelPalette",
    "Layer",
    "LayerContext",
    "OverlayLayer",
    "PanTool",
    "PointTool",
    "PolygonTool",
    "SceneLayer",
    "SelectTool",
    "Theme",
    "ThemeManager",
    "ThemedMixin",
    "Tool",
    "ToolEvent",
    "ToolResult",
    "builtin_tools",
    "current_theme",
    "draw_hud_text",
    "label_color",
    "make_annotation_item",
    "set_theme",
    "theme_manager",
]
