"""Qt-free core: geometry, frames, annotations, events, registries, commands.

Nothing in this package imports ``PySide6``.  That is a deliberate constraint,
enforced by ``tests/test_layering.py``: the data model must be constructible and
testable without a ``QApplication``.
"""

from __future__ import annotations

from .annotations import (
    MANUAL_SOURCE,
    Annotation,
    AnnotationChange,
    AnnotationKind,
    AnnotationStore,
)
from .commands import Command, CommandRegistry
from .errors import (
    DecodeError,
    FileOperationError,
    InferenceError,
    MediaError,
    PersistenceError,
    PluginError,
    PluginLoadError,
    PluginNotAvailableError,
    SourceOpenError,
    VidTriageError,
)
from .events import Event, Subscription
from .frames import Frame, FrameRef, MediaInfo, source_id_for
from .geometry import Mask, Point, Polygon, Rect, Size, ViewTransform, bounding_rect_of
from .logging import attach_file_log, configure_logging, get_logger
from .registry import Registry, RegistryChange

__all__ = [
    "MANUAL_SOURCE",
    "Annotation",
    "AnnotationChange",
    "AnnotationKind",
    "AnnotationStore",
    "Command",
    "CommandRegistry",
    "DecodeError",
    "Event",
    "FileOperationError",
    "Frame",
    "FrameRef",
    "InferenceError",
    "Mask",
    "MediaError",
    "MediaInfo",
    "PersistenceError",
    "PluginError",
    "PluginLoadError",
    "PluginNotAvailableError",
    "Point",
    "Polygon",
    "Rect",
    "Registry",
    "RegistryChange",
    "Size",
    "SourceOpenError",
    "Subscription",
    "VidTriageError",
    "ViewTransform",
    "attach_file_log",
    "bounding_rect_of",
    "configure_logging",
    "get_logger",
    "source_id_for",
]
