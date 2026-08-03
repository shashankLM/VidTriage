"""Media I/O: sources, the decode thread, and playback policy."""

from __future__ import annotations

from .clock import MAX_SPEED, MIN_SPEED, EndMode, PlaybackClock
from .controller import PlaybackController
from .decoder import DecodeWorker
from .source import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    ImageFileSource,
    MediaSource,
    VideoFileSource,
    open_source,
)

__all__ = [
    "IMAGE_EXTENSIONS",
    "MAX_SPEED",
    "MIN_SPEED",
    "VIDEO_EXTENSIONS",
    "DecodeWorker",
    "EndMode",
    "ImageFileSource",
    "MediaSource",
    "PlaybackClock",
    "PlaybackController",
    "VideoFileSource",
    "open_source",
]
