"""Media sources: the thing a frame comes out of.

Only ever touched from the decode thread.  Nothing here is Qt-aware, so a source
can be driven directly in a test or a batch script with no event loop.

Frame indexing rule: ``read()`` returns the frame at the source's current
position and then advances.  OpenCV's ``CAP_PROP_POS_FRAMES`` reports the
position of the *next* frame, not the one just read, which is the off-by-one
that the previous player papered over with ``max(0, current - 1)`` seeks
scattered through resize and step handling.  Here the position is tracked
explicitly instead.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from ..core.errors import SourceOpenError
from ..core.frames import Frame, FrameRef, MediaInfo, source_id_for
from ..core.geometry import Size
from ..core.logging import get_logger

__all__ = [
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "ImageFileSource",
    "MediaSource",
    "VideoFileSource",
    "open_source",
]

_log = get_logger(__name__)

VIDEO_EXTENSIONS = frozenset({
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
    ".m4v", ".mpg", ".mpeg", ".3gp", ".ts", ".mts", ".m2ts",
})

IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
})

_DEFAULT_FPS = 30.0
# Seeking backwards a few frames is slower than decoding forwards to the target,
# because a seek forces the demuxer back to the previous keyframe.
_MAX_FORWARD_SCAN = 12


class MediaSource(ABC):
    """A random-access sequence of frames."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        self.source_id = source_id_for(self.path)

    @property
    @abstractmethod
    def info(self) -> MediaInfo: ...

    @property
    @abstractmethod
    def position(self) -> int:
        """Index of the frame the next :meth:`read` will return."""

    @abstractmethod
    def read(self) -> Frame | None:
        """Decode at the current position and advance. ``None`` at end of stream."""

    @abstractmethod
    def seek(self, index: int) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    def read_at(self, index: int) -> Frame | None:
        self.seek(index)
        return self.read()

    def __enter__(self) -> MediaSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class VideoFileSource(MediaSource):
    """OpenCV-backed video reader.

    Container metadata is treated as a hint, not a fact: ``FRAME_COUNT`` is
    routinely wrong or zero for streamed and variable-frame-rate files, and
    ``FPS`` can be zero or NaN.  Both are sanitised on open, and the true frame
    count is corrected downward the first time a read fails short.
    """

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        if not self.path.exists():
            raise SourceOpenError(f"File does not exist: {self.path}")

        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            cap.release()
            raise SourceOpenError(f"Cannot open video: {self.path.name}")

        self._cap = cap
        self._position = 0
        self._closed = False

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or math.isnan(fps) or fps <= 0 or fps > 1000:
            _log.warning("%s: implausible fps %r, assuming %g", self.path.name, fps, _DEFAULT_FPS)
            fps = _DEFAULT_FPS

        raw_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_count = int(raw_count) if raw_count and raw_count > 0 else 0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            # Some codecs only report dimensions once a frame has been decoded.
            ok, probe = cap.read()
            if ok and probe is not None:
                height, width = probe.shape[:2]
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                cap.release()
                self._closed = True
                raise SourceOpenError(f"Cannot determine dimensions: {self.path.name}")

        self._info = MediaInfo(
            source_id=self.source_id,
            path=self.path,
            size=Size(float(width), float(height)),
            frame_count=frame_count,
            fps=float(fps),
        )
        _log.debug(
            "Opened %s: %dx%d @ %.3ffps, %d frames",
            self.path.name, width, height, fps, frame_count,
        )

    @property
    def info(self) -> MediaInfo:
        return self._info

    @property
    def position(self) -> int:
        return self._position

    def read(self) -> Frame | None:
        if self._closed:
            return None
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            self._note_true_end()
            return None

        index = self._position
        self._position += 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Frame(
            ref=FrameRef(self.source_id, index),
            image=rgb,
            timestamp=self._info.time_of(index),
        )

    def seek(self, index: int) -> None:
        if self._closed:
            return
        target = max(0, index)
        if self._info.frame_count > 0:
            target = min(target, self._info.frame_count - 1)
        if target == self._position:
            return

        # Short forward hops are cheaper as sequential grabs than as a real seek,
        # which would rewind the demuxer to the preceding keyframe.
        delta = target - self._position
        if 0 < delta <= _MAX_FORWARD_SCAN:
            for _ in range(delta):
                if not self._cap.grab():
                    self._note_true_end()
                    return
                self._position += 1
            return

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(target))
        self._position = target

    def _note_true_end(self) -> None:
        """Learn the real frame count from where decoding actually ran out.

        Covers both an over-reporting container and one that reports nothing at
        all (``frame_count == 0``), which is common for streamed and VFR files.
        """
        if self._position > 0 and self._info.frame_count != self._position:
            _log.debug(
                "%s: frame count %d -> %d (corrected at end of stream)",
                self.path.name, self._info.frame_count, self._position,
            )
            self._info = replace(self._info, frame_count=self._position)

    def close(self) -> None:
        if not self._closed:
            self._cap.release()
            self._closed = True


class ImageFileSource(MediaSource):
    """A still image presented as a one-frame source.

    Lets the canvas, layers, tools and every inference model work on image
    datasets with no special-casing anywhere downstream.
    """

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        bgr = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SourceOpenError(f"Cannot open image: {self.path.name}")

        self._image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = self._image.shape[:2]
        self._info = MediaInfo(
            source_id=self.source_id,
            path=self.path,
            size=Size(float(width), float(height)),
            frame_count=1,
            fps=0.0,
        )
        self._position = 0
        self._closed = False

    @property
    def info(self) -> MediaInfo:
        return self._info

    @property
    def position(self) -> int:
        return self._position

    def read(self) -> Frame | None:
        # Closed is end-of-stream, exactly as it is for a video. Without the
        # check, a seek after close rewinds to a pixel buffer that close() has
        # already emptied, and the caller gets a 0x0 frame instead of None.
        if self._closed or self._position != 0:
            return None
        self._position = 1
        return Frame(ref=FrameRef(self.source_id, 0), image=self._image.copy())

    def seek(self, index: int) -> None:
        if self._closed:
            return
        self._position = 0 if index <= 0 else 1

    def close(self) -> None:
        self._closed = True
        self._image = np.zeros((0, 0, 3), dtype=np.uint8)


def open_source(path: Path) -> MediaSource:
    """Pick a source implementation from the file extension."""
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return ImageFileSource(path)
    if suffix in VIDEO_EXTENSIONS:
        return VideoFileSource(path)
    # Unknown extension: let OpenCV try the video path and report honestly.
    return VideoFileSource(path)
