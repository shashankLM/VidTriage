"""Frame and media-identity primitives.

**Colour convention:** a :class:`Frame` always holds **RGB** ``uint8``.  The
conversion from OpenCV's native BGR happens exactly once, inside the decoder
thread.  Every downstream consumer — Qt's ``QImage``, ultralytics, SAM,
exporters — then gets what it expects without a scattering of ``cvtColor``
calls that are easy to forget and impossible to spot in review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import Rect, Size

__all__ = ["Frame", "FrameRef", "MediaInfo", "source_id_for"]


def source_id_for(path: Path) -> str:
    """Stable identity for a media file.

    The resolved absolute path is used so that annotations survive the file
    being referred to via a symlink or a relative path, but *not* the file being
    moved.  Triage deliberately moves videos between class folders, so anything
    that persists annotations must re-key on move — see
    ``persistence.sidecar``, which stores the sidecar next to the video.
    """
    return str(path.resolve())


@dataclass(frozen=True, slots=True)
class FrameRef:
    """Identifies one frame of one source. The key for every annotation."""

    source_id: str
    index: int

    def __str__(self) -> str:
        return f"{Path(self.source_id).name}#{self.index}"


@dataclass(frozen=True, slots=True)
class MediaInfo:
    """Static properties of an opened media source."""

    source_id: str
    path: Path
    size: Size
    frame_count: int
    fps: float

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    @property
    def is_still(self) -> bool:
        return self.frame_count <= 1

    def time_of(self, index: int) -> float:
        return index / self.fps if self.fps > 0 else 0.0

    def index_of(self, seconds: float) -> int:
        if self.fps <= 0:
            return 0
        return max(0, min(int(seconds * self.fps), max(self.frame_count - 1, 0)))


@dataclass(frozen=True)
class Frame:
    """One decoded frame: RGB ``uint8`` pixels plus where it came from."""

    ref: FrameRef
    image: np.ndarray  # (H, W, 3) uint8, RGB
    timestamp: float = 0.0
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        img = self.image
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Frame image must be (H, W, 3) RGB, got shape {img.shape}")
        if img.dtype != np.uint8:
            raise ValueError(f"Frame image must be uint8, got {img.dtype}")

    @property
    def size(self) -> Size:
        h, w = self.image.shape[:2]
        return Size(float(w), float(h))

    @property
    def rect(self) -> Rect:
        h, w = self.image.shape[:2]
        return Rect(0, 0, float(w), float(h))

    @property
    def index(self) -> int:
        return self.ref.index

    def crop(self, region: Rect) -> np.ndarray:
        """RGB pixels inside ``region``, clipped to the frame and never empty."""
        rows, cols = region.to_pixel_slice(self.size)
        return self.image[rows, cols]

    def to_bgr(self) -> np.ndarray:
        """A BGR view for the handful of OpenCV calls that insist on it."""
        return self.image[:, :, ::-1]

    def with_image(self, image: np.ndarray) -> Frame:
        """Same identity, different pixels — for filters that transform a frame."""
        return Frame(ref=self.ref, image=image, timestamp=self.timestamp, meta=dict(self.meta))
