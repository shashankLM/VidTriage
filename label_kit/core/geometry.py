"""Image-space geometry primitives.

Every shape in label-kit is expressed in **image pixel coordinates**: origin at
the top-left of the decoded frame, ``x`` growing right, ``y`` growing down,
units of source pixels (not display pixels, not normalised).

Keeping one canonical space is what makes model prompts, stored annotations and
on-screen graphics interchangeable.  Widget coordinates only exist inside the
view layer, and :class:`ViewTransform` is the single sanctioned bridge between
the two.

This module deliberately has no Qt dependency so it can be tested headlessly.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "Geometry",
    "Mask",
    "Point",
    "Polygon",
    "Rect",
    "Size",
    "ViewTransform",
    "bounding_rect_of",
]


@dataclass(frozen=True, slots=True)
class Point:
    """A single location in image pixel coordinates."""

    x: float
    y: float

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y

    def translated(self, dx: float, dy: float) -> Point:
        return Point(self.x + dx, self.y + dy)

    def scaled(self, factor: float) -> Point:
        return Point(self.x * factor, self.y * factor)

    def distance_to(self, other: Point) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def clamped_to(self, size: Size) -> Point:
        return Point(
            min(max(self.x, 0.0), max(size.width - 1, 0.0)),
            min(max(self.y, 0.0), max(size.height - 1, 0.0)),
        )

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def as_int(self) -> tuple[int, int]:
        return (round(self.x), round(self.y))


@dataclass(frozen=True, slots=True)
class Size:
    """Pixel dimensions of an image or region."""

    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0:
            raise ValueError(f"Size must be non-negative, got {self.width}x{self.height}")

    @property
    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0

    def as_int(self) -> tuple[int, int]:
        return (round(self.width), round(self.height))


@dataclass(frozen=True, slots=True)
class Rect:
    """An axis-aligned rectangle, always stored normalised (x1<=x2, y1<=y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        # Frozen dataclass: normalise via object.__setattr__ rather than rejecting
        # inverted input, because rubber-band drags legitimately produce it.
        # Read both ends before writing either — writing x1 first would make the
        # subsequent read of x1 see the new value and collapse the rect.
        x1, x2 = self.x1, self.x2
        if x1 > x2:
            object.__setattr__(self, "x1", x2)
            object.__setattr__(self, "x2", x1)
        y1, y2 = self.y1, self.y2
        if y1 > y2:
            object.__setattr__(self, "y1", y2)
            object.__setattr__(self, "y2", y1)

    @classmethod
    def from_corners(cls, a: Point, b: Point) -> Rect:
        return cls(a.x, a.y, b.x, b.y)

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> Rect:
        return cls(x, y, x + w, y + h)

    @classmethod
    def from_center(cls, center: Point, size: Size) -> Rect:
        return cls(
            center.x - size.width / 2,
            center.y - size.height / 2,
            center.x + size.width / 2,
            center.y + size.height / 2,
        )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def size(self) -> Size:
        return Size(self.width, self.height)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point:
        return Point((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def top_left(self) -> Point:
        return Point(self.x1, self.y1)

    @property
    def bottom_right(self) -> Point:
        return Point(self.x2, self.y2)

    @property
    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    def contains(self, p: Point) -> bool:
        return self.x1 <= p.x <= self.x2 and self.y1 <= p.y <= self.y2

    def translated(self, dx: float, dy: float) -> Rect:
        return Rect(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)

    def expanded(self, margin: float) -> Rect:
        return Rect(self.x1 - margin, self.y1 - margin, self.x2 + margin, self.y2 + margin)

    def scaled(self, factor: float) -> Rect:
        return Rect(self.x1 * factor, self.y1 * factor, self.x2 * factor, self.y2 * factor)

    def intersected(self, other: Rect) -> Rect:
        x1, y1 = max(self.x1, other.x1), max(self.y1, other.y1)
        x2, y2 = min(self.x2, other.x2), min(self.y2, other.y2)
        if x2 < x1 or y2 < y1:
            return Rect(0, 0, 0, 0)
        return Rect(x1, y1, x2, y2)

    def united(self, other: Rect) -> Rect:
        return Rect(
            min(self.x1, other.x1), min(self.y1, other.y1),
            max(self.x2, other.x2), max(self.y2, other.y2),
        )

    def iou(self, other: Rect) -> float:
        inter = self.intersected(other).area
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def clamped_to(self, size: Size) -> Rect:
        """Clip to the image bounds. May return an empty rect if fully outside."""
        return self.intersected(Rect(0, 0, size.width, size.height))

    def to_pixel_slice(self, size: Size) -> tuple[slice, slice]:
        """Integer ``(row_slice, col_slice)`` for cropping a numpy image.

        Always yields a slice that is in-bounds and at least 1px in each
        dimension when the rect overlaps the image at all, so callers never get
        a zero-sized crop from a sloppy one-pixel drag.
        """
        clipped = self.clamped_to(size)
        x1 = math.floor(clipped.x1)
        y1 = math.floor(clipped.y1)
        x2 = math.ceil(clipped.x2)
        y2 = math.ceil(clipped.y2)
        x2 = min(max(x2, x1 + 1), int(size.width))
        y2 = min(max(y2, y1 + 1), int(size.height))
        x1 = max(0, min(x1, x2 - 1))
        y1 = max(0, min(y1, y2 - 1))
        return slice(y1, y2), slice(x1, x2)

    def as_xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def normalised(self, size: Size) -> tuple[float, float, float, float]:
        """``(cx, cy, w, h)`` in 0..1 — the YOLO label convention."""
        if size.is_empty:
            return (0.0, 0.0, 0.0, 0.0)
        c = self.center
        return (
            c.x / size.width, c.y / size.height,
            self.width / size.width, self.height / size.height,
        )


@dataclass(frozen=True)
class Polygon:
    """A closed polyline in image coordinates."""

    points: tuple[Point, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", tuple(self.points))

    @classmethod
    def from_iterable(cls, pts: Iterable[Sequence[float]]) -> Polygon:
        return cls(tuple(Point(float(p[0]), float(p[1])) for p in pts))

    def __len__(self) -> int:
        return len(self.points)

    @property
    def is_valid(self) -> bool:
        return len(self.points) >= 3

    @property
    def bounding_rect(self) -> Rect:
        if not self.points:
            return Rect(0, 0, 0, 0)
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        return Rect(min(xs), min(ys), max(xs), max(ys))

    @property
    def area(self) -> float:
        """Absolute shoelace area."""
        if len(self.points) < 3:
            return 0.0
        total = 0.0
        for i, p in enumerate(self.points):
            q = self.points[(i + 1) % len(self.points)]
            total += p.x * q.y - q.x * p.y
        return abs(total) / 2

    def translated(self, dx: float, dy: float) -> Polygon:
        return Polygon(tuple(p.translated(dx, dy) for p in self.points))

    def as_flat_list(self) -> list[float]:
        """``[x0, y0, x1, y1, …]`` — the COCO segmentation convention."""
        return [v for p in self.points for v in (p.x, p.y)]

    def as_array(self) -> np.ndarray:
        if not self.points:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array([[p.x, p.y] for p in self.points], dtype=np.float32)


@dataclass(frozen=True)
class Mask:
    """A binary segmentation mask covering ``bounds`` within the full image.

    Storing the mask cropped to its bounding box rather than at full frame size
    keeps memory sane when a frame carries dozens of SAM masks.
    """

    data: np.ndarray  # bool, shape (bounds.height, bounds.width)
    bounds: Rect

    def __post_init__(self) -> None:
        if self.data.dtype != np.bool_:
            object.__setattr__(self, "data", self.data.astype(bool))
        if self.data.ndim != 2:
            raise ValueError(f"Mask data must be 2-D, got shape {self.data.shape}")

    @classmethod
    def from_full_frame(cls, mask: np.ndarray) -> Mask:
        """Crop a full-frame boolean mask down to its tight bounding box."""
        m = mask.astype(bool)
        if m.ndim != 2:
            raise ValueError(f"Expected 2-D mask, got shape {mask.shape}")
        rows = np.flatnonzero(m.any(axis=1))
        cols = np.flatnonzero(m.any(axis=0))
        if rows.size == 0 or cols.size == 0:
            return cls(np.zeros((0, 0), dtype=bool), Rect(0, 0, 0, 0))
        y1, y2 = int(rows[0]), int(rows[-1]) + 1
        x1, x2 = int(cols[0]), int(cols[-1]) + 1
        return cls(m[y1:y2, x1:x2].copy(), Rect(x1, y1, x2, y2))

    @property
    def is_empty(self) -> bool:
        return self.data.size == 0 or not self.data.any()

    @property
    def pixel_count(self) -> int:
        return int(self.data.sum())

    @property
    def bounding_rect(self) -> Rect:
        return self.bounds

    def to_full_frame(self, size: Size) -> np.ndarray:
        """Re-expand into a full-frame boolean array of ``size``."""
        w, h = size.as_int()
        out = np.zeros((h, w), dtype=bool)
        if self.is_empty:
            return out
        x1, y1 = int(self.bounds.x1), int(self.bounds.y1)
        mh, mw = self.data.shape
        y2, x2 = min(y1 + mh, h), min(x1 + mw, w)
        if y2 <= y1 or x2 <= x1:
            return out
        out[y1:y2, x1:x2] = self.data[: y2 - y1, : x2 - x1]
        return out


Geometry = Point | Rect | Polygon | Mask


def bounding_rect_of(geometry: Geometry) -> Rect:
    """Bounding rect for any geometry kind — a zero-area rect for a bare point."""
    if isinstance(geometry, Point):
        return Rect(geometry.x, geometry.y, geometry.x, geometry.y)
    if isinstance(geometry, Rect):
        return geometry
    return geometry.bounding_rect


@dataclass(frozen=True, slots=True)
class ViewTransform:
    """Maps image pixels to widget pixels: ``widget = image * scale + offset``.

    The view layer keeps one of these in sync with the actual Qt view so that
    layers, tools and hit-tests can reason about screen distances (a 6px grab
    handle stays 6px regardless of zoom) without reaching into Qt internals.
    """

    scale: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError(f"ViewTransform.scale must be positive, got {self.scale}")

    @classmethod
    def fit(cls, image: Size, viewport: Size, allow_upscale: bool = True) -> ViewTransform:
        """Letterbox ``image`` inside ``viewport``, centred."""
        if image.is_empty or viewport.is_empty:
            return cls()
        scale = min(viewport.width / image.width, viewport.height / image.height)
        if not allow_upscale:
            scale = min(scale, 1.0)
        return cls(
            scale=scale,
            offset_x=(viewport.width - image.width * scale) / 2,
            offset_y=(viewport.height - image.height * scale) / 2,
        )

    def image_to_widget(self, p: Point) -> Point:
        return Point(p.x * self.scale + self.offset_x, p.y * self.scale + self.offset_y)

    def widget_to_image(self, p: Point) -> Point:
        return Point((p.x - self.offset_x) / self.scale, (p.y - self.offset_y) / self.scale)

    def image_rect_to_widget(self, r: Rect) -> Rect:
        return Rect.from_corners(
            self.image_to_widget(r.top_left), self.image_to_widget(r.bottom_right),
        )

    def widget_rect_to_image(self, r: Rect) -> Rect:
        return Rect.from_corners(
            self.widget_to_image(r.top_left), self.widget_to_image(r.bottom_right),
        )

    def widget_length_to_image(self, length: float) -> float:
        """Convert a screen distance (e.g. a grab radius) into image units."""
        return length / self.scale

    def zoomed(self, factor: float, anchor: Point) -> ViewTransform:
        """Scale by ``factor`` keeping the widget-space ``anchor`` pixel fixed."""
        new_scale = self.scale * factor
        if new_scale <= 0:
            return self
        img_anchor = self.widget_to_image(anchor)
        return replace(
            self,
            scale=new_scale,
            offset_x=anchor.x - img_anchor.x * new_scale,
            offset_y=anchor.y - img_anchor.y * new_scale,
        )

    def panned(self, dx: float, dy: float) -> ViewTransform:
        return replace(self, offset_x=self.offset_x + dx, offset_y=self.offset_y + dy)
