"""Frame-level annotations and the store that owns them.

An :class:`Annotation` is a geometry (point / box / polygon / mask) pinned to a
single :class:`~vidtriage.core.frames.FrameRef`, carrying a label and its
provenance.  Manual drawings and model predictions are *the same type* — the
``source`` field is the only difference — which is what lets a SAM mask be
edited by hand, relabelled, or exported side by side with a hand-drawn box.

:class:`AnnotationStore` is the single mutable owner.  It emits events on every
change and records a reversible edit for undo.  Nothing else in the app is
allowed to mutate an annotation in place.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Literal

import numpy as np

from .events import Event
from .frames import FrameRef
from .geometry import Geometry, Mask, Point, Polygon, Rect, Size, bounding_rect_of
from .logging import get_logger

__all__ = [
    "MANUAL_SOURCE",
    "Annotation",
    "AnnotationChange",
    "AnnotationKind",
    "AnnotationStore",
    "rle_decode",
    "rle_encode",
]

_log = get_logger(__name__)

MANUAL_SOURCE = "manual"


class AnnotationKind(StrEnum):
    POINT = "point"
    BOX = "box"
    POLYGON = "polygon"
    MASK = "mask"

    @classmethod
    def of(cls, geometry: Geometry) -> AnnotationKind:
        if isinstance(geometry, Point):
            return cls.POINT
        if isinstance(geometry, Rect):
            return cls.BOX
        if isinstance(geometry, Polygon):
            return cls.POLYGON
        if isinstance(geometry, Mask):
            return cls.MASK
        raise TypeError(f"unsupported geometry type: {type(geometry).__name__}")


# ── mask run-length coding ──────────────────────────────────────────────


def rle_encode(mask: np.ndarray) -> list[int]:
    """Column-major run lengths starting with a run of zeros (COCO convention)."""
    flat = np.asarray(mask, dtype=bool).ravel(order="F")
    if flat.size == 0:
        return []
    # Boundaries between runs, plus the implicit end.
    changes = np.flatnonzero(np.diff(flat)) + 1
    edges = np.concatenate(([0], changes, [flat.size]))
    counts = np.diff(edges).tolist()
    # COCO always starts counting zeros; prepend an empty run if the mask starts set.
    if flat[0]:
        counts.insert(0, 0)
    return [int(c) for c in counts]


def rle_decode(counts: Iterable[int], shape: tuple[int, int]) -> np.ndarray:
    """Inverse of :func:`rle_encode`."""
    height, width = shape
    flat = np.zeros(height * width, dtype=bool)
    position = 0
    value = False
    for count in counts:
        end = min(position + int(count), flat.size)
        if value and end > position:
            flat[position:end] = True
        position = end
        value = not value
        if position >= flat.size:
            break
    return flat.reshape((height, width), order="F")


# ── geometry serialisation ──────────────────────────────────────────────


def _geometry_to_dict(geometry: Geometry) -> dict[str, Any]:
    kind = AnnotationKind.of(geometry)
    if kind is AnnotationKind.POINT:
        return {"kind": kind.value, "x": geometry.x, "y": geometry.y}
    if kind is AnnotationKind.BOX:
        return {"kind": kind.value, "xyxy": list(geometry.as_xyxy())}
    if kind is AnnotationKind.POLYGON:
        return {"kind": kind.value, "points": geometry.as_flat_list()}
    mask: Mask = geometry
    return {
        "kind": kind.value,
        "bounds": list(mask.bounds.as_xyxy()),
        "shape": list(mask.data.shape),
        "rle": rle_encode(mask.data),
    }


def _geometry_from_dict(data: dict[str, Any]) -> Geometry:
    kind = AnnotationKind(data["kind"])
    if kind is AnnotationKind.POINT:
        return Point(float(data["x"]), float(data["y"]))
    if kind is AnnotationKind.BOX:
        x1, y1, x2, y2 = (float(v) for v in data["xyxy"])
        return Rect(x1, y1, x2, y2)
    if kind is AnnotationKind.POLYGON:
        flat = [float(v) for v in data["points"]]
        return Polygon.from_iterable(zip(flat[0::2], flat[1::2], strict=False))
    shape = (int(data["shape"][0]), int(data["shape"][1]))
    x1, y1, x2, y2 = (float(v) for v in data["bounds"])
    return Mask(rle_decode(data["rle"], shape), Rect(x1, y1, x2, y2))


# ── the annotation ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Annotation:
    """One labelled shape on one frame. Immutable — edits produce a new object."""

    frame: FrameRef
    geometry: Geometry
    label: str = ""
    score: float | None = None
    source: str = MANUAL_SOURCE
    attributes: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    @property
    def kind(self) -> AnnotationKind:
        return AnnotationKind.of(self.geometry)

    @property
    def bounding_rect(self) -> Rect:
        return bounding_rect_of(self.geometry)

    @property
    def is_prediction(self) -> bool:
        return self.source != MANUAL_SOURCE

    def with_geometry(self, geometry: Geometry) -> Annotation:
        return replace(self, geometry=geometry)

    def with_label(self, label: str) -> Annotation:
        return replace(self, label=label)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "frame": self.frame.index,
            "geometry": _geometry_to_dict(self.geometry),
            "label": self.label,
            "source": self.source,
        }
        if self.score is not None:
            payload["score"] = self.score
        if self.attributes:
            payload["attributes"] = self.attributes
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any], source_id: str) -> Annotation:
        return cls(
            frame=FrameRef(source_id, int(data["frame"])),
            geometry=_geometry_from_dict(data["geometry"]),
            label=str(data.get("label", "")),
            score=float(data["score"]) if data.get("score") is not None else None,
            source=str(data.get("source", MANUAL_SOURCE)),
            attributes=dict(data.get("attributes", {})),
            id=str(data.get("id") or uuid.uuid4().hex[:16]),
        )


@dataclass(frozen=True, slots=True)
class AnnotationChange:
    action: Literal["added", "removed", "updated", "reset"]
    annotations: tuple[Annotation, ...]

    @property
    def frames(self) -> set[FrameRef]:
        return {a.frame for a in self.annotations}


# ── undo ────────────────────────────────────────────────────────────────


class _Edit(ABC):
    @abstractmethod
    def apply(self, store: AnnotationStore) -> None: ...

    @abstractmethod
    def revert(self, store: AnnotationStore) -> None: ...

    @property
    @abstractmethod
    def description(self) -> str: ...


@dataclass(slots=True)
class _AddEdit(_Edit):
    items: tuple[Annotation, ...]

    def apply(self, store: AnnotationStore) -> None:
        store._insert(self.items)

    def revert(self, store: AnnotationStore) -> None:
        store._delete(self.items)

    @property
    def description(self) -> str:
        return f"Add {len(self.items)} annotation(s)"


@dataclass(slots=True)
class _RemoveEdit(_Edit):
    items: tuple[Annotation, ...]

    def apply(self, store: AnnotationStore) -> None:
        store._delete(self.items)

    def revert(self, store: AnnotationStore) -> None:
        store._insert(self.items)

    @property
    def description(self) -> str:
        return f"Delete {len(self.items)} annotation(s)"


@dataclass(slots=True)
class _UpdateEdit(_Edit):
    before: tuple[Annotation, ...]
    after: tuple[Annotation, ...]

    def apply(self, store: AnnotationStore) -> None:
        store._replace(self.after)

    def revert(self, store: AnnotationStore) -> None:
        store._replace(self.before)

    @property
    def description(self) -> str:
        return f"Edit {len(self.after)} annotation(s)"


# ── the store ───────────────────────────────────────────────────────────


class AnnotationStore:
    """Owns every annotation for one media source, indexed by frame.

    Thread affinity: GUI thread only.  Inference results arrive from worker
    threads and *must* be marshalled through a queued Qt signal before being
    handed to :meth:`add` — see ``plugins.runner``.
    """

    def __init__(self, source_id: str = "", max_undo: int = 200) -> None:
        self.source_id = source_id
        self._by_frame: dict[int, list[Annotation]] = {}
        self._by_id: dict[str, Annotation] = {}
        self._undo: list[_Edit] = []
        self._redo: list[_Edit] = []
        self._max_undo = max_undo
        self._recording = True
        self._dirty = False

        self.changed: Event[AnnotationChange] = Event("annotations.changed")

    # ── queries ─────────────────────────────────────────────────────────

    def for_frame(self, frame: FrameRef | int) -> list[Annotation]:
        index = frame.index if isinstance(frame, FrameRef) else frame
        return list(self._by_frame.get(index, ()))

    def by_id(self, annotation_id: str) -> Annotation | None:
        return self._by_id.get(annotation_id)

    def all(self) -> list[Annotation]:
        return list(self._by_id.values())

    def frames_with_annotations(self) -> list[int]:
        return sorted(k for k, v in self._by_frame.items() if v)

    def labels(self) -> list[str]:
        return sorted({a.label for a in self._by_id.values() if a.label})

    def count_for_frame(self, frame: FrameRef | int) -> int:
        index = frame.index if isinstance(frame, FrameRef) else frame
        return len(self._by_frame.get(index, ()))

    def hit_test(
        self,
        frame: FrameRef | int,
        point: Point,
        tolerance: float = 0.0,
    ) -> Annotation | None:
        """Topmost annotation whose bounds contain ``point``.

        Smallest-area-first so a tiny box nested inside a big one stays
        selectable — the usual behaviour of every annotation tool.
        """
        candidates = [
            a for a in self.for_frame(frame)
            if a.bounding_rect.expanded(tolerance).contains(point)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda a: a.bounding_rect.area)

    @property
    def is_empty(self) -> bool:
        return not self._by_id

    @property
    def is_dirty(self) -> bool:
        """True when there are unsaved changes."""
        return self._dirty

    def mark_clean(self) -> None:
        self._dirty = False

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self) -> Iterator[Annotation]:
        return iter(list(self._by_id.values()))

    # ── mutation (public, undoable) ─────────────────────────────────────

    def add(self, annotations: Annotation | Iterable[Annotation]) -> list[Annotation]:
        items = _as_tuple(annotations)
        if not items:
            return []
        self._push(_AddEdit(items))
        return list(items)

    def remove(self, annotations: Annotation | Iterable[Annotation] | str) -> list[Annotation]:
        if isinstance(annotations, str):
            found = self._by_id.get(annotations)
            items: tuple[Annotation, ...] = (found,) if found else ()
        else:
            items = _as_tuple(annotations)
        # Only remove what is actually present, so undo restores an accurate state.
        items = tuple(a for a in items if a.id in self._by_id)
        if not items:
            return []
        self._push(_RemoveEdit(items))
        return list(items)

    def update(self, annotations: Annotation | Iterable[Annotation]) -> list[Annotation]:
        items = _as_tuple(annotations)
        before = tuple(self._by_id[a.id] for a in items if a.id in self._by_id)
        after = tuple(a for a in items if a.id in self._by_id)
        if not after:
            return []
        self._push(_UpdateEdit(before, after))
        return list(after)

    def remove_frame(self, frame: FrameRef | int) -> list[Annotation]:
        return self.remove(self.for_frame(frame))

    def remove_by_source(self, source: str) -> list[Annotation]:
        """Drop every prediction from one model — the 'clear auto-labels' action."""
        return self.remove([a for a in self._by_id.values() if a.source == source])

    def clear(self) -> None:
        self.remove(list(self._by_id.values()))

    # ── bulk load (not undoable) ────────────────────────────────────────

    def reset(self, annotations: Iterable[Annotation] = (), source_id: str | None = None) -> None:
        """Replace all content, e.g. when loading a sidecar. Clears undo history."""
        if source_id is not None:
            self.source_id = source_id
        self._by_frame.clear()
        self._by_id.clear()
        self._undo.clear()
        self._redo.clear()
        for annotation in annotations:
            self._by_id[annotation.id] = annotation
            self._by_frame.setdefault(annotation.frame.index, []).append(annotation)
        self._dirty = False
        self.changed.emit(AnnotationChange("reset", tuple(self._by_id.values())))

    # ── undo / redo ─────────────────────────────────────────────────────

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo_description(self) -> str | None:
        return self._undo[-1].description if self._undo else None

    def undo(self) -> bool:
        if not self._undo:
            return False
        edit = self._undo.pop()
        self._recording = False
        try:
            edit.revert(self)
        finally:
            self._recording = True
        self._redo.append(edit)
        self._dirty = True
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        edit = self._redo.pop()
        self._recording = False
        try:
            edit.apply(self)
        finally:
            self._recording = True
        self._undo.append(edit)
        self._dirty = True
        return True

    # ── internals ───────────────────────────────────────────────────────

    def _push(self, edit: _Edit) -> None:
        edit.apply(self)
        if not self._recording:
            return
        self._undo.append(edit)
        if len(self._undo) > self._max_undo:
            del self._undo[0]
        self._redo.clear()
        self._dirty = True

    def _insert(self, items: tuple[Annotation, ...]) -> None:
        for annotation in items:
            self._by_id[annotation.id] = annotation
            self._by_frame.setdefault(annotation.frame.index, []).append(annotation)
        self._dirty = True
        self.changed.emit(AnnotationChange("added", items))

    def _delete(self, items: tuple[Annotation, ...]) -> None:
        for annotation in items:
            self._by_id.pop(annotation.id, None)
            bucket = self._by_frame.get(annotation.frame.index)
            if bucket:
                self._by_frame[annotation.frame.index] = [
                    a for a in bucket if a.id != annotation.id
                ]
        self._dirty = True
        self.changed.emit(AnnotationChange("removed", items))

    def _replace(self, items: tuple[Annotation, ...]) -> None:
        for annotation in items:
            previous = self._by_id.get(annotation.id)
            self._by_id[annotation.id] = annotation
            # A geometry edit can move an annotation to a different frame bucket.
            if previous is not None and previous.frame.index != annotation.frame.index:
                bucket = self._by_frame.get(previous.frame.index, [])
                self._by_frame[previous.frame.index] = [
                    a for a in bucket if a.id != annotation.id
                ]
                self._by_frame.setdefault(annotation.frame.index, []).append(annotation)
            else:
                bucket = self._by_frame.setdefault(annotation.frame.index, [])
                for i, existing in enumerate(bucket):
                    if existing.id == annotation.id:
                        bucket[i] = annotation
                        break
                else:
                    bucket.append(annotation)
        self._dirty = True
        self.changed.emit(AnnotationChange("updated", items))

    # ── serialisation ───────────────────────────────────────────────────

    def to_dict(self, image_size: Size | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": 1,
            "source_id": self.source_id,
            "annotations": [a.to_dict() for a in self._by_id.values()],
        }
        if image_size is not None:
            payload["image_size"] = [image_size.width, image_size.height]
        return payload

    def load_dict(self, data: dict[str, Any], source_id: str | None = None) -> None:
        sid = source_id or str(data.get("source_id") or self.source_id)
        loaded: list[Annotation] = []
        for raw in data.get("annotations", []):
            try:
                loaded.append(Annotation.from_dict(raw, sid))
            except (KeyError, TypeError, ValueError) as exc:
                _log.warning("Skipping malformed annotation %r: %s", raw.get("id"), exc)
        self.reset(loaded, source_id=sid)


def _as_tuple(value: Annotation | Iterable[Annotation]) -> tuple[Annotation, ...]:
    if isinstance(value, Annotation):
        return (value,)
    return tuple(value)
