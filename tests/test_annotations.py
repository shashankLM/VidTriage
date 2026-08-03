"""The annotation store: indexing, undo, RLE, serialisation."""

from __future__ import annotations

import numpy as np
import pytest

from label_kit.core.annotations import (
    MANUAL_SOURCE,
    Annotation,
    AnnotationKind,
    AnnotationStore,
    rle_decode,
    rle_encode,
)
from label_kit.core.frames import FrameRef
from label_kit.core.geometry import Mask, Point, Polygon, Rect, Size

SOURCE = "/videos/clip.mp4"


def box(frame: int = 0, x: float = 0, y: float = 0, label: str = "car", **kw) -> Annotation:
    return Annotation(FrameRef(SOURCE, frame), Rect(x, y, x + 10, y + 10), label=label, **kw)


class TestRle:
    @pytest.mark.parametrize(
        "mask",
        [
            np.zeros((4, 5), bool),
            np.ones((4, 5), bool),
            np.eye(6, dtype=bool),
            np.random.default_rng(0).random((17, 23)) > 0.5,
        ],
    )
    def test_roundtrip(self, mask):
        assert np.array_equal(rle_decode(rle_encode(mask), mask.shape), mask)

    def test_starts_with_a_zero_run(self):
        """COCO's convention: counts always begin with a run of background."""
        counts = rle_encode(np.ones((2, 2), bool))
        assert counts[0] == 0


class TestKindDispatch:
    @pytest.mark.parametrize(
        ("geometry", "expected"),
        [
            (Point(1, 2), AnnotationKind.POINT),
            (Rect(0, 0, 1, 1), AnnotationKind.BOX),
            (Polygon.from_iterable([(0, 0), (1, 0), (1, 1)]), AnnotationKind.POLYGON),
            (Mask.from_full_frame(np.ones((2, 2), bool)), AnnotationKind.MASK),
        ],
    )
    def test_kind_follows_geometry(self, geometry, expected):
        assert Annotation(FrameRef(SOURCE, 0), geometry).kind is expected


class TestStore:
    def test_indexes_by_frame(self):
        store = AnnotationStore(SOURCE)
        store.add([box(frame=1), box(frame=1, x=20), box(frame=7)])
        assert len(store.for_frame(1)) == 2
        assert len(store.for_frame(7)) == 1
        assert store.for_frame(99) == []
        assert store.frames_with_annotations() == [1, 7]

    def test_hit_test_prefers_the_smallest_shape(self):
        """A small box inside a big one must stay selectable."""
        store = AnnotationStore(SOURCE)
        big = Annotation(FrameRef(SOURCE, 0), Rect(0, 0, 100, 100), label="big")
        small = Annotation(FrameRef(SOURCE, 0), Rect(40, 40, 60, 60), label="small")
        store.add([big, small])
        assert store.hit_test(0, Point(50, 50)).label == "small"
        assert store.hit_test(0, Point(5, 5)).label == "big"
        assert store.hit_test(0, Point(500, 500)) is None

    def test_undo_redo_add(self):
        store = AnnotationStore(SOURCE)
        store.add(box())
        assert len(store) == 1 and store.can_undo
        assert store.undo() and len(store) == 0
        assert store.can_redo
        assert store.redo() and len(store) == 1

    def test_undo_restores_previous_geometry(self):
        store = AnnotationStore(SOURCE)
        original = box()
        store.add(original)
        store.update(original.with_geometry(Rect(0, 0, 99, 99)))
        assert store.by_id(original.id).geometry.x2 == 99
        store.undo()
        assert store.by_id(original.id).geometry.x2 == 10

    def test_update_can_move_an_annotation_between_frames(self):
        store = AnnotationStore(SOURCE)
        annotation = box(frame=1)
        store.add(annotation)
        moved = Annotation(
            FrameRef(SOURCE, 5), annotation.geometry, annotation.label, id=annotation.id,
        )
        store.update(moved)
        assert store.for_frame(1) == []
        assert len(store.for_frame(5)) == 1

    def test_undo_of_a_delete_restores_it(self):
        store = AnnotationStore(SOURCE)
        annotation = box()
        store.add(annotation)
        store.remove(annotation)
        assert len(store) == 0
        store.undo()
        assert store.by_id(annotation.id) is not None

    def test_removing_an_unknown_annotation_is_a_no_op(self):
        """Otherwise undo would 'restore' something that was never there."""
        store = AnnotationStore(SOURCE)
        store.add(box())
        before = store.can_undo
        assert store.remove(box(frame=3)) == []
        assert store.can_undo == before

    def test_remove_by_source_clears_one_model(self):
        store = AnnotationStore(SOURCE)
        store.add([
            box(label="a", source="yolo.detect"),
            box(label="b", source="sam.predict"),
            box(label="c"),
        ])
        removed = store.remove_by_source("yolo.detect")
        assert len(removed) == 1
        assert {a.source for a in store} == {"sam.predict", MANUAL_SOURCE}

    def test_reset_clears_undo_history(self):
        store = AnnotationStore(SOURCE)
        store.add(box())
        store.reset([box(frame=4)])
        assert not store.can_undo and not store.can_redo
        assert len(store) == 1 and not store.is_dirty

    def test_dirty_flag_tracks_edits(self):
        store = AnnotationStore(SOURCE)
        assert not store.is_dirty
        store.add(box())
        assert store.is_dirty
        store.mark_clean()
        assert not store.is_dirty

    def test_change_events_report_the_action(self):
        store = AnnotationStore(SOURCE)
        seen = []
        store.changed.connect(lambda change: seen.append(change.action))
        annotation = box()
        store.add(annotation)
        store.update(annotation.with_label("van"))
        store.remove(annotation)
        assert seen == ["added", "updated", "removed"]

    def test_a_raising_handler_does_not_corrupt_the_store(self):
        store = AnnotationStore(SOURCE)
        store.changed.connect(lambda _c: (_ for _ in ()).throw(RuntimeError("boom")))
        seen = []
        store.changed.connect(lambda c: seen.append(c.action))
        store.add(box())
        assert len(store) == 1
        assert seen == ["added"]


class TestSerialisation:
    @pytest.mark.parametrize(
        "geometry",
        [
            Point(1.5, 2.5),
            Rect(1, 2, 30, 40),
            Polygon.from_iterable([(0, 0), (10, 0), (10, 10)]),
        ],
    )
    def test_geometry_roundtrip(self, geometry):
        original = Annotation(FrameRef(SOURCE, 3), geometry, label="x", score=0.5)
        restored = Annotation.from_dict(original.to_dict(), SOURCE)
        assert restored.geometry == original.geometry
        assert restored.id == original.id
        assert restored.score == 0.5

    def test_mask_roundtrip(self):
        full = np.zeros((20, 30), bool)
        full[5:12, 8:20] = True
        original = Annotation(FrameRef(SOURCE, 0), Mask.from_full_frame(full))
        restored = Annotation.from_dict(original.to_dict(), SOURCE)
        assert np.array_equal(
            restored.geometry.to_full_frame(Size(30, 20)),
            original.geometry.to_full_frame(Size(30, 20)),
        )

    def test_store_roundtrip(self):
        store = AnnotationStore(SOURCE)
        store.add([box(frame=1), box(frame=2, label="van", score=0.8, source="yolo")])
        restored = AnnotationStore()
        restored.load_dict(store.to_dict(Size(640, 480)))
        assert len(restored) == 2
        assert restored.labels() == ["car", "van"]

    def test_malformed_entries_are_skipped_not_fatal(self):
        store = AnnotationStore()
        store.load_dict({
            "source_id": SOURCE,
            "annotations": [
                {"frame": 0, "geometry": {"kind": "box", "xyxy": [0, 0, 1, 1]}, "label": "ok"},
                {"frame": 0, "geometry": {"kind": "nonsense"}},
                {"no_frame_key": True},
            ],
        })
        assert len(store) == 1
        assert store.all()[0].label == "ok"
