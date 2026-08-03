"""Sidecars, atomic writes, exporters and settings."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vidtriage.core.annotations import Annotation, AnnotationStore
from vidtriage.core.frames import FrameRef, source_id_for
from vidtriage.core.geometry import Mask, Point, Polygon, Rect, Size
from vidtriage.persistence.exporters import (
    CocoExporter,
    CsvExporter,
    ExportRequest,
    YoloExporter,
    items_from_store,
)
from vidtriage.persistence.settings import Settings
from vidtriage.persistence.sidecar import (
    SIDECAR_SUFFIX,
    load_annotations,
    load_into_store,
    save_annotations,
    save_store,
    sidecar_path_for,
    write_json_atomic,
)


@pytest.fixture
def media(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not really a video")
    return path


def annotations_for(media, count=3):
    source = source_id_for(media)
    return [
        Annotation(FrameRef(source, i), Rect(i, i, i + 10, i + 20), label=f"c{i}", score=0.5)
        for i in range(count)
    ]


class TestSidecar:
    def test_path_keeps_the_full_filename(self, tmp_path):
        """``clip.mp4`` and ``clip.avi`` must not share one sidecar."""
        assert sidecar_path_for(tmp_path / "clip.mp4").name == f"clip.mp4{SIDECAR_SUFFIX}"
        assert (
            sidecar_path_for(tmp_path / "clip.avi")
            != sidecar_path_for(tmp_path / "clip.mp4")
        )

    def test_roundtrip(self, media):
        original = annotations_for(media)
        save_annotations(media, original, Size(640, 480))
        restored = load_annotations(media)
        assert len(restored) == 3
        assert {a.id for a in restored} == {a.id for a in original}
        assert restored[0].label == "c0"

    def test_mask_survives_the_roundtrip(self, media):
        full = np.zeros((40, 60), bool)
        full[5:20, 10:35] = True
        save_annotations(
            media,
            [Annotation(FrameRef(source_id_for(media), 0), Mask.from_full_frame(full))],
        )
        restored = load_annotations(media)[0]
        assert np.array_equal(restored.geometry.to_full_frame(Size(60, 40)), full)

    def test_saving_nothing_removes_the_sidecar(self, media):
        save_annotations(media, annotations_for(media))
        assert sidecar_path_for(media).exists()
        save_annotations(media, [])
        assert not sidecar_path_for(media).exists()

    def test_missing_sidecar_yields_no_annotations(self, media):
        assert load_annotations(media) == []

    def test_corrupt_sidecar_does_not_block_opening_the_video(self, media):
        sidecar_path_for(media).write_text("{ this is not json")
        assert load_annotations(media) == []

    def test_store_helpers(self, media):
        store = AnnotationStore(source_id_for(media))
        store.add(annotations_for(media, 2))
        save_store(store, media, Size(100, 100))
        assert not store.is_dirty

        restored = AnnotationStore()
        assert load_into_store(restored, media) == 2
        assert restored.source_id == source_id_for(media)


class TestAtomicWrite:
    def test_leaves_no_temp_files(self, tmp_path):
        target = tmp_path / "out.json"
        write_json_atomic(target, {"a": 1})
        assert json.loads(target.read_text()) == {"a": 1}
        assert list(tmp_path.iterdir()) == [target]

    def test_previous_content_survives_a_failed_write(self, tmp_path, monkeypatch):
        """A crash mid-save must not truncate an hour of labelling."""
        target = tmp_path / "out.json"
        write_json_atomic(target, {"good": True})

        def explode(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("os.replace", explode)
        from vidtriage.core.errors import PersistenceError

        with pytest.raises(PersistenceError):
            write_json_atomic(target, {"good": False})

        assert json.loads(target.read_text()) == {"good": True}
        assert list(tmp_path.iterdir()) == [target], "temp file was left behind"


class TestExporters:
    @pytest.fixture
    def request_for(self, tmp_path, media):
        def _make(destination_name="out", write_frames=False):
            source = source_id_for(media)
            full = np.zeros((100, 200), bool)
            full[10:30, 20:60] = True
            annotations = [
                Annotation(FrameRef(source, 0), Rect(10, 20, 50, 80), label="car", score=0.9),
                Annotation(FrameRef(source, 0), Point(5, 5), label="tip"),
                Annotation(
                    FrameRef(source, 2),
                    Polygon.from_iterable([(0, 0), (10, 0), (10, 10)]),
                    label="sign",
                ),
                Annotation(FrameRef(source, 2), Mask.from_full_frame(full), label="road"),
            ]
            return ExportRequest(
                destination=tmp_path / destination_name,
                items=items_from_store(media, Size(200, 100), annotations),
                write_frames=write_frames,
            )

        return _make

    def test_items_are_grouped_by_frame(self, request_for):
        request = request_for()
        assert [item.frame_index for item in request.items] == [0, 2]
        assert len(request.items[0].annotations) == 2
        assert request.labels == ["car", "road", "sign", "tip"]

    def test_coco_structure(self, request_for):
        report = CocoExporter().export(request_for("out.json"))
        payload = json.loads(report.written[0].read_text())

        assert len(payload["images"]) == 2
        assert len(payload["annotations"]) == 4
        assert {c["name"] for c in payload["categories"]} == {"car", "road", "sign", "tip"}

        car = next(a for a in payload["annotations"] if a["bbox"] == [10, 20, 40, 60])
        assert car["score"] == 0.9

        polygon = next(a for a in payload["annotations"] if isinstance(a.get("segmentation"), list))
        assert polygon["segmentation"] == [[0, 0, 10, 0, 10, 10]]

        mask = next(
            a for a in payload["annotations"] if isinstance(a.get("segmentation"), dict)
        )
        assert mask["segmentation"]["size"] == [100, 200]
        assert mask["area"] == 20 * 40

    def test_coco_warns_about_points(self, request_for):
        report = CocoExporter().export(request_for("out.json"))
        assert any("point" in w for w in report.warnings)

    def test_coco_forces_a_json_extension(self, request_for):
        report = CocoExporter().export(request_for("noext"))
        assert report.written[0].suffix == ".json"

    def test_yolo_writes_labels_and_classes(self, request_for):
        request = request_for("yolo_out")
        report = YoloExporter().export(request)
        classes = (request.destination / "classes.txt").read_text().split()
        assert classes == ["car", "road", "sign", "tip"]

        label_file = request.destination / "labels" / "clip_000000.txt"
        lines = label_file.read_text().strip().splitlines()
        assert len(lines) == 2
        class_id, cx, cy, w, h = lines[0].split()
        assert classes[int(class_id)] == "car"
        # Rect(10,20,50,80) in a 200x100 image → centre (30,50), size (40,60).
        assert float(cx) == pytest.approx(0.15)
        assert float(cy) == pytest.approx(0.50)
        assert float(w) == pytest.approx(0.20)
        assert float(h) == pytest.approx(0.60)
        assert report.annotation_count == 4

    def test_yolo_reports_that_masks_were_flattened(self, request_for):
        report = YoloExporter().export(request_for("yolo_out"))
        assert any("bounding boxes" in w for w in report.warnings)

    def test_csv_has_one_row_per_annotation(self, request_for):
        import csv

        report = CsvExporter().export(request_for("out.csv"))
        rows = list(csv.DictReader(report.written[0].open()))
        assert len(rows) == 4
        assert {r["kind"] for r in rows} == {"box", "point", "polygon", "mask"}
        car = next(r for r in rows if r["label"] == "car")
        assert car["frame"] == "0" and car["image_width"] == "200"

    def test_frame_extraction_reports_an_unreadable_source(self, request_for):
        """The fixture 'video' is 18 bytes of nonsense; that must be a warning."""
        report = CocoExporter().export(request_for("out.json", write_frames=True))
        assert any("cannot open" in w or "unreadable" in w for w in report.warnings)


class TestSettings:
    def test_dotted_keys(self, tmp_path):
        settings = Settings(tmp_path / "s.json")
        settings.set("window.size.width", 1280)
        assert settings.get("window.size.width") == 1280
        assert settings.get("window.size") == {"width": 1280}
        assert settings.get("window.missing", "fallback") == "fallback"

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "s.json"
        Settings(path).set("ui.theme", "Nord Dark")
        assert Settings(path).get("ui.theme") == "Nord Dark"

    def test_corrupt_file_is_backed_up_not_fatal(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text("}{ broken")
        settings = Settings(path)
        assert settings.get("anything") is None
        assert path.with_suffix(".json.corrupt").exists()
        settings.set("ok", 1)
        assert Settings(path).get("ok") == 1

    def test_update_writes_once(self, tmp_path):
        settings = Settings(tmp_path / "s.json")
        settings.update({"a.b": 1, "a.c": 2})
        assert settings.get("a") == {"b": 1, "c": 2}

    def test_remove(self, tmp_path):
        settings = Settings(tmp_path / "s.json")
        settings.set("a.b", 1)
        settings.remove("a.b")
        assert settings.get("a.b") is None
