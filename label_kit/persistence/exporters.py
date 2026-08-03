"""Exporters — turning annotations into something a trainer can read.

Export is an extension point, not a fixed menu: an exporter is registered like
any other contribution, so adding Pascal VOC or a bespoke internal format is one
class and no edits elsewhere.

Frame extraction is opt-in. COCO and YOLO both reference image files, so an
annotation-only export is fine for inspection but useless for training; passing
``write_frames=True`` re-decodes and writes the referenced frames alongside the
labels, producing a directory you can point a training run at directly.
"""

from __future__ import annotations

import csv
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.annotations import Annotation, AnnotationKind, rle_encode
from ..core.geometry import Mask, Polygon, Rect, Size, bounding_rect_of
from ..core.logging import get_logger
from .sidecar import load_annotations, load_image_size, write_json_atomic

__all__ = [
    "CocoExporter",
    "CsvExporter",
    "ExportItem",
    "ExportReport",
    "ExportRequest",
    "Exporter",
    "YoloExporter",
    "builtin_exporters",
    "items_from_library",
    "items_from_store",
]

_log = get_logger(__name__)

_UNLABELLED = "unlabelled"
_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class ExportItem:
    """One frame's worth of annotations, with everything needed to describe it."""

    media_path: Path
    frame_index: int
    image_size: Size
    annotations: tuple[Annotation, ...]

    @property
    def stem(self) -> str:
        """Filename base that stays unique across videos and frames."""
        return f"{self.media_path.stem}_{self.frame_index:06d}"


@dataclass
class ExportRequest:
    destination: Path
    items: list[ExportItem]
    write_frames: bool = False
    image_format: str = "jpg"
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def labels(self) -> list[str]:
        """Every label present, sorted — the category list for the export."""
        found = {a.label or _UNLABELLED for item in self.items for a in item.annotations}
        return sorted(found)

    @property
    def annotation_count(self) -> int:
        return sum(len(item.annotations) for item in self.items)

    def stems(self) -> list[str]:
        """A filename base per item, in item order, guaranteed unique.

        :attr:`ExportItem.stem` is ``<filename>_<frame>``, which stops being
        unique the moment two source files share a name — ``a/clip.mp4`` and
        ``b/clip.mp4`` want the same label file and the same extracted frame.
        One would overwrite the other and the result would look like a complete
        dataset, so the parent directory disambiguates instead.
        """
        counts = Counter(item.stem for item in self.items)
        taken: set[str] = set()
        stems: list[str] = []
        for item in self.items:
            base = item.stem
            if counts[base] > 1:
                parent = _SLUG_UNSAFE.sub("-", item.media_path.parent.name).strip("-")
                base = f"{parent}_{base}" if parent else base
            candidate, suffix = base, 2
            while candidate in taken:
                candidate, suffix = f"{base}_{suffix}", suffix + 1
            taken.add(candidate)
            stems.append(candidate)
        return stems


@dataclass
class ExportReport:
    written: list[Path] = field(default_factory=list)
    item_count: int = 0
    annotation_count: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        parts = [
            f"{self.annotation_count} annotation(s) across {self.item_count} frame(s)",
            f"{len(self.written)} file(s) written",
        ]
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        return " · ".join(parts)


class Exporter(ABC):
    """Base class for an output format."""

    id: str = ""
    title: str = ""
    #: Qt file-dialog filter, e.g. ``"JSON (*.json)"``.
    file_filter: str = "All Files (*)"
    #: True when the destination is a directory rather than a single file.
    writes_directory: bool = False
    default_extension: str = ""
    #: Plugin that contributed this exporter, filled in by ``PluginContext``.
    owner: str | None = None

    @abstractmethod
    def export(self, request: ExportRequest) -> ExportReport: ...

    # ── shared helpers ──────────────────────────────────────────────────

    @staticmethod
    def _write_frames(request: ExportRequest, into: Path) -> tuple[list[Path], list[str]]:
        """Decode and save the referenced frames. Returns (written, warnings).

        Grouped by media file and sorted by frame index so each video is opened
        once and seeked forwards — the difference between a few seconds and
        several minutes on a long export.
        """
        import cv2

        from ..media.source import open_source

        written: list[Path] = []
        warnings: list[str] = []
        into.mkdir(parents=True, exist_ok=True)

        by_media: dict[Path, list[tuple[ExportItem, str]]] = {}
        for item, stem in zip(request.items, request.stems(), strict=True):
            by_media.setdefault(item.media_path, []).append((item, stem))

        for media_path, pairs in by_media.items():
            try:
                source = open_source(media_path)
            except Exception as exc:  # noqa: BLE001 - one bad file must not abort the export
                warnings.append(f"{media_path.name}: cannot open ({exc})")
                continue
            try:
                for item, stem in sorted(pairs, key=lambda pair: pair[0].frame_index):
                    frame = source.read_at(item.frame_index)
                    if frame is None:
                        warnings.append(f"{media_path.name}: frame {item.frame_index} unreadable")
                        continue
                    out = into / f"{stem}.{request.image_format}"
                    if cv2.imwrite(str(out), frame.to_bgr()):
                        written.append(out)
                    else:
                        warnings.append(f"could not write {out.name}")
            finally:
                source.close()

        return written, warnings

    @staticmethod
    def _box_of(annotation: Annotation) -> Rect:
        return bounding_rect_of(annotation.geometry)

    @staticmethod
    def _label_of(annotation: Annotation) -> str:
        return annotation.label or _UNLABELLED


class CocoExporter(Exporter):
    """COCO detection/segmentation JSON.

    Polygons export as ``segmentation`` point lists and masks as uncompressed
    RLE, both of which ``pycocotools`` reads directly. Points have no COCO
    equivalent, so they are exported as zero-area boxes and reported as a
    warning rather than silently dropped.
    """

    id = "coco"
    title = "COCO JSON"
    file_filter = "COCO JSON (*.json)"
    default_extension = ".json"

    def export(self, request: ExportRequest) -> ExportReport:
        report = ExportReport(item_count=len(request.items))
        categories = {label: i + 1 for i, label in enumerate(request.labels)}

        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        next_id = 1

        for image_id, (item, stem) in enumerate(
            zip(request.items, request.stems(), strict=True), start=1,
        ):
            width, height = item.image_size.as_int()
            images.append({
                "id": image_id,
                "file_name": f"{stem}.{request.image_format}",
                "width": width,
                "height": height,
                "label_kit_source": item.media_path.name,
                "label_kit_frame": item.frame_index,
            })

            for annotation in item.annotations:
                box = self._box_of(annotation)
                entry: dict[str, Any] = {
                    "id": next_id,
                    "image_id": image_id,
                    "category_id": categories[self._label_of(annotation)],
                    "bbox": [round(v, 2) for v in box.as_xywh()],
                    "area": round(box.area, 2),
                    "iscrowd": 0,
                    "label_kit_source": annotation.source,
                }
                if annotation.score is not None:
                    entry["score"] = round(annotation.score, 4)

                geometry = annotation.geometry
                if isinstance(geometry, Polygon):
                    entry["segmentation"] = [[round(v, 2) for v in geometry.as_flat_list()]]
                    entry["area"] = round(geometry.area, 2)
                elif isinstance(geometry, Mask):
                    full = geometry.to_full_frame(item.image_size)
                    entry["segmentation"] = {"counts": rle_encode(full), "size": [height, width]}
                    entry["area"] = int(full.sum())
                elif annotation.kind is AnnotationKind.POINT:
                    report.warnings.append(
                        f"{stem}: point '{self._label_of(annotation)}' exported "
                        f"as a zero-area box (COCO has no point geometry)",
                    )

                annotations.append(entry)
                next_id += 1

        report.annotation_count = len(annotations)
        destination = request.destination
        if destination.suffix.lower() != ".json":
            destination = destination.with_suffix(".json")

        write_json_atomic(destination, {
            "info": {"description": "Exported by label-kit"},
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": cid, "name": name, "supercategory": "none"}
                for name, cid in categories.items()
            ],
        })
        report.written.append(destination)

        if request.write_frames:
            frames, warnings = self._write_frames(request, destination.parent / "images")
            report.written.extend(frames)
            report.warnings.extend(warnings)

        return report


class YoloExporter(Exporter):
    """Ultralytics-style label directory: one ``.txt`` per frame plus ``classes.txt``.

    YOLO's detection format is boxes only, so polygons and masks export as their
    bounding boxes. That is lossy, so it is reported per shape kind rather than
    passing silently.

    **Class ids in an existing ``classes.txt`` are preserved.** A label file
    stores an id, not a name, so the two only mean anything together. Rewriting
    the class list from just this export's labels used to renumber every earlier
    file in the directory — export a folder of cats, then a folder of dogs, and
    the cats came back labelled dog with nothing to indicate it. Existing ids
    keep their positions and new labels are appended after them.
    """

    id = "yolo"
    title = "YOLO labels (directory)"
    file_filter = "Directory"
    writes_directory = True

    def export(self, request: ExportRequest) -> ExportReport:
        report = ExportReport(item_count=len(request.items))

        root = request.destination
        label_dir = root / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        classes_file = root / "classes.txt"

        existing = _read_classes(classes_file)
        labels = existing + [label for label in request.labels if label not in existing]
        class_ids = {label: i for i, label in enumerate(labels)}

        lossy: set[str] = set()
        for item, stem in zip(request.items, request.stems(), strict=True):
            lines: list[str] = []
            for annotation in item.annotations:
                if annotation.kind in (AnnotationKind.POLYGON, AnnotationKind.MASK):
                    lossy.add(annotation.kind.value)
                if annotation.kind is AnnotationKind.POINT:
                    lossy.add("point")
                cx, cy, w, h = self._box_of(annotation).normalised(item.image_size)
                lines.append(
                    f"{class_ids[self._label_of(annotation)]} "
                    f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}",
                )
                report.annotation_count += 1

            out = label_dir / f"{stem}.txt"
            out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            report.written.append(out)

        classes_file.write_text("\n".join(labels) + "\n", encoding="utf-8")
        report.written.append(classes_file)

        if existing:
            added = len(labels) - len(existing)
            report.warnings.append(
                f"classes.txt already listed {len(existing)} class(es); their ids were "
                f"kept and {added} new one(s) appended",
            )
        for kind in sorted(lossy):
            report.warnings.append(
                f"{kind} annotations were reduced to bounding boxes (YOLO detection format)",
            )

        if request.write_frames:
            frames, warnings = self._write_frames(request, root / "images")
            report.written.extend(frames)
            report.warnings.extend(warnings)

        return report


class CsvExporter(Exporter):
    """One row per annotation. For spreadsheets, QA review and quick greps."""

    id = "csv"
    title = "CSV (one row per annotation)"
    file_filter = "CSV Files (*.csv)"
    default_extension = ".csv"

    _COLUMNS = (
        "media", "frame", "kind", "label", "score", "source",
        "x1", "y1", "x2", "y2", "width", "height",
        "image_width", "image_height",
    )

    def export(self, request: ExportRequest) -> ExportReport:
        report = ExportReport(item_count=len(request.items))
        destination = request.destination
        if destination.suffix.lower() != ".csv":
            destination = destination.with_suffix(".csv")
        destination.parent.mkdir(parents=True, exist_ok=True)

        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(self._COLUMNS)
            for item in request.items:
                iw, ih = item.image_size.as_int()
                for annotation in item.annotations:
                    box = self._box_of(annotation)
                    writer.writerow([
                        item.media_path.name,
                        item.frame_index,
                        annotation.kind.value,
                        self._label_of(annotation),
                        "" if annotation.score is None else f"{annotation.score:.4f}",
                        annotation.source,
                        f"{box.x1:.2f}", f"{box.y1:.2f}", f"{box.x2:.2f}", f"{box.y2:.2f}",
                        f"{box.width:.2f}", f"{box.height:.2f}",
                        iw, ih,
                    ])
                    report.annotation_count += 1

        report.written.append(destination)
        return report


def builtin_exporters() -> list[Exporter]:
    return [CocoExporter(), YoloExporter(), CsvExporter()]


def _read_classes(path: Path) -> list[str]:
    """Class names already in a ``classes.txt``, in id order. Empty if absent."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def items_from_store(
    media_path: Path,
    image_size: Size,
    annotations: Iterable[Annotation],
) -> list[ExportItem]:
    """Group a store's annotations into one :class:`ExportItem` per frame."""
    by_frame: dict[int, list[Annotation]] = {}
    for annotation in annotations:
        by_frame.setdefault(annotation.frame.index, []).append(annotation)
    return [
        ExportItem(media_path, index, image_size, tuple(by_frame[index]))
        for index in sorted(by_frame)
    ]


def items_from_library(paths: Iterable[Path]) -> tuple[list[ExportItem], list[str]]:
    """Export items for every annotated file in a playlist, read from sidecars.

    One export can then cover a whole session's work. Doing it per file was not
    just tedious: the only way to assemble a dataset was to point several
    exports at one directory, which is exactly the case the formats handle worst.

    Unannotated files are skipped silently. A file whose dimensions cannot be
    established is skipped *loudly* — YOLO coordinates are normalised, so
    guessing a size would write plausible, wrong numbers.
    """
    items: list[ExportItem] = []
    warnings: list[str] = []
    for path in paths:
        annotations = load_annotations(path)
        if not annotations:
            continue
        size = load_image_size(path) or _probe_size(path)
        if size is None:
            warnings.append(f"{path.name}: cannot determine image size, skipped")
            continue
        items.extend(items_from_store(path, size, annotations))
    return items, warnings


def _probe_size(media_path: Path) -> Size | None:
    """Open the media purely to read its dimensions.

    The fallback for a sidecar written before sizes were recorded, or by hand.
    """
    from ..media.source import open_source

    try:
        source = open_source(media_path)
    except Exception as exc:  # noqa: BLE001 - an unreadable file is a skip, not a crash
        _log.warning("Cannot size %s: %s", media_path.name, exc)
        return None
    try:
        return source.info.size
    finally:
        source.close()
