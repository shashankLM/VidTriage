"""Triage session state — the single owner of every classification decision.

The UI reads through properties and mutates through action methods. Nothing else
moves a file or edits a history list, which is what keeps undo trustworthy: the
undo stack and the filesystem cannot drift apart if there is exactly one path
that changes either.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ....core.errors import FileOperationError
from ....core.logging import get_logger
from .config import save_session
from .io_ops import (
    discover_videos,
    move_to_class,
    move_to_errors,
    scan_output_subfolders,
    undo_move,
)
from .models import ERRORS_FOLDER, MAX_CLASSES, ClassEntry, TriageConfig, VideoItem

__all__ = ["Session"]

_log = get_logger(__name__)


class Session:
    """All videos, classes and undo state for one input/output directory pair."""

    def __init__(self, input_dir: Path, output_dir: Path, classes: list[ClassEntry]) -> None:
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.classes = list(classes)
        self._videos: dict[str, VideoItem] = {}
        self._undo_order: list[VideoItem] = []

    # ── derived state ───────────────────────────────────────────────────

    @property
    def class_map(self) -> dict[str, ClassEntry]:
        """Key → class, for the 1–9 shortcuts."""
        return {c.key: c for c in self.classes}

    @property
    def pending(self) -> list[VideoItem]:
        return sorted(
            (v for v in self._videos.values() if v.is_pending),
            key=lambda v: v.name.lower(),
        )

    @property
    def classified(self) -> list[VideoItem]:
        return sorted(
            (v for v in self._videos.values() if not v.is_pending),
            key=lambda v: v.name.lower(),
        )

    @property
    def all_videos(self) -> list[VideoItem]:
        return list(self._videos.values())

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_order)

    def playback_path_of(self, item: VideoItem) -> Path:
        return item.playback_path(self.output_dir)

    def destination_of(self, item: VideoItem) -> Path | None:
        return item.destination_path(self.output_dir)

    def find_by_path(self, path: Path) -> VideoItem | None:
        """The item whose file currently lives at ``path``."""
        resolved = Path(path).resolve()
        for item in self._videos.values():
            if item.original_path == resolved or self.playback_path_of(item) == resolved:
                return item
        return None

    # ── lifecycle ───────────────────────────────────────────────────────

    def load(self) -> None:
        """Rebuild state from disk. Safe to call repeatedly."""
        self._videos.clear()
        self._undo_order.clear()

        for path in discover_videos(self.input_dir):
            self._videos[str(path)] = VideoItem(original_path=path)

        self._scan_output_folders()

        # A pending item whose source file has vanished was moved by something
        # other than this session; drop it rather than show a broken entry.
        for key in [
            k for k, v in self._videos.items()
            if v.is_pending and not v.original_path.exists()
        ]:
            del self._videos[key]

        _log.info(
            "Session loaded: %d pending, %d classified",
            len(self.pending), len(self.classified),
        )

    def find_duplicate_names(self) -> dict[str, list[Path]]:
        """Filenames appearing more than once — these would collide on move.

        Destinations are derived from the filename, so duplicates must be caught
        before any classification happens.
        """
        by_name: defaultdict[str, list[Path]] = defaultdict(list)
        for item in self._videos.values():
            by_name[item.name].append(item.original_path)
        return {name: paths for name, paths in by_name.items() if len(paths) > 1}

    def _scan_output_folders(self) -> None:
        """Pick up already-classified videos, inventing classes for new folders."""
        tracked = {
            str(self.destination_of(item))
            for item in self._videos.values() if not item.is_pending
        }
        classes_changed = False

        for folder_name, videos in scan_output_subfolders(self.output_dir):
            is_error = folder_name == ERRORS_FOLDER

            if not is_error and not any(c.name == folder_name for c in self.classes):
                entry = self._make_class_for(folder_name)
                if entry is None:
                    _log.warning(
                        "Ignoring folder %r: all %d class keys are taken",
                        folder_name, MAX_CLASSES,
                    )
                    continue
                classes_changed = True

            for video_path in videos:
                if str(video_path) in tracked:
                    continue
                item = VideoItem(original_path=video_path)
                item.history.append(ERRORS_FOLDER if is_error else folder_name)
                self._videos[str(video_path)] = item

        if classes_changed:
            self.save_config()

    def _make_class_for(self, name: str) -> ClassEntry | None:
        used = {c.key for c in self.classes}
        key = next((str(i) for i in range(1, MAX_CLASSES + 1) if str(i) not in used), None)
        if key is None:
            return None
        entry = ClassEntry(key=key, name=name)
        self.classes.append(entry)
        self.classes.sort(key=lambda c: c.key)
        return entry

    # ── actions ─────────────────────────────────────────────────────────

    def classify(self, item: VideoItem, class_entry: ClassEntry) -> Path:
        """Move ``item`` into ``class_entry``'s folder and record the decision.

        Works for pending and already-classified videos alike — reclassifying
        moves the file from its current class folder to the new one.

        Raises:
            FileOperationError: the source is missing, or the destination is
                occupied by a different file.
        """
        source = self.playback_path_of(item)
        if not source.exists():
            raise FileOperationError(f"Source not found: {source}")

        destination = move_to_class(source, self.output_dir, class_entry)
        item.history.append(class_entry.name)
        self._undo_order.append(item)
        return destination

    def mark_error(self, item: VideoItem) -> Path:
        source = self.playback_path_of(item)
        if not source.exists():
            raise FileOperationError(f"Source not found: {source}")

        destination = move_to_errors(source, self.output_dir)
        item.history.append(ERRORS_FOLDER)
        self._undo_order.append(item)
        return destination

    def undo_last(self) -> VideoItem | None:
        """Reverse the most recent classify or mark-error.

        Returns the affected item, or ``None`` when there was nothing to undo.
        Unlike the previous implementation this reports *why* an undo could not
        proceed instead of returning ``None`` from three indistinguishable
        failure paths, and it leaves the undo stack intact when the filesystem
        no longer matches so a retry is possible after the user fixes things.
        """
        if not self._undo_order:
            return None

        item = self._undo_order[-1]
        if not item.history:
            self._undo_order.pop()
            return None

        current = self.destination_of(item)
        if current is None or not current.exists():
            self._undo_order.pop()
            _log.warning(
                "Cannot undo %s: expected it at %s but it is not there",
                item.name, current,
            )
            raise FileOperationError(
                f"Cannot undo {item.name} — the file is no longer at\n{current}",
            )

        self._undo_order.pop()
        item.history.pop()

        if item.is_pending:
            undo_move(current, item.original_path)
            return item

        previous = next((c for c in self.classes if c.name == item.class_name), None)
        if previous is not None:
            move_to_class(current, self.output_dir, previous)
        else:
            # The earlier class no longer exists; returning the file home beats
            # leaving it in a folder nothing references.
            _log.warning(
                "Class %r is gone; returning %s to its input directory",
                item.class_name, item.name,
            )
            undo_move(current, item.original_path)
            item.history.clear()
        return item

    # ── class management ────────────────────────────────────────────────

    def add_class(self, key: str, name: str) -> ClassEntry:
        entry = ClassEntry(key=key, name=name)
        self.classes.append(entry)
        self.classes.sort(key=lambda c: c.key)
        self.save_config()
        return entry

    def set_classes(self, classes: list[ClassEntry]) -> None:
        self.classes = list(classes)
        self.save_config()

    # ── persistence ─────────────────────────────────────────────────────

    def to_config(self) -> TriageConfig:
        return TriageConfig(
            input_dir=self.input_dir, output_dir=self.output_dir, classes=self.classes,
        )

    def save_config(self) -> None:
        save_session(self.to_config())
