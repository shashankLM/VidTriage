"""Single source of truth for a VidTriage session.

A ``Session`` owns *all* mutable data state — directories, classes, videos,
and the undo stack.  The UI reads from it via properties and mutates through
action methods; no other code should move files or write to the CSV log
directly.

Typical lifecycle::

    session = Session(input_dir, output_dir, classes)
    session.load()          # discover videos, reconcile log, scan folders

    dupes = session.find_duplicate_names()
    if dupes:
        ...                 # warn user, rename or abort

    session.classify(item, entry)   # mutates item.history, moves file, logs
    session.undo_last()             # reverses the most recent action

    session.add_class("3", "truck") # adds a new class, persists config
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .models import ERRORS_FOLDER, AppConfig, ClassEntry, VideoItem
from .config import save_config as _persist_config
from .io_ops import (
    discover_videos,
    move_to_class,
    move_to_errors,
    scan_output_subfolders,
    undo_move,
    setup_logger,
)


class Session:
    """Central data store for one classification session.

    Attributes:
        input_dir:  Directory containing source videos.
        output_dir: Root directory for classified output (sub-folders per class).
        classes:    Ordered list of active class definitions.

    Internal state (not for direct external access):
        _videos:     All known videos, keyed by ``str(original_path)``.
        _undo_order: Global undo stack — list of VideoItems in action order.
                     ``undo_last()`` pops from here, then pops the item's
                     ``history[-1]`` to reverse the action.
    """

    def __init__(self, input_dir: Path, output_dir: Path, classes: list[ClassEntry]) -> None:
        self.input_dir = input_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.classes = classes
        self._videos: dict[str, VideoItem] = {}
        self._undo_order: list[VideoItem] = []
        setup_logger(output_dir)

    # ── derived state ───────────────────────────────────────

    @property
    def class_map(self) -> dict[str, ClassEntry]:
        """Keyboard-key → ClassEntry lookup (max 9 entries)."""
        return {c.key: c for c in self.classes}

    @property
    def pending(self) -> list[VideoItem]:
        """Videos not yet classified (or undone back to pending)."""
        return sorted(
            (v for v in self._videos.values() if v.is_pending),
            key=lambda v: v.original_path.name.lower(),
        )

    @property
    def classified(self) -> list[VideoItem]:
        """Videos that have a current classification (including errors)."""
        return sorted(
            (v for v in self._videos.values() if not v.is_pending),
            key=lambda v: v.original_path.name.lower(),
            reverse=True,
        )

    @property
    def all_videos(self) -> list[VideoItem]:
        return list(self._videos.values())

    def get_video(self, key: str) -> VideoItem | None:
        """Look up a video by its ``str(original_path)`` key."""
        return self._videos.get(key)

    def destination_of(self, item: VideoItem) -> Path | None:
        return item.destination_path(self.output_dir)

    def playback_path_of(self, item: VideoItem) -> Path:
        return item.playback_path(self.output_dir)

    # ── lifecycle ───────────────────────────────────────────

    def load(self) -> None:
        """Populate session state from disk.

        1. Discover video files in ``input_dir``.
        2. Scan ``output_dir`` sub-folders for classified videos.
        3. Prune pending items whose source file no longer exists
           (already moved to an output folder).

        Safe to call more than once (clears previous state first).
        """
        self._videos.clear()
        self._undo_order.clear()

        for path in discover_videos(self.input_dir):
            self._videos[str(path)] = VideoItem(original_path=path)

        self._scan_output_folders()

        gone = [k for k, v in self._videos.items()
                if v.is_pending and not v.original_path.exists()]
        for k in gone:
            del self._videos[k]

    def find_duplicate_names(self) -> dict[str, list[Path]]:
        """Videos that share a filename — potential move collisions.

        Returns a dict mapping ``filename → [path, …]`` only for names
        that appear more than once.  Call after ``load()`` and before
        any classify actions.
        """
        by_name: defaultdict[str, list[Path]] = defaultdict(list)
        for item in self._videos.values():
            by_name[item.original_path.name].append(item.original_path)
        return {name: paths for name, paths in by_name.items() if len(paths) > 1}

    def _scan_output_folders(self) -> None:
        """Pick up classified videos from output sub-folders.

        Also auto-creates ClassEntry objects for unknown folder names
        (using the next free key 1-9).
        """
        tracked_dests = {
            str(self.destination_of(item))
            for item in self._videos.values()
            if not item.is_pending
        }

        classes_changed = False
        for folder_name, videos in scan_output_subfolders(self.output_dir):
            is_error = folder_name == ERRORS_FOLDER

            if not is_error and not any(c.name == folder_name for c in self.classes):
                if len(self.classes) >= 9:
                    continue
                used_keys = {c.key for c in self.classes}
                new_key = next(
                    (str(i) for i in range(1, 10) if str(i) not in used_keys),
                    None,
                )
                if new_key is None:
                    continue
                self.classes.append(ClassEntry(key=new_key, name=folder_name))
                self.classes.sort(key=lambda c: int(c.key))
                classes_changed = True

            for video_path in videos:
                if str(video_path) in tracked_dests:
                    continue
                item = VideoItem(original_path=video_path)
                item.history.append(ERRORS_FOLDER if is_error else folder_name)
                self._videos[str(video_path)] = item

        if classes_changed:
            self.save_config()

    # ── actions ─────────────────────────────────────────────

    def classify(self, item: VideoItem, class_entry: ClassEntry) -> None:
        """Classify (or reclassify) a video into *class_entry*.

        Works for both pending and already-classified items — the file is
        moved from its current location to ``output_dir/<class_name>/``.

        Appends ``class_entry.name`` to ``item.history``.
        """
        source = self.playback_path_of(item)
        if not source.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        move_to_class(source, self.output_dir, class_entry)
        item.history.append(class_entry.name)
        self._undo_order.append(item)

    def mark_error(self, item: VideoItem) -> None:
        """Move a video to the ``_errors/`` folder.

        Appends ``"_errors"`` to ``item.history``.
        """
        source = self.playback_path_of(item)
        if not source.exists():
            raise FileNotFoundError(f"Source not found: {source}")

        move_to_errors(source, self.output_dir)
        item.history.append(ERRORS_FOLDER)
        self._undo_order.append(item)

    def undo_last(self) -> VideoItem | None:
        """Reverse the most recent classify / mark_error action.

        Pops the last item from ``_undo_order``, pops its ``history[-1]``,
        and moves the file:

        * If the video is now pending → file returns to ``original_path``.
        * If the video falls back to a previous class → file moves to that
          class folder.

        Returns the affected ``VideoItem``, or ``None`` if nothing to undo.
        """
        if not self._undo_order:
            return None

        item = self._undo_order.pop()
        if not item.history:
            return None

        current_dest = self.destination_of(item)
        if not current_dest or not current_dest.exists():
            return None

        item.history.pop()

        if item.is_pending:
            undo_move(current_dest, item.original_path)
        else:
            prev_name = item.class_name
            prev_entry = next(
                (c for c in self.classes if c.name == prev_name), None,
            )
            if prev_entry:
                move_to_class(current_dest, self.output_dir, prev_entry)
            else:
                undo_move(current_dest, item.original_path)
                item.history.clear()

        return item

    # ── class management ────────────────────────────────────

    def add_class(self, key: str, name: str) -> ClassEntry:
        """Create a new class at *key*, sort by key, and persist config."""
        entry = ClassEntry(key=key, name=name)
        self.classes.append(entry)
        self.classes.sort(key=lambda c: int(c.key))
        self.save_config()
        return entry

    def set_classes(self, classes: list[ClassEntry]) -> None:
        """Replace the full class list and persist config."""
        self.classes = classes
        self.save_config()

    # ── persistence ─────────────────────────────────────────

    def save_config(self) -> None:
        """Write current dirs + classes to ``~/.vidtriage/config.json``."""
        _persist_config(AppConfig(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            classes=self.classes,
        ))
