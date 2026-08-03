"""The media playlist.

A flat, ordered list of media files with a cursor. Whoever populates it —
the triage plugin from its session, a plain "Open Folder", or a drag-and-drop —
the rest of the app navigates it the same way.

Decoupling "what is open" from "who decided to open it" is what lets triage be a
plugin: it drives the library, but nothing about playback, annotation or export
depends on triage being installed.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from ..core.logging import get_logger
from ..media.source import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

__all__ = ["MEDIA_EXTENSIONS", "MediaLibrary", "discover_media"]

_log = get_logger(__name__)

MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS


def discover_media(directory: Path, recursive: bool = False) -> list[Path]:
    """Media files in ``directory``, sorted case-insensitively by name.

    Videos and stills alike — an image is a one-frame source (see
    :class:`~label_kit.media.source.ImageFileSource`), so nothing downstream
    needs to know which it got.

    A missing or unreadable directory yields an empty list rather than raising:
    it is routine to point at a folder that does not exist yet, and callers
    scanning several directories should not lose all of them to one bad mount.
    """
    if not directory.is_dir():
        return []
    try:
        walker = list(directory.rglob("*") if recursive else directory.iterdir())
    except OSError as exc:
        _log.warning("Cannot list %s: %s", directory, exc)
        return []
    found = [
        path for path in walker
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
    ]
    return sorted(found, key=lambda p: p.name.lower())


class MediaLibrary(QObject):
    """An ordered list of media paths plus the currently open one."""

    items_changed = Signal()
    current_changed = Signal(object)  # Path | None

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._items: list[Path] = []
        self._index = -1

    # ── contents ────────────────────────────────────────────────────────

    @property
    def items(self) -> list[Path]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def set_items(self, paths: list[Path], keep_current: bool = True) -> None:
        """Replace the list, preserving the cursor position where possible."""
        previous = self.current
        self._items = [Path(p) for p in paths]

        if keep_current and previous is not None and previous in self._items:
            self._index = self._items.index(previous)
        else:
            self._index = 0 if self._items else -1

        self.items_changed.emit()
        if self.current != previous:
            self.current_changed.emit(self.current)

    def add(self, path: Path) -> None:
        path = Path(path)
        if path not in self._items:
            self._items.append(path)
            self.items_changed.emit()

    def clear(self) -> None:
        self._items.clear()
        self._index = -1
        self.items_changed.emit()
        self.current_changed.emit(None)

    # ── cursor ──────────────────────────────────────────────────────────

    @property
    def current(self) -> Path | None:
        if 0 <= self._index < len(self._items):
            return self._items[self._index]
        return None

    @property
    def index(self) -> int:
        return self._index

    @property
    def has_next(self) -> bool:
        return self._index < len(self._items) - 1

    @property
    def has_previous(self) -> bool:
        return self._index > 0

    def set_index(self, index: int) -> bool:
        if not (0 <= index < len(self._items)) or index == self._index:
            return False
        self._index = index
        self.current_changed.emit(self.current)
        return True

    def open(self, path: Path) -> bool:
        """Select ``path``, appending it to the list if it is not already there."""
        path = Path(path)
        if path not in self._items:
            self._items.append(path)
            self.items_changed.emit()
        return self.set_index(self._items.index(path))

    def next(self) -> bool:
        return self.set_index(self._index + 1)

    def previous(self) -> bool:
        return self.set_index(self._index - 1)
