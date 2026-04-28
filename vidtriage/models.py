"""Data models for VidTriage.

ClassEntry  – a single keyboard-bound class (key "1"-"9", human name).
VideoItem   – one video with its full annotation history.
AppConfig   – serialisable configuration (dirs + classes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ERRORS_FOLDER = "_errors"


@dataclass
class ClassEntry:
    """A classification category mapped to a keyboard shortcut.

    Attributes:
        key:  Single digit "1"-"9" used as the keyboard shortcut.
        name: Human-readable class label (also the output sub-folder name).
    """

    key: str
    name: str


@dataclass
class VideoItem:
    """A single video and its classification journey.

    The *history* list is the core state.  Each element is either a class
    name (``str``) meaning "classified as …", or ``None`` meaning "returned
    to pending (undo)".

    Examples::

        []                        → never touched, pending
        ["cat"]                   → classified as cat
        ["cat", "dog"]            → was cat, reclassified to dog
        ["cat", None]             → was cat, then undone → pending
        ["cat", None, "dog"]      → cat → undo → dog

    ``"_errors"`` is used as the class name for error-flagged videos.

    The on-disk destination is derived by ``Session.destination_of()`` as
    ``output_dir / class_name / original_path.name`` — no stored path needed
    because filenames are guaranteed unique at launch.

    Attributes:
        original_path:    Where the file lived in the input directory.
        history:          Chronological list of annotation events.
    """

    original_path: Path
    history: list[str | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.original_path = self.original_path.resolve()

    @property
    def class_name(self) -> str | None:
        """Current class label, or ``None`` if pending."""
        if not self.history or self.history[-1] is None:
            return None
        return self.history[-1]

    @property
    def is_pending(self) -> bool:
        """``True`` when the video has not been classified (or was undone)."""
        return self.class_name is None

    @property
    def is_error(self) -> bool:
        """``True`` when the video is in the ``_errors`` folder."""
        return self.class_name == ERRORS_FOLDER

    def destination_path(self, output_dir: Path) -> Path | None:
        """Derived on-disk location: ``output_dir / class_name / filename``."""
        if self.is_pending:
            return None
        return output_dir / self.class_name / self.original_path.name

    def playback_path(self, output_dir: Path) -> Path:
        """Best path for playback — destination if it exists, else original."""
        dest = self.destination_path(output_dir)
        if dest and dest.exists():
            return dest
        return self.original_path


@dataclass
class AppConfig:
    """Persisted user configuration (directories and class list).

    Serialised to ``~/.vidtriage/config.json`` by :func:`config.save_config`.
    """

    input_dir: Path | None = None
    output_dir: Path | None = None
    classes: list[ClassEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.input_dir:
            self.input_dir = self.input_dir.resolve()
        if self.output_dir:
            self.output_dir = self.output_dir.resolve()
