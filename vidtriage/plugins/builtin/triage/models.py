"""Triage data model: classes, videos, and a video's classification history."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ERRORS_FOLDER", "MAX_CLASSES", "ClassEntry", "TriageConfig", "VideoItem"]

ERRORS_FOLDER = "_errors"
#: Classes are bound to keys 1–9, so nine is the ceiling.
MAX_CLASSES = 9


@dataclass(frozen=True)
class ClassEntry:
    """A classification category bound to a number key.

    Attributes:
        key:  Single digit ``"1"``–``"9"``.
        name: Label, and also the output sub-folder name.
    """

    key: str
    name: str


@dataclass
class VideoItem:
    """One video and its full classification history.

    ``history`` is the state. Each entry is a class name, or ``None`` meaning
    "returned to pending". Keeping the whole chain rather than a single current
    label is what makes undo able to step back through reclassifications::

        []                    never touched → pending
        ["cat"]               classified as cat
        ["cat", "dog"]        reclassified cat → dog
        ["cat", None]         classified, then undone → pending
        ["cat", None, "dog"]  cat → undo → dog

    The file itself never moves. ``original_path`` is where it is, for the whole
    life of the session; the class is a fact recorded in the log, and
    :meth:`snapshot_path` is only consulted when the user asks for a copy.
    """

    original_path: Path
    history: list[str | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.original_path = self.original_path.resolve()

    @property
    def name(self) -> str:
        return self.original_path.name

    @property
    def class_name(self) -> str | None:
        """Current label, or ``None`` when pending."""
        if not self.history or self.history[-1] is None:
            return None
        return self.history[-1]

    @property
    def is_pending(self) -> bool:
        return self.class_name is None

    @property
    def is_error(self) -> bool:
        return self.class_name == ERRORS_FOLDER

    def snapshot_path(self, output_dir: Path) -> Path | None:
        """Where a snapshot would place this video. ``None`` while pending."""
        if self.is_pending:
            return None
        return output_dir / str(self.class_name) / self.original_path.name


@dataclass
class TriageConfig:
    """A saved triage session: the two directories and the class list."""

    input_dir: Path | None = None
    output_dir: Path | None = None
    classes: list[ClassEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.input_dir:
            self.input_dir = Path(self.input_dir).resolve()
        if self.output_dir:
            self.output_dir = Path(self.output_dir).resolve()

    @property
    def is_complete(self) -> bool:
        return bool(self.input_dir and self.output_dir and self.classes)
