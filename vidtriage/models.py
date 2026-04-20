from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class FileStatus(Enum):
    PENDING = "pending"
    CLASSIFIED = "classified"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class ClassEntry:
    key: str
    name: str


@dataclass
class VideoItem:
    original_path: Path
    status: FileStatus = FileStatus.PENDING
    class_entry: ClassEntry | None = None
    destination_path: Path | None = None


@dataclass
class AppConfig:
    input_dir: Path | None = None
    output_dir: Path | None = None
    classes: list[ClassEntry] = field(default_factory=list)
