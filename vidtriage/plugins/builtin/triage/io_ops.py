"""File operations for triage.

Two behaviours here are deliberate departures from the previous implementation,
both because they concern the user's actual video files:

**Collisions are refused, never overwritten.** ``shutil.move`` onto an existing
path replaces it. If two different videos ever resolve to the same destination,
the old code destroyed one of them silently. Here that raises
:class:`~vidtriage.core.errors.FileOperationError` naming both paths, and the UI
reports it. Refusing an operation is recoverable; deleting footage is not.

**Annotation sidecars travel with the video.** Triage moves files between class
folders; a sidecar left behind would orphan every annotation on that clip.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ....core.errors import FileOperationError
from ....core.logging import get_logger
from ....media.source import VIDEO_EXTENSIONS
from ....persistence.sidecar import sidecar_path_for
from .models import ERRORS_FOLDER, ClassEntry

__all__ = [
    "discover_videos",
    "move_media",
    "move_to_class",
    "move_to_errors",
    "scan_output_subfolders",
    "undo_move",
]

_log = get_logger(__name__)


def discover_videos(directory: Path) -> list[Path]:
    """Video files directly inside ``directory``, sorted by name.

    A missing or unreadable directory yields an empty list rather than raising:
    it is routine for an output folder not to exist yet.
    """
    if not directory.is_dir():
        return []
    try:
        entries = sorted(directory.iterdir())
    except OSError as exc:
        _log.warning("Cannot list %s: %s", directory, exc)
        return []
    return [
        path for path in entries
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]


def scan_output_subfolders(output_dir: Path) -> list[tuple[str, list[Path]]]:
    """``[(folder_name, [videos…]), …]`` for each non-empty class folder."""
    results: list[tuple[str, list[Path]]] = []
    if not output_dir.is_dir():
        return results

    try:
        entries = sorted(output_dir.iterdir())
    except OSError as exc:
        _log.warning("Cannot list %s: %s", output_dir, exc)
        return results

    for subdir in entries:
        # Skip dotfolders and bare-numeric names, which are almost always
        # tooling output rather than a class the user created.
        if not subdir.is_dir() or subdir.name.startswith(".") or subdir.name.isdigit():
            continue
        videos = discover_videos(subdir)
        if videos:
            results.append((subdir.name, videos))
    return results


def move_media(source: Path, destination: Path) -> Path:
    """Move a video and its annotation sidecar. Refuses to clobber.

    Raises:
        FileOperationError: if the source is gone, the destination is taken by a
            different file, or the move fails.
    """
    if not source.exists():
        raise FileOperationError(f"Source no longer exists: {source}")
    if source.resolve() == destination.resolve():
        return destination
    if destination.exists():
        raise FileOperationError(
            f"Refusing to overwrite an existing file.\n\n"
            f"Destination: {destination}\nSource: {source}\n\n"
            f"Rename or remove the destination and try again.",
        )

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(os.fspath(source), os.fspath(destination))
    except OSError as exc:
        raise FileOperationError(f"Could not move {source.name}: {exc}") from exc

    _move_sidecar(source, destination)
    return destination


def _move_sidecar(source: Path, destination: Path) -> None:
    """Best-effort: keep annotations attached to the video that owns them."""
    old = sidecar_path_for(source)
    if not old.exists():
        return
    new = sidecar_path_for(destination)
    try:
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(os.fspath(old), os.fspath(new))
        _log.debug("Moved sidecar %s -> %s", old.name, new)
    except OSError as exc:
        # The video already moved; failing here would leave inconsistent state
        # for something the user can fix by hand.
        _log.warning("Could not move sidecar %s: %s", old.name, exc)


def move_to_class(source: Path, output_dir: Path, class_entry: ClassEntry) -> Path:
    destination = output_dir / class_entry.name / source.name
    result = move_media(source, destination)
    _log.info("CLASSIFY [%s:%s] %s -> %s", class_entry.key, class_entry.name, source, result)
    return result


def move_to_errors(source: Path, output_dir: Path) -> Path:
    destination = output_dir / ERRORS_FOLDER / source.name
    result = move_media(source, destination)
    _log.info("ERROR %s -> %s", source, result)
    return result


def undo_move(destination: Path, original_path: Path) -> Path:
    result = move_media(destination, original_path)
    _log.info("UNDO %s -> %s", destination, result)
    return result
