"""Filesystem operations for triage: finding videos, and copying them out.

**Nothing here moves a file.** Classification is recorded in a log (see
:mod:`ledger`), so the input tree is read-only for the whole session and the
only write path is :func:`copy_media`, which materialises a snapshot into a
directory the user names.

**Collisions are refused, never overwritten.** If two videos ever resolve to the
same destination, the earlier implementation destroyed one of them silently.
Here that raises :class:`~vidtriage.core.errors.FileOperationError` naming both
paths. Refusing an operation is recoverable; deleting footage is not.

**Annotation sidecars travel with the video**, or a snapshot would arrive with
every annotation orphaned.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ....core.errors import FileOperationError
from ....core.logging import get_logger
from ....media.source import VIDEO_EXTENSIONS
from ....persistence.sidecar import sidecar_path_for

__all__ = [
    "copy_media",
    "discover_videos",
    "scan_output_subfolders",
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
    """``[(folder_name, [videos…]), …]`` for each non-empty class folder.

    Only used to import an old move-based session once; see
    :meth:`~vidtriage.plugins.builtin.triage.session.Session.import_legacy_output`.
    """
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


def copy_media(source: Path, destination: Path, *, link: bool = False) -> Path:
    """Copy a video and its annotation sidecar to ``destination``.

    Args:
        link: Try a hardlink first. A corpus of video is large enough that
            copying it is the slow part of a snapshot, and a hardlink is free.
            Falls back to a real copy when the destination is on another
            filesystem or the platform refuses — so passing this is always safe,
            never a different result, only a faster one.

    Raises:
        FileOperationError: the source is gone, the destination is taken, or the
            copy failed.
    """
    if not source.exists():
        raise FileOperationError(f"Source no longer exists: {source}")
    if destination.exists():
        raise FileOperationError(
            f"Refusing to overwrite an existing file.\n\n"
            f"Destination: {destination}\nSource: {source}",
        )

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _place(source, destination, link=link)
    except OSError as exc:
        raise FileOperationError(f"Could not copy {source.name}: {exc}") from exc

    _copy_sidecar(source, destination, link=link)
    return destination


def _place(source: Path, destination: Path, *, link: bool) -> None:
    if link:
        try:
            os.link(os.fspath(source), os.fspath(destination))
        except OSError as exc:
            # Cross-device, a filesystem without hardlinks, or a hardlink limit.
            # None of those are the user's problem — copy and carry on.
            _log.debug("Hardlink failed for %s (%s); copying", source.name, exc)
        else:
            return
    shutil.copy2(os.fspath(source), os.fspath(destination))


def _copy_sidecar(source: Path, destination: Path, *, link: bool) -> None:
    """Best-effort: keep annotations attached to the video that owns them."""
    old = sidecar_path_for(source)
    if not old.exists():
        return
    new = sidecar_path_for(destination)
    try:
        _place(old, new, link=link)
    except OSError as exc:
        # The video is already in place; failing the whole snapshot over a
        # sidecar would be a worse outcome than a warning the user can act on.
        _log.warning("Could not copy sidecar %s: %s", old.name, exc)
