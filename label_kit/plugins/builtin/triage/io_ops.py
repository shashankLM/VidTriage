"""Filesystem operations for triage: finding media, and copying it out.

Videos and still images are both triaged, and neither the session nor the log
distinguishes them — an image is a one-frame source, so a decision about a
``.png`` is the same kind of record as a decision about an ``.mp4``.

**Nothing here moves a file.** Classification is recorded in a log (see
:mod:`ledger`), so the input tree is read-only for the whole session and the
only write path is :func:`copy_media`, which materialises a snapshot into a
directory the user names.

**Collisions are refused, never overwritten.** If two videos ever resolve to the
same destination, the earlier implementation destroyed one of them silently.
Here that raises :class:`~label_kit.core.errors.FileOperationError` naming both
paths. Refusing an operation is recoverable; deleting footage is not.

**Annotation sidecars travel with the video**, or a snapshot would arrive with
every annotation orphaned.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ....app.library import discover_media
from ....core.errors import FileOperationError
from ....core.logging import get_logger
from ....persistence.sidecar import sidecar_path_for

__all__ = [
    "copy_media",
    "scan_output_subfolders",
]

_log = get_logger(__name__)


def scan_output_subfolders(output_dir: Path) -> list[tuple[str, list[Path]]]:
    """``[(folder_name, [media…]), …]`` for each non-empty class folder.

    Only used to import an old move-based session once; see
    :meth:`~label_kit.plugins.builtin.triage.session.Session.import_legacy_output`.
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
        found = discover_media(subdir)
        if found:
            results.append((subdir.name, found))
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
    """Best-effort: keep annotations attached to the file that owns them."""
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
