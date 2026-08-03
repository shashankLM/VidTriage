"""Annotation sidecar files.

Annotations live in ``<file>.labelkit.json``, next to the media they describe.
Two consequences, both deliberate:

* A snapshot copies media into class folders. A sidecar that sits beside the
  file gets copied with it; a central index keyed by path would not.
* The format is plain, readable JSON — diffable in git, greppable, and fixable
  by hand when something goes wrong.

Writes are atomic (temp file in the same directory, then ``os.replace``), so a
crash or a full disk mid-save cannot leave a truncated file where a good one used
to be. Losing an hour of labelling to a partial write is not an acceptable
failure mode.

**Legacy sidecars are still read.** The project used to be called VidTriage and
wrote ``<file>.vidtriage.json``; those files are next to real footage on real
disks, so :func:`load_annotations` falls back to the old name when no new one
exists. Saving always writes the new name and leaves the old file untouched —
annotation data is the user's, not ours to delete. Once you are satisfied
everything has migrated, ``find . -name '*.vidtriage.json' -delete`` is safe.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ..core.annotations import Annotation, AnnotationStore
from ..core.errors import PersistenceError
from ..core.frames import source_id_for
from ..core.geometry import Size
from ..core.logging import get_logger

__all__ = [
    "LEGACY_SIDECAR_SUFFIX",
    "SIDECAR_SUFFIX",
    "delete_sidecar",
    "existing_sidecar_for",
    "load_annotations",
    "load_image_size",
    "load_into_store",
    "save_annotations",
    "save_store",
    "sidecar_path_for",
    "write_json_atomic",
]

_log = get_logger(__name__)

SIDECAR_SUFFIX = ".labelkit.json"
#: What the project wrote when it was called VidTriage. Read, never written.
LEGACY_SIDECAR_SUFFIX = ".vidtriage.json"
_FORMAT_VERSION = 1


def sidecar_path_for(media_path: Path) -> Path:
    """``/media/clip.mp4`` → ``/media/clip.mp4.labelkit.json``.

    The full name is kept, extension included, so ``clip.mp4`` and ``clip.avi``
    in one folder do not fight over the same sidecar.
    """
    return media_path.with_name(media_path.name + SIDECAR_SUFFIX)


def existing_sidecar_for(media_path: Path) -> Path | None:
    """The sidecar to read: the current name, else the legacy one, else ``None``.

    Only reads consult the legacy name. Writes always go to the current one, so
    a file migrates the first time its annotations are saved.
    """
    current = sidecar_path_for(media_path)
    if current.exists():
        return current
    legacy = media_path.with_name(media_path.name + LEGACY_SIDECAR_SUFFIX)
    return legacy if legacy.exists() else None


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON so the destination is either the old file or the new one.

    The temp file goes in the *same directory* as the target: ``os.replace`` is
    only atomic within a filesystem, and ``/tmp`` is frequently a different one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = None
    temp_path: Path | None = None
    try:
        fd, temp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp",
        )
        temp_path = Path(temp_name)
        handle = os.fdopen(fd, "w", encoding="utf-8")
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        os.replace(temp_path, path)
        temp_path = None
    except OSError as exc:
        raise PersistenceError(f"Could not write {path}: {exc}") from exc
    finally:
        if handle is not None:
            handle.close()
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def save_annotations(
    media_path: Path,
    annotations: list[Annotation],
    image_size: Size | None = None,
    extra: dict[str, Any] | None = None,
) -> Path | None:
    """Write a sidecar for ``media_path``. Returns its path, or ``None``.

    An empty annotation list deletes any existing sidecar rather than leaving an
    empty one, so the directory does not accumulate meaningless files.
    """
    path = sidecar_path_for(media_path)
    if not annotations:
        delete_sidecar(media_path)
        return None

    payload: dict[str, Any] = {
        "version": _FORMAT_VERSION,
        "media": media_path.name,
        "annotations": [a.to_dict() for a in annotations],
    }
    if image_size is not None:
        payload["image_size"] = [image_size.width, image_size.height]
    if extra:
        payload["extra"] = extra

    write_json_atomic(path, payload)
    _log.debug("Saved %d annotation(s) to %s", len(annotations), path.name)
    return path


def _read_payload(media_path: Path) -> tuple[Path, dict[str, Any]] | None:
    """``(sidecar path, parsed object)``, or ``None`` if there is nothing usable.

    A malformed sidecar must not stop the user opening the video, so every
    failure here is logged and reported as "no sidecar".
    """
    path = existing_sidecar_for(media_path)
    if path is None:
        return None
    if path.name.endswith(LEGACY_SIDECAR_SUFFIX):
        _log.info("Reading legacy sidecar %s; it will migrate on next save", path.name)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _log.warning("Ignoring unreadable sidecar %s: %s", path.name, exc)
        return None
    if not isinstance(data, dict):
        _log.warning("Ignoring malformed sidecar %s: not an object", path.name)
        return None
    return path, data


def load_annotations(media_path: Path) -> list[Annotation]:
    """Read a sidecar. Missing or corrupt files yield an empty list.

    Individual bad entries are skipped rather than discarding the whole file.
    """
    found = _read_payload(media_path)
    if found is None:
        return []
    path, data = found

    source_id = source_id_for(media_path)
    annotations: list[Annotation] = []
    for raw in data.get("annotations", []):
        try:
            annotations.append(Annotation.from_dict(raw, source_id))
        except (KeyError, TypeError, ValueError) as exc:
            _log.warning("Skipping bad annotation in %s: %s", path.name, exc)
    return annotations


def load_image_size(media_path: Path) -> Size | None:
    """The media dimensions recorded in the sidecar, if it has them.

    Lets an export cover a whole playlist without opening and decoding every
    file just to learn how big it is — which matters because YOLO coordinates
    are normalised, so the size is not optional.
    """
    found = _read_payload(media_path)
    if found is None:
        return None
    raw = found[1].get("image_size")
    if not (isinstance(raw, list) and len(raw) == 2):
        return None
    try:
        width, height = float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None
    return Size(width, height) if width > 0 and height > 0 else None


def load_into_store(store: AnnotationStore, media_path: Path) -> int:
    """Replace ``store``'s contents with the sidecar's. Returns the count."""
    annotations = load_annotations(media_path)
    store.reset(annotations, source_id=source_id_for(media_path))
    return len(annotations)


def save_store(store: AnnotationStore, media_path: Path, image_size: Size | None = None) -> Path | None:
    """Persist a store and mark it clean."""
    path = save_annotations(media_path, store.all(), image_size)
    store.mark_clean()
    return path


def delete_sidecar(media_path: Path) -> bool:
    """Remove the sidecar, under either name. ``True`` if anything was deleted.

    Deleting *both* is load-bearing. Clearing the last annotation deletes the
    sidecar, and if the legacy file survived that, :func:`load_annotations`
    would fall back to it and resurrect the annotations the user just removed.
    """
    deleted = False
    for suffix in (SIDECAR_SUFFIX, LEGACY_SIDECAR_SUFFIX):
        path = media_path.with_name(media_path.name + suffix)
        if not path.exists():
            continue
        try:
            path.unlink()
            deleted = True
        except OSError as exc:
            _log.warning("Could not delete %s: %s", path.name, exc)
    return deleted
