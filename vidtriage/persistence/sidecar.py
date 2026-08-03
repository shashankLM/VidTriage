"""Annotation sidecar files.

Annotations live in ``<video>.vidtriage.json``, next to the media they describe.
Two consequences, both deliberate:

* Triage **moves** videos between class folders. A sidecar that travels with the
  file keeps its annotations attached; a central index keyed by path would break
  on every classify.
* The format is plain, readable JSON — diffable in git, greppable, and fixable
  by hand when something goes wrong.

Writes are atomic (temp file in the same directory, then ``os.replace``), so a
crash or a full disk mid-save cannot leave a truncated file where a good one used
to be. Losing an hour of labelling to a partial write is not an acceptable
failure mode.
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
    "SIDECAR_SUFFIX",
    "delete_sidecar",
    "load_annotations",
    "load_into_store",
    "save_annotations",
    "save_store",
    "sidecar_path_for",
    "write_json_atomic",
]

_log = get_logger(__name__)

SIDECAR_SUFFIX = ".vidtriage.json"
_FORMAT_VERSION = 1


def sidecar_path_for(media_path: Path) -> Path:
    """``/videos/clip.mp4`` → ``/videos/clip.mp4.vidtriage.json``.

    The full name is kept, extension included, so ``clip.mp4`` and ``clip.avi``
    in one folder do not fight over the same sidecar.
    """
    return media_path.with_name(media_path.name + SIDECAR_SUFFIX)


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


def load_annotations(media_path: Path) -> list[Annotation]:
    """Read a sidecar. Missing or corrupt files yield an empty list.

    A malformed sidecar must not stop the user opening the video, so the failure
    is logged and treated as "no annotations yet". Individual bad entries are
    skipped rather than discarding the whole file.
    """
    path = sidecar_path_for(media_path)
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _log.warning("Ignoring unreadable sidecar %s: %s", path.name, exc)
        return []
    if not isinstance(data, dict):
        _log.warning("Ignoring malformed sidecar %s: not an object", path.name)
        return []

    source_id = source_id_for(media_path)
    annotations: list[Annotation] = []
    for raw in data.get("annotations", []):
        try:
            annotations.append(Annotation.from_dict(raw, source_id))
        except (KeyError, TypeError, ValueError) as exc:
            _log.warning("Skipping bad annotation in %s: %s", path.name, exc)
    return annotations


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
    path = sidecar_path_for(media_path)
    if not path.exists():
        return False
    try:
        path.unlink()
        return True
    except OSError as exc:
        _log.warning("Could not delete %s: %s", path.name, exc)
        return False
