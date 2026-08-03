"""Saved triage sessions, keyed by output directory.

Kept in ``~/.vidtriage/sessions.json``, most-recently-used first, so relaunching
drops you back into the directory pair and class list you were last working on.
The legacy single-session file layout is still read, so upgrading does not lose
an existing setup.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ....core.logging import get_logger
from ....persistence.settings import CONFIG_DIR
from ....persistence.sidecar import write_json_atomic
from .models import MAX_CLASSES, ClassEntry, TriageConfig

__all__ = [
    "SESSIONS_FILE",
    "load_all_sessions",
    "load_last_session",
    "parse_classes",
    "save_session",
]

_log = get_logger(__name__)

SESSIONS_FILE = CONFIG_DIR / "sessions.json"
_LEGACY_FILE = CONFIG_DIR / "config.json"


def _parse_entry(data: dict[str, Any]) -> TriageConfig:
    classes = [
        ClassEntry(key=str(c["key"]), name=str(c["name"]))
        for c in data.get("classes", [])
        if c.get("name") and c.get("name") != c.get("key")
    ]
    return TriageConfig(
        input_dir=Path(data["input_dir"]) if data.get("input_dir") else None,
        output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
        classes=classes,
    )


def _read_raw() -> list[dict[str, Any]]:
    for path in (SESSIONS_FILE, _LEGACY_FILE):
        if not path.exists():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _log.warning("Ignoring unreadable session file %s: %s", path, exc)
            continue
        if isinstance(raw, dict) and "sessions" in raw:
            entries = raw["sessions"]
            return entries if isinstance(entries, list) else []
        if isinstance(raw, dict) and raw.get("output_dir"):
            return [raw]  # legacy: one bare session object
    return []


def load_all_sessions() -> list[TriageConfig]:
    """Every saved session, most recently used first."""
    entries = _read_raw()
    entries.sort(key=lambda e: str(e.get("last_used", "")), reverse=True)

    sessions: list[TriageConfig] = []
    for entry in entries:
        try:
            sessions.append(_parse_entry(entry))
        except (KeyError, TypeError, ValueError) as exc:
            _log.warning("Skipping malformed session entry: %s", exc)
    return sessions


def load_last_session() -> TriageConfig:
    sessions = load_all_sessions()
    return sessions[0] if sessions else TriageConfig()


def save_session(config: TriageConfig) -> None:
    """Insert or update the session for ``config.output_dir``."""
    if not config.output_dir:
        return

    entries = _read_raw()
    new_entry = {
        "input_dir": str(config.input_dir) if config.input_dir else None,
        "output_dir": str(config.output_dir),
        "classes": [{"key": c.key, "name": c.name} for c in config.classes],
        "last_used": datetime.now().isoformat(timespec="seconds"),
    }

    key = str(config.output_dir)
    for i, entry in enumerate(entries):
        if entry.get("output_dir") == key:
            entries[i] = new_entry
            break
    else:
        entries.append(new_entry)

    try:
        write_json_atomic(SESSIONS_FILE, {"sessions": entries})
    except Exception as exc:  # noqa: BLE001 - never lose a session over a failed write
        _log.warning("Could not save sessions to %s: %s", SESSIONS_FILE, exc)


def parse_classes(text: str) -> tuple[list[ClassEntry], list[str]]:
    """Parse one class name per line into keyed entries.

    Returns ``(entries, errors)``. Keys are assigned 1–9 in order. Duplicates
    are rejected rather than silently collapsed, because two classes with the
    same name would share an output folder.
    """
    entries: list[ClassEntry] = []
    errors: list[str] = []
    seen: set[str] = set()

    for raw_line in text.splitlines():
        name = raw_line.strip()
        if not name:
            continue
        if name.lower() in seen:
            errors.append(f"Duplicate class name: {name!r}")
            continue
        if "/" in name or "\\" in name or name in (".", ".."):
            errors.append(f"{name!r} is not a usable folder name")
            continue
        if len(entries) >= MAX_CLASSES:
            errors.append(f"Maximum {MAX_CLASSES} classes (keys 1-{MAX_CLASSES})")
            break
        seen.add(name.lower())
        entries.append(ClassEntry(key=str(len(entries) + 1), name=name))

    if not entries and not errors:
        errors.append("No classes defined")
    return entries, errors
