"""Application settings and UI state.

A single JSON-backed key/value store under ``~/.vidtriage/``. Keys are dotted
(``"window.width"``, ``"canvas.show_labels"``) so any plugin can persist state
without a schema migration or a new file.

Nothing here ever raises on a bad or missing file. Settings are a convenience —
losing them costs the user a window size, not their work — so a corrupt file is
logged, backed up once, and replaced with defaults.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..core.logging import get_logger
from .sidecar import write_json_atomic

__all__ = ["CONFIG_DIR", "Settings", "default_settings_path", "user_plugin_dir"]

_log = get_logger(__name__)

CONFIG_DIR = Path.home() / ".vidtriage"


def default_settings_path() -> Path:
    return CONFIG_DIR / "settings.json"


def user_plugin_dir() -> Path:
    """Where drop-in plugins live. Created on first launch with a README."""
    directory = CONFIG_DIR / "plugins"
    if not directory.exists():
        try:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "README.txt").write_text(
                "Drop a .py file or a package directory here to add a VidTriage plugin.\n"
                "It must define a module-level PLUGIN attribute naming a Plugin subclass.\n"
                "See View > Plugins in the app for what was found and why anything failed.\n",
                encoding="utf-8",
            )
        except OSError as exc:
            _log.warning("Could not create user plugin directory %s: %s", directory, exc)
    return directory


class Settings:
    """Dotted-key JSON store."""

    def __init__(self, path: Path | None = None, autosave: bool = True) -> None:
        self.path = path or default_settings_path()
        self.autosave = autosave
        self._data: dict[str, Any] = self._load()
        self._dirty = False

    # ── loading ─────────────────────────────────────────────────────────

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _log.warning("Settings at %s are unreadable (%s); starting fresh", self.path, exc)
            self._back_up_corrupt_file()
            return {}
        if not isinstance(data, dict):
            _log.warning("Settings at %s are not an object; starting fresh", self.path)
            return {}
        return data

    def _back_up_corrupt_file(self) -> None:
        """Keep one copy of the bad file so a hand-edit gone wrong is recoverable."""
        with contextlib.suppress(OSError):
            self.path.replace(self.path.with_suffix(".json.corrupt"))

    # ── access ──────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, key: str, value: Any) -> None:
        parts = key.split(".")
        node = self._data
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        if node.get(parts[-1]) == value:
            return
        node[parts[-1]] = value
        self._dirty = True
        if self.autosave:
            self.save()

    def update(self, values: dict[str, Any]) -> None:
        """Set many dotted keys, saving at most once."""
        autosave, self.autosave = self.autosave, False
        try:
            for key, value in values.items():
                self.set(key, value)
        finally:
            self.autosave = autosave
        if self.autosave and self._dirty:
            self.save()

    def remove(self, key: str) -> None:
        parts = key.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.get(part)
            if not isinstance(node, dict):
                return
        if node.pop(parts[-1], None) is not None:
            self._dirty = True
            if self.autosave:
                self.save()

    def section(self, prefix: str) -> dict[str, Any]:
        value = self.get(prefix, {})
        return dict(value) if isinstance(value, dict) else {}

    def __contains__(self, key: str) -> bool:
        sentinel = object()
        return self.get(key, sentinel) is not sentinel

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    # ── saving ──────────────────────────────────────────────────────────

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def save(self) -> bool:
        try:
            write_json_atomic(self.path, self._data)
        except Exception as exc:  # noqa: BLE001 - never let a settings write break the app
            _log.warning("Could not save settings to %s: %s", self.path, exc)
            return False
        self._dirty = False
        return True
