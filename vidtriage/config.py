from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .models import AppConfig, ClassEntry

CONFIG_DIR = Path.home() / ".vidtriage"
CONFIG_FILE = CONFIG_DIR / "config.json"


def _parse_session(data: dict) -> AppConfig:
    classes = [
        ClassEntry(key=c["key"], name=c["name"])
        for c in data.get("classes", [])
        if c.get("name") != c.get("key")
    ]
    input_dir = Path(data["input_dir"]) if data.get("input_dir") else None
    output_dir = Path(data["output_dir"]) if data.get("output_dir") else None
    return AppConfig(input_dir=input_dir, output_dir=output_dir, classes=classes)


def load_all_sessions() -> list[AppConfig]:
    """All saved sessions, most recently used first."""
    if not CONFIG_FILE.exists():
        return []
    try:
        raw = json.loads(CONFIG_FILE.read_text())
    except (json.JSONDecodeError, TypeError):
        return []

    if "sessions" not in raw:
        config = _parse_session(raw)
        if config.output_dir:
            return [config]
        return []

    entries = raw.get("sessions", [])
    entries.sort(key=lambda e: e.get("last_used", ""), reverse=True)

    sessions: list[AppConfig] = []
    for entry in entries:
        try:
            sessions.append(_parse_session(entry))
        except (KeyError, TypeError):
            continue
    return sessions


def load_config() -> AppConfig:
    """Most recently used session, or empty config."""
    sessions = load_all_sessions()
    return sessions[0] if sessions else AppConfig()


def save_config(config: AppConfig) -> None:
    """Upsert session by output_dir, update last_used timestamp."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    sessions_data: list[dict] = []
    if CONFIG_FILE.exists():
        try:
            raw = json.loads(CONFIG_FILE.read_text())
            if "sessions" in raw:
                sessions_data = raw["sessions"]
            elif raw.get("output_dir"):
                sessions_data = [raw]
        except (json.JSONDecodeError, TypeError):
            pass

    new_entry = {
        "input_dir": str(config.input_dir) if config.input_dir else None,
        "output_dir": str(config.output_dir) if config.output_dir else None,
        "classes": [{"key": c.key, "name": c.name} for c in config.classes],
        "last_used": datetime.now().isoformat(),
    }

    output_key = str(config.output_dir) if config.output_dir else None
    found = False
    for i, entry in enumerate(sessions_data):
        if entry.get("output_dir") == output_key:
            sessions_data[i] = new_entry
            found = True
            break
    if not found:
        sessions_data.append(new_entry)

    CONFIG_FILE.write_text(json.dumps({"sessions": sessions_data}, indent=2))


def parse_classes(text: str) -> tuple[list[ClassEntry], list[str]]:
    entries: list[ClassEntry] = []
    errors: list[str] = []

    for raw_line in text.splitlines():
        name = raw_line.strip()
        if not name:
            continue
        if len(entries) >= 9:
            errors.append("Maximum 9 classes (keys 1-9)")
            return [], errors
        entries.append(ClassEntry(key=str(len(entries) + 1), name=name))

    if not entries:
        errors.append("No classes defined")
        return entries, errors

    return entries, errors
