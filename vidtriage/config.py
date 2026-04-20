from __future__ import annotations

import json
from pathlib import Path

from .models import AppConfig, ClassEntry

CONFIG_DIR = Path.home() / ".vidtriage"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_config() -> AppConfig:
    if not CONFIG_FILE.exists():
        return AppConfig()
    try:
        data = json.loads(CONFIG_FILE.read_text())
        classes = [ClassEntry(key=c["key"], name=c["name"]) for c in data.get("classes", [])]
        input_dir = Path(data["input_dir"]) if data.get("input_dir") else None
        output_dir = Path(data["output_dir"]) if data.get("output_dir") else None
        return AppConfig(input_dir=input_dir, output_dir=output_dir, classes=classes)
    except (json.JSONDecodeError, KeyError, TypeError):
        return AppConfig()


def save_config(config: AppConfig) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "input_dir": str(config.input_dir) if config.input_dir else None,
        "output_dir": str(config.output_dir) if config.output_dir else None,
        "classes": [{"key": c.key, "name": c.name} for c in config.classes],
    }
    CONFIG_FILE.write_text(json.dumps(data, indent=2))


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

    for i in range(len(entries) + 1, 10):
        entries.append(ClassEntry(key=str(i), name=str(i)))

    return entries, errors
