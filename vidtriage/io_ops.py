from __future__ import annotations

import csv
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .models import ClassEntry

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
    ".m4v", ".mpg", ".mpeg", ".3gp", ".ts", ".mts",
}

_logger: logging.Logger | None = None


def setup_logger(output_dir: Path) -> None:
    global _logger
    _logger = logging.getLogger("vidtriage")
    _logger.setLevel(logging.DEBUG)
    _logger.handlers.clear()

    log_file = output_dir / "vidtriage_activity.log"
    fh = logging.FileHandler(str(log_file), mode="a")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _logger.addHandler(fh)
    _logger.info("Session started, input_dir logs in: %s", output_dir)


def _log(msg: str) -> None:
    if _logger:
        _logger.info(msg)


def discover_videos(directory: Path) -> list[Path]:
    files = [
        f for f in sorted(directory.iterdir())
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ]
    _log(f"Discovered {len(files)} videos in {directory}")
    return files


def _unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    counter = 1
    while True:
        candidate = parent / f"{stem} ({counter}){suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def move_to_class(source: Path, output_dir: Path, class_entry: ClassEntry) -> Path:
    class_dir = output_dir / class_entry.name
    class_dir.mkdir(parents=True, exist_ok=True)
    dest = _unique_dest(class_dir / source.name)
    shutil.move(str(source), str(dest))
    _log(f"CLASSIFY [{class_entry.key}:{class_entry.name}] {source} -> {dest}")
    return dest


def move_to_errors(source: Path, output_dir: Path) -> Path:
    error_dir = output_dir / "_errors"
    error_dir.mkdir(parents=True, exist_ok=True)
    dest = _unique_dest(error_dir / source.name)
    shutil.move(str(source), str(dest))
    _log(f"ERROR {source} -> {dest}")
    return dest


def undo_move(destination: Path, original_path: Path) -> None:
    original_path.parent.mkdir(parents=True, exist_ok=True)
    dest = _unique_dest(original_path)
    shutil.move(str(destination), str(dest))
    _log(f"UNDO {destination} -> {dest}")


def log_action(
    output_dir: Path,
    source_path: Path,
    destination_path: Path,
    class_key: str,
    class_name: str,
    action: str,
) -> None:
    log_file = output_dir / "vidtriage_log.csv"
    write_header = not log_file.exists()
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "source_path", "destination_path", "class_key", "class_name", "action"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            str(source_path),
            str(destination_path),
            class_key,
            class_name,
            action,
        ])


def load_log(output_dir: Path) -> list[dict[str, str]]:
    log_file = output_dir / "vidtriage_log.csv"
    if not log_file.exists():
        return []
    rows: list[dict[str, str]] = []
    with open(log_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows
