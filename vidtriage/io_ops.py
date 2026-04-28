from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .models import ERRORS_FOLDER, ClassEntry

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
    _logger.info("Session started, output_dir: %s", output_dir)


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


def scan_output_subfolders(output_dir: Path) -> list[tuple[str, list[Path]]]:
    """Return ``[(folder_name, [video_paths, …]), …]`` for each non-empty subfolder."""
    results: list[tuple[str, list[Path]]] = []
    if not output_dir.is_dir():
        return results
    for subdir in sorted(output_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("."):
            continue
        if subdir.name.isdigit():
            continue
        vids = discover_videos(subdir)
        if vids:
            results.append((subdir.name, vids))
    return results


def move_to_class(source: Path, output_dir: Path, class_entry: ClassEntry) -> Path:
    class_dir = output_dir / class_entry.name
    class_dir.mkdir(parents=True, exist_ok=True)
    dest = class_dir / source.name
    shutil.move(str(source), str(dest))
    _log(f"CLASSIFY [{class_entry.key}:{class_entry.name}] {source} -> {dest}")
    return dest


def move_to_errors(source: Path, output_dir: Path) -> Path:
    error_dir = output_dir / ERRORS_FOLDER
    error_dir.mkdir(parents=True, exist_ok=True)
    dest = error_dir / source.name
    shutil.move(str(source), str(dest))
    _log(f"ERROR {source} -> {dest}")
    return dest


def undo_move(destination: Path, original_path: Path) -> None:
    original_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(destination), str(original_path))
    _log(f"UNDO {destination} -> {original_path}")
