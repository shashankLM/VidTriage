"""Materialise the log as folders of video — the deliverable, built on demand.

The log says what every video is. A snapshot turns that into the shape other
tools expect::

    <outdir>/green/clip_a.mp4
    <outdir>/red/clip_b.mp4
    <outdir>/_errors/clip_c.mp4

Three rules, all in service of the same idea — a snapshot is derived data, and
derived data must never be able to damage its source or lie about its contents.

**The target must be empty or absent.** Snapshotting into a directory that
already has content would leave stale copies behind whenever a video's class
changed between runs: the old copy sits in the old class folder and nothing
distinguishes it from a current one. Refusing is the only answer that cannot
silently produce a wrong deliverable.

**Filename collisions are found before anything is written.** Destinations are
derived from the filename, so two videos called ``clip.mp4`` in different source
folders would land on top of each other. The whole plan is checked first, so a
collision aborts with both paths named and nothing half-written.

**Sources are only ever read.** A failed snapshot costs you a directory to
delete, never a video.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ....core.errors import FileOperationError
from ....core.logging import get_logger
from .io_ops import copy_media
from .models import VideoItem

__all__ = ["SnapshotPlan", "SnapshotResult", "plan_snapshot", "write_snapshot"]

_log = get_logger(__name__)


@dataclass
class SnapshotPlan:
    """What a snapshot would write, and why it might not be able to."""

    pairs: list[tuple[VideoItem, Path]] = field(default_factory=list)
    collisions: dict[str, list[Path]] = field(default_factory=dict)
    missing: list[VideoItem] = field(default_factory=list)
    skipped_pending: int = 0

    @property
    def is_writable(self) -> bool:
        return not self.collisions and bool(self.pairs)


@dataclass
class SnapshotResult:
    """What a snapshot actually wrote."""

    written: int = 0
    output_dir: Path | None = None
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{self.written} file(s) copied"]
        if self.output_dir:
            parts.append(f"to {self.output_dir}")
        if self.warnings:
            parts.append(f"· {len(self.warnings)} warning(s)")
        return " ".join(parts)


def plan_snapshot(items: list[VideoItem], output_dir: Path) -> SnapshotPlan:
    """Work out every source→destination pair, and every reason not to proceed.

    Separate from :func:`write_snapshot` so the UI can show the user what is
    about to happen — and, more importantly, so a collision is reported before
    the first byte is written rather than halfway through.
    """
    plan = SnapshotPlan()
    by_destination: defaultdict[str, list[Path]] = defaultdict(list)

    for item in items:
        destination = item.snapshot_path(output_dir)
        if destination is None:
            plan.skipped_pending += 1
            continue
        if not item.original_path.exists():
            plan.missing.append(item)
            continue
        by_destination[str(destination)].append(item.original_path)
        plan.pairs.append((item, destination))

    plan.collisions = {
        destination: sources
        for destination, sources in by_destination.items() if len(sources) > 1
    }
    if plan.collisions:
        plan.pairs = []
    return plan


def write_snapshot(
    items: list[VideoItem],
    output_dir: Path,
    *,
    link: bool = False,
) -> SnapshotResult:
    """Copy every classified video into ``output_dir/<class>/``.

    Args:
        link: Prefer hardlinks, falling back to copying per file. See
            :func:`~vidtriage.plugins.builtin.triage.io_ops.copy_media`.

    Raises:
        FileOperationError: the target is non-empty, nothing is classified, or
            two videos would collide. Nothing is written in any of those cases.
    """
    output_dir = Path(output_dir)
    _require_empty(output_dir)

    plan = plan_snapshot(items, output_dir)
    if plan.collisions:
        raise FileOperationError(_collision_message(plan.collisions))
    if not plan.pairs:
        raise FileOperationError(
            "Nothing to snapshot — no video has been classified yet.",
        )

    result = SnapshotResult(output_dir=output_dir)
    for item in plan.missing:
        result.warnings.append(f"{item.name}: source file is gone, skipped")

    for item, destination in plan.pairs:
        try:
            copy_media(item.original_path, destination, link=link)
        except FileOperationError as exc:
            # One unreadable file must not abandon a snapshot that is most of
            # the way done; the warning names it so the user can fix and re-run.
            result.warnings.append(str(exc))
            continue
        result.written += 1

    _log.info(
        "Snapshot: %d file(s) -> %s%s",
        result.written, output_dir, " (hardlinked)" if link else "",
    )
    return result


def _require_empty(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise FileOperationError(f"Not a directory: {output_dir}")
    try:
        occupied = any(output_dir.iterdir())
    except OSError as exc:
        raise FileOperationError(f"Cannot read {output_dir}: {exc}") from exc
    if occupied:
        raise FileOperationError(
            f"{output_dir} is not empty.\n\n"
            f"A snapshot must go into a new or empty directory — writing into "
            f"one that already has content would leave copies from a previous "
            f"run in class folders they no longer belong to.",
        )


def _collision_message(collisions: dict[str, list[Path]]) -> str:
    lines = [
        "Two or more videos would be written to the same place.",
        "",
        "A snapshot names files by their original filename, so duplicates "
        "collide. Rename the sources, then snapshot again.",
        "",
    ]
    for destination, sources in list(collisions.items())[:10]:
        lines.append(f"{destination}")
        lines.extend(f"    ← {source}" for source in sources)
    if len(collisions) > 10:
        lines.append(f"…and {len(collisions) - 10} more")
    return "\n".join(lines)
