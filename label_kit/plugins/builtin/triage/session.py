"""Triage session state — the single owner of every classification decision.

The UI reads through properties and mutates through action methods. Nothing else
appends to the log or edits a history list, which is what keeps undo
trustworthy: the undo stack and the log cannot drift apart if there is exactly
one path that changes either.

**Files are never moved.** A session reads its input directory, replays a stack
of logs to recover what was already decided, and appends each new decision to
one fresh log of its own. That is the whole persistence model — see
:mod:`ledger` for why the log is the database rather than the folder layout.
"""

from __future__ import annotations

from pathlib import Path

from ....app.library import discover_media
from ....core.logging import get_logger
from .config import save_session
from .io_ops import scan_output_subfolders
from .ledger import (
    Decision,
    Ledger,
    NullLedger,
    default_log_dir,
    discover_logs,
    identity_for,
    new_log_path,
    now_stamp,
    replay,
)
from .models import ERRORS_FOLDER, MAX_CLASSES, ClassEntry, MediaItem, TriageConfig

__all__ = ["Session"]

_log = get_logger(__name__)


class Session:
    """Every media item, class and undo state for one input directory.

    Args:
        input_dir: The corpus. Read-only for the life of the session.
        output_dir: Where a snapshot would be written. Nothing is put there
            until the user asks for one.
        classes: The class list bound to keys 1–9.
        logs: Logs to replay, **oldest first** — later logs override earlier
            ones. ``None`` means auto-discover this corpus's own log directory.
        log_path: Where this session's decisions are appended. ``None`` mints a
            fresh timestamped log, which is what makes a rerun a separate,
            overlayable pass rather than an edit of the previous one.
        record: ``False`` for a read-only pass — ``--snapshot`` uses it so
            building the deliverable does not itself become an overlay layer.
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        classes: list[ClassEntry],
        *,
        logs: list[Path] | None = None,
        log_path: Path | None = None,
        record: bool = True,
    ) -> None:
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.classes = list(classes)
        self.log_dir = default_log_dir(self.input_dir)
        self.logs = list(logs) if logs is not None else discover_logs(self.log_dir)
        self._ledger: Ledger = (
            Ledger(Path(log_path) if log_path else new_log_path(self.log_dir), self.input_dir)
            if record else NullLedger()
        )
        self.log_path = self._ledger.path
        self._items: dict[str, MediaItem] = {}
        self._undo_order: list[MediaItem] = []

    # ── derived state ───────────────────────────────────────────────────

    @property
    def pending(self) -> list[MediaItem]:
        return sorted(
            (item for item in self._items.values() if item.is_pending),
            key=lambda item: item.name.lower(),
        )

    @property
    def classified(self) -> list[MediaItem]:
        return sorted(
            (item for item in self._items.values() if not item.is_pending),
            key=lambda item: item.name.lower(),
        )

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_order)

    def find_by_path(self, path: Path) -> MediaItem | None:
        resolved = Path(path).resolve()
        return next(
            (item for item in self._items.values() if item.original_path == resolved),
            None,
        )

    # ── lifecycle ───────────────────────────────────────────────────────

    def load(self) -> None:
        """Rebuild state from the input directory and the log stack."""
        self._items.clear()
        self._undo_order.clear()

        for path in discover_media(self.input_dir):
            self._items[identity_for(path, self.input_dir)] = MediaItem(original_path=path)

        imported = self.import_legacy_output()
        self._apply(replay(self.logs))

        _log.info(
            "Session loaded: %d pending, %d classified, %d log(s)%s",
            len(self.pending), len(self.classified), len(self.logs),
            f", {imported} imported from a previous move-based run" if imported else "",
        )

    def _apply(self, state: dict[str, str | None]) -> None:
        """Set each item's class from a replayed log stack."""
        for key, class_name in state.items():
            item = self._items.get(key) or self._adopt(key)
            if item is None:
                continue
            item.history = [] if class_name is None else [class_name]
            unknown = class_name not in (None, ERRORS_FOLDER) and not any(
                c.name == class_name for c in self.classes
            )
            if unknown:
                self._make_class_for(str(class_name))

    def _adopt(self, key: str) -> MediaItem | None:
        """Pick up an item the log knows about but discovery did not find.

        Only absolute keys qualify, and only if the file is still there. That is
        how a legacy-imported file keeps working on the second run: it lives in
        an old output folder rather than the input directory, so nothing but the
        log knows where it is. A *relative* key with no match means the file has
        left the corpus — that decision stays in the log, unapplied, because the
        file may well come back.
        """
        path = Path(key)
        if not path.is_absolute() or not path.is_file():
            return None
        item = MediaItem(original_path=path)
        self._items[key] = item
        return item

    def import_legacy_output(self) -> int:
        """Adopt media an older, move-based run left in class folders.

        Runs once. Those files are not in the input directory any more — the old
        implementation moved them — so they are tracked where they actually are
        and their class is written into this session's log. Nothing is moved
        back: the point of the change is to stop relocating footage, and doing
        one big move to prove it would be absurd.

        "Once" means *once per corpus*, which is why the guard asks the log
        directory rather than the replay stack. Those differ: emptying the stack
        in the setup dialog is how a user says "ignore the previous passes", and
        reading that as "this corpus has never been triaged" would rescan the
        output directory — quite possibly a snapshot this tool wrote — and
        fabricate a full set of classifications out of it.
        """
        if self.logs or discover_logs(self.log_dir):
            return 0

        imported = 0
        for folder_name, videos in scan_output_subfolders(self.output_dir):
            needs_class = folder_name != ERRORS_FOLDER and not any(
                c.name == folder_name for c in self.classes
            )
            if needs_class and self._make_class_for(folder_name) is None:
                _log.warning(
                    "Ignoring folder %r: all %d class keys are taken",
                    folder_name, MAX_CLASSES,
                )
                continue

            for path in videos:
                key = identity_for(path, self.input_dir)
                if key in self._items:
                    continue
                item = MediaItem(original_path=path)
                item.history.append(folder_name)
                self._items[key] = item
                self._ledger.append(Decision(key, folder_name, now_stamp()))
                imported += 1

        if imported:
            if self._ledger.records:
                self.logs = [*self.logs, self.log_path]
                self.save_config()
            _log.info("Imported %d already-classified file(s) into %s", imported, self.log_path)
        return imported

    def _make_class_for(self, name: str) -> ClassEntry | None:
        used = {c.key for c in self.classes}
        key = next((str(i) for i in range(1, MAX_CLASSES + 1) if str(i) not in used), None)
        if key is None:
            return None
        entry = ClassEntry(key=key, name=name)
        self.classes.append(entry)
        self.classes.sort(key=lambda c: c.key)
        return entry

    # ── actions ─────────────────────────────────────────────────────────

    def classify(self, item: MediaItem, class_entry: ClassEntry) -> None:
        """Record ``item`` as ``class_entry``. Touches no file.

        Works for pending and already-classified videos alike — reclassifying
        just appends another decision, and the later one wins on replay.
        """
        self._record(item, class_entry.name)

    def mark_error(self, item: MediaItem) -> None:
        self._record(item, ERRORS_FOLDER)

    def _record(self, item: MediaItem, class_name: str) -> None:
        item.history.append(class_name)
        self._undo_order.append(item)
        self._ledger.append(
            Decision(identity_for(item.original_path, self.input_dir), class_name, now_stamp()),
        )

    def undo_last(self) -> MediaItem | None:
        """Reverse the most recent decision made *in this session*.

        Undo is itself a decision: it appends a record restoring the previous
        class, or ``None`` for pending. The log is never rewritten, so the trail
        of what you actually did survives — and a log that is only ever appended
        to is one that a concurrent reader can never catch mid-edit.

        Returns the affected item, or ``None`` when there was nothing to undo.
        """
        if not self._undo_order:
            return None

        item = self._undo_order.pop()
        if item.history:
            item.history.pop()

        self._ledger.append(
            Decision(
                identity_for(item.original_path, self.input_dir),
                item.class_name,
                now_stamp(),
            ),
        )
        return item

    # ── class management ────────────────────────────────────────────────

    def add_class(self, key: str, name: str) -> ClassEntry:
        entry = ClassEntry(key=key, name=name)
        self.classes.append(entry)
        self.classes.sort(key=lambda c: c.key)
        self.save_config()
        return entry

    def set_classes(self, classes: list[ClassEntry]) -> None:
        self.classes = list(classes)
        self.save_config()

    # ── persistence ─────────────────────────────────────────────────────

    def to_config(self) -> TriageConfig:
        return TriageConfig(
            input_dir=self.input_dir, output_dir=self.output_dir, classes=self.classes,
        )

    def save_config(self) -> None:
        save_session(self.to_config())
