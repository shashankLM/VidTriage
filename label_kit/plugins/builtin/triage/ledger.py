"""The classification log — an append-only journal of decisions.

Triage used to move your files: the class folder a video sat in *was* the
record of what you decided. That made the source tree the database, so a
mis-key rewrote your footage layout, undo meant moving files back, and two
passes over the same corpus could not disagree without one of them destroying
the other's work.

Here nothing on disk moves. A decision is one line appended to a log:

.. code-block:: json

    {"v": 1, "at": "2026-07-31T18:20:05Z", "file": "clips/a.mp4", "class": "green"}
    {"v": 1, "at": "2026-07-31T18:20:11Z", "file": "clips/b.mp4", "class": null}

``class: null`` means "back to pending" — that is what undo writes. State is a
fold over the records: **the last decision for a file wins**. Because the fold
is order-dependent and nothing else is, replaying several logs in sequence gives
overlay semantics for free — see :func:`replay`. A later pass corrects an
earlier one without either log being edited, and dropping a log from the stack
cleanly un-applies it.

Identity is the path relative to the input directory (POSIX separators, so a log
written on Windows replays on Linux). Files outside that directory — the ones a
:mod:`legacy import <label_kit.plugins.builtin.triage.session>` picks up from an
old run's output folders — are recorded absolute.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ....core.logging import get_logger
from ....persistence.settings import CONFIG_DIR

__all__ = [
    "LOG_SUFFIX",
    "Decision",
    "Ledger",
    "NullLedger",
    "default_log_dir",
    "discover_logs",
    "identity_for",
    "new_log_path",
    "read_log",
    "replay",
    "summarise",
]

_log = get_logger(__name__)

LOG_SUFFIX = ".triage.jsonl"
FORMAT_VERSION = 1

_LOG_KIND = "labelkit-triage-log"
#: Header kinds to skip when reading. The first is what label-kit wrote before
#: the rename; logs written then are still the only record of those decisions.
_HEADER_KINDS = frozenset({_LOG_KIND, "vidtriage-triage-log"})

_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class Decision:
    """One classification decision.

    Attributes:
        file: Identity of the video — see the module docstring.
        class_name: The class it was filed under, or ``None`` for pending.
        at: ISO-8601 UTC timestamp, for humans reading the log.
    """

    file: str
    class_name: str | None
    at: str = ""

    def to_json(self) -> dict[str, object]:
        return {"v": FORMAT_VERSION, "at": self.at, "file": self.file, "class": self.class_name}

    @classmethod
    def from_json(cls, data: dict[str, object]) -> Decision:
        file = data.get("file")
        if not isinstance(file, str) or not file:
            raise ValueError("record has no 'file'")
        class_name = data.get("class")
        if class_name is not None and not isinstance(class_name, str):
            raise ValueError(f"'class' must be a string or null, got {type(class_name).__name__}")
        return cls(file=file, class_name=class_name, at=str(data.get("at", "")))


def now_stamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def identity_for(path: Path, input_dir: Path) -> str:
    """The key a decision is recorded under.

    Relative to ``input_dir`` where possible so a log survives the whole corpus
    being moved or mounted elsewhere; absolute otherwise.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(input_dir).resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


class Ledger:
    """Append-only writer for one log file.

    Opens, writes and closes per record rather than holding a handle for the
    session. Decisions arrive at human speed, so the cost is irrelevant, and it
    means a crash — or a second label-kit running over the same corpus — cannot
    lose buffered records or leave a half-written line.
    """

    #: False on :class:`NullLedger`; callers use it to skip bookkeeping that
    #: only makes sense when decisions are actually being persisted.
    records = True

    def __init__(self, path: Path, input_dir: Path | None = None) -> None:
        self.path = Path(path)
        self.input_dir = Path(input_dir) if input_dir else None

    def append(self, decision: Decision) -> None:
        """Write one decision. Failures are logged, never raised.

        A lost log line costs one re-key. Interrupting triage with a dialog
        because the disk hiccuped costs the whole session.
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not self.path.exists()
            with self.path.open("a", encoding="utf-8") as handle:
                if new_file:
                    handle.write(json.dumps(self._header()) + "\n")
                handle.write(json.dumps(decision.to_json()) + "\n")
        except OSError as exc:
            _log.error("Could not append to %s: %s", self.path, exc)

    def _header(self) -> dict[str, object]:
        return {
            "v": FORMAT_VERSION,
            "kind": _LOG_KIND,
            "input_dir": str(self.input_dir) if self.input_dir else None,
            "created": now_stamp(),
        }


class NullLedger(Ledger):
    """Discards every record.

    For passes that only *read* the decisions — ``--snapshot`` most of all. A
    snapshot that quietly appended a log of its own would turn every build of
    the deliverable into another overlay layer.
    """

    records = False

    def __init__(self) -> None:
        super().__init__(Path(os.devnull))

    def append(self, decision: Decision) -> None:
        pass


def read_log(path: Path) -> list[Decision]:
    """Every decision in one log, in file order.

    A malformed line is skipped with a warning rather than failing the read: a
    log truncated by a crash should still yield the decisions before the tear.
    """
    decisions: list[Decision] = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        _log.warning("Cannot read log %s: %s", path, exc)
        return decisions

    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            data = json.loads(stripped)
            if not isinstance(data, dict):
                raise ValueError("not an object")
            if data.get("kind") in _HEADER_KINDS:
                continue
            decisions.append(Decision.from_json(data))
        except (json.JSONDecodeError, ValueError) as exc:
            _log.warning("%s:%d ignored — %s", Path(path).name, number, exc)
    return decisions


def replay(logs: list[Path]) -> dict[str, str | None]:
    """Fold logs into the final state: ``{file identity: class or None}``.

    Order is the whole contract. Logs are applied in the order given and records
    within a log in file order, so the last write wins at both levels — pass the
    stack oldest-first and a later pass overrides an earlier one.
    """
    state: dict[str, str | None] = {}
    for path in logs:
        for decision in read_log(path):
            state[decision.file] = decision.class_name
    return state


def summarise(path: Path) -> tuple[int, str, str]:
    """``(decision count, first timestamp, last timestamp)`` for a log.

    Used by the setup dialog to label each log in the stack, so the order can be
    judged without opening the files.
    """
    decisions = read_log(path)
    stamps = [d.at for d in decisions if d.at]
    return len(decisions), (stamps[0] if stamps else ""), (stamps[-1] if stamps else "")


def default_log_dir(input_dir: Path) -> Path:
    """Where logs for this corpus live, under the config dir.

    Deliberately not inside the input or output tree: the input directory is
    exactly what this feature exists to stop writing to, and an output tree
    should be reproducible from the logs rather than contain them.
    """
    resolved = Path(input_dir).resolve()
    slug = _SLUG_UNSAFE.sub("-", resolved.name) or "corpus"
    # sha1, not hash(): str hashing is salted per process, so the built-in would
    # hand every launch a different directory and lose the previous run's logs.
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:8]
    return CONFIG_DIR / "logs" / f"{slug}-{digest}"


def discover_logs(directory: Path) -> list[Path]:
    """Every log in ``directory``, oldest first.

    Names carry a UTC timestamp, so sorting by name sorts by time — which is
    also the correct default overlay order.
    """
    if not Path(directory).is_dir():
        return []
    try:
        return sorted(p for p in Path(directory).iterdir() if p.name.endswith(LOG_SUFFIX))
    except OSError as exc:
        _log.warning("Cannot list %s: %s", directory, exc)
        return []


def new_log_path(directory: Path) -> Path:
    """A fresh, uniquely named log for this run."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    candidate = Path(directory) / f"{stamp}{LOG_SUFFIX}"
    counter = 1
    while candidate.exists():
        candidate = Path(directory) / f"{stamp}-{counter}{LOG_SUFFIX}"
        counter += 1
    return candidate
