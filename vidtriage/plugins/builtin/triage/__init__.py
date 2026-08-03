"""Whole-video triage — the original VidTriage workflow, as a plugin.

Press a number key, the decision is logged, the next video loads. It contributes
commands, a panel and an exporter, and can be switched off in View ▸ Plugins —
leaving a plain frame-annotation tool with no trace of classification in the UI.

If the built-in workflow can be expressed this way, so can yours.

**Your files are never moved.** Classification appends a line to a log; the
input tree is read-only. See :mod:`ledger` for the record format and overlay
rules, and :mod:`snapshot` for turning a log back into folders of video.

Three things follow from that, and they are the point of the design:

* A mis-key costs a keystroke, not a file operation. Undo appends a correcting
  record instead of moving footage back.
* Classifying is instant. The old implementation had to flush annotations and
  stop the decoder before every keypress so the file could be moved out from
  under it; nothing holds a lock on a file that stays put.
* Two passes over the same corpus can disagree. Each run writes its own log, and
  they are replayed oldest-first with later winning, so a second pass corrects a
  first without either being rewritten.

**One keybinding moved.** ``Ctrl+Z`` now undoes an *annotation* edit, because
there are two independent histories and annotation edits are far more frequent.
Undo of a classification is ``U``. Both appear in Help ▸ Keyboard Shortcuts,
which is generated from the command registry and so cannot go stale.
"""

from __future__ import annotations

import csv
from pathlib import Path

from PySide6.QtWidgets import QDialog, QFileDialog, QInputDialog, QMessageBox

from ....app.dialogs import show_html
from ....core.commands import Command
from ....core.errors import VidTriageError
from ....core.logging import attach_file_log, get_logger
from ...api import Plugin, PluginContext
from .config import find_session_for_input, load_last_session, parse_classes
from .explorer import FileExplorerWidget
from .models import MAX_CLASSES, ClassEntry, TriageConfig, VideoItem
from .session import Session
from .snapshot import plan_snapshot, write_snapshot
from .wizard import SetupWizard

__all__ = ["PLUGIN", "TriagePlugin"]

_log = get_logger(__name__)


class TriagePlugin(Plugin):
    id = "triage"
    name = "Video Triage"
    description = "Classify whole videos into folders with the number keys"
    default_enabled = True

    def __init__(self) -> None:
        super().__init__()
        self.session: Session | None = None
        self._ctx: PluginContext | None = None
        self._explorer: FileExplorerWidget | None = None
        self._order: list[VideoItem] = []
        self._class_command_ids: list[str] = []
        self._suppress_library_sync = False

    # ── lifecycle ───────────────────────────────────────────────────────

    def activate(self, ctx: PluginContext) -> None:
        self._ctx = ctx

        ctx.add_panel(
            id="triage.explorer", title="Files", area="left",
            visible_by_default=True, shortcut="E",
            factory=self._build_explorer,
        )
        self._register_static_commands(ctx)
        ctx.connect(ctx.app.library.current_changed, self._on_library_current_changed)
        ctx.connect(ctx.app.playback.reached_end, self._on_video_ended)

        session = self._session_from_launch(ctx) or self._session_from_history(ctx)
        if session is not None:
            self._start_session(session)
        else:
            ctx.app.status("Triage: use File ▸ Triage Session… to pick directories", 8000)

    def _session_from_launch(self, ctx: PluginContext) -> Session | None:
        """A session for the directory named with ``-i``, if one was.

        The flag is documented as choosing a triage session, so it has to reach
        here — otherwise it only fills the media library, the explorer keeps
        showing whatever ran last, and the two disagree about what is on screen.
        """
        input_dir = ctx.app.launch_options.get("triage_input")
        if input_dir is None or not Path(input_dir).is_dir():
            return None

        known = find_session_for_input(input_dir)
        output_dir = (
            ctx.app.launch_options.get("triage_output")
            or (known.output_dir if known else None)
            or Path(input_dir).parent / f"{Path(input_dir).name}_triage"
        )
        return Session(
            Path(input_dir), Path(output_dir), known.classes if known else [],
            logs=ctx.app.launch_options.get("triage_logs"),
        )

    def _session_from_history(self, ctx: PluginContext) -> Session | None:
        last = load_last_session()
        if not (last.is_complete and last.input_dir and last.input_dir.is_dir()):
            return None
        return Session(
            last.input_dir, last.output_dir, last.classes,
            logs=ctx.app.launch_options.get("triage_logs"),
        )

    def deactivate(self) -> None:
        self._clear_class_commands()
        self.session = None
        self._explorer = None
        self._order = []
        self._ctx = None

    def _build_explorer(self) -> FileExplorerWidget:
        self._explorer = FileExplorerWidget()
        self._explorer.file_selected.connect(self._on_explorer_selected)
        self._refresh_explorer()
        return self._explorer

    # ── commands ────────────────────────────────────────────────────────

    def _register_static_commands(self, ctx: PluginContext) -> None:
        ctx.add_command(
            id="triage.setup", title="Triage Session…", shortcut="Ctrl+T",
            menu="File", section="0", order=30, handler=self._open_wizard,
            description="Choose the input/output directories and class list",
        )
        ctx.add_command(
            id="triage.undo", title="Undo Classification", shortcut="U",
            menu="Edit", section="2", order=10, handler=self._undo,
            is_enabled=lambda: bool(self.session and self.session.can_undo),
        )
        ctx.add_command(
            id="triage.error", title="Move To _errors", shortcut="X",
            menu="Edit", section="2", order=20, handler=self._mark_error,
            is_enabled=self._has_current,
        )
        ctx.add_command(
            id="triage.skip", title="Skip To Next Pending", shortcut="S",
            menu="Edit", section="2", order=30, handler=self._skip,
            is_enabled=lambda: bool(self.session and self.session.pending),
        )
        ctx.add_command(
            id="triage.classes", title="Change Classes…", menu="Edit", section="2",
            order=40, handler=self._change_classes, is_enabled=lambda: self.session is not None,
        )
        ctx.add_command(
            id="triage.focus", title="Switch Pending / Classified", shortcut="Tab",
            menu="View", section="2", handler=self._toggle_list_focus,
            is_enabled=lambda: self._explorer is not None,
        )
        ctx.add_command(
            id="triage.search", title="Find In Files", shortcut="Ctrl+F",
            menu="View", section="2", order=5, handler=self._focus_search,
            is_enabled=lambda: self.session is not None,
            description="Filter the file panel by name or class · Esc clears",
        )
        ctx.add_command(
            id="triage.summary", title="Triage Summary", menu="View", section="8",
            handler=self._show_summary, is_enabled=lambda: self.session is not None,
        )
        ctx.add_command(
            id="triage.export", title="Export Classifications…", menu="File", section="1",
            order=40, handler=self._export_classifications,
            is_enabled=lambda: self.session is not None,
            description="CSV of every video and the class it was filed under",
        )
        ctx.add_command(
            id="triage.snapshot", title="Snapshot To Class Folders…", shortcut="Ctrl+Shift+E",
            menu="File", section="1", order=50, handler=self._write_snapshot,
            is_enabled=lambda: bool(self.session and self.session.classified),
            description="Copy every classified video into <outdir>/<class>/",
        )

    def _sync_class_commands(self) -> None:
        """One command per class, so 1–9 are bound to whatever the user defined."""
        ctx = self._ctx
        if ctx is None:
            return
        self._clear_class_commands()
        if self.session is None:
            return

        for entry in self.session.classes:
            command = Command(
                id=f"triage.classify.{entry.key}",
                title=f"[{entry.key}]  {entry.name}",
                shortcut=entry.key,
                menu="Edit/Classify As",
                order=int(entry.key) if entry.key.isdigit() else 99,
                owner=self.id,
                handler=lambda e=entry: self._classify(e),
                is_enabled=self._has_current,
            )
            ctx.app.commands.add(command, replace=True)
            self._class_command_ids.append(command.id)

        # Unused digits create a class on the fly, matching the old behaviour.
        used = {c.key for c in self.session.classes}
        for digit in (str(i) for i in range(1, MAX_CLASSES + 1)):
            if digit in used:
                continue
            command = Command(
                id=f"triage.newclass.{digit}",
                title=f"[{digit}]  New class…",
                shortcut=digit,
                menu="Edit/Classify As",
                section="9",
                order=int(digit),
                owner=self.id,
                handler=lambda d=digit: self._prompt_new_class(d),
                is_enabled=self._has_current,
            )
            ctx.app.commands.add(command, replace=True)
            self._class_command_ids.append(command.id)

    def _clear_class_commands(self) -> None:
        if self._ctx is None:
            return
        for command_id in self._class_command_ids:
            self._ctx.app.commands.unregister(command_id)
        self._class_command_ids.clear()

    # ── session ─────────────────────────────────────────────────────────

    def _open_wizard(self) -> None:
        ctx = self._ctx
        if ctx is None:
            return
        wizard = SetupWizard(
            ctx.app.window,
            prefill_input=self.session.input_dir if self.session else None,
            prefill_output=self.session.output_dir if self.session else None,
        )
        if wizard.exec() != QDialog.DialogCode.Accepted or wizard.result_config is None:
            return
        config: TriageConfig = wizard.result_config
        self._start_session(
            Session(
                config.input_dir, config.output_dir, config.classes,
                logs=wizard.result_logs,
            ),
        )

    def _start_session(self, session: Session) -> None:
        ctx = self._ctx
        if ctx is None:
            return

        attach_file_log(session.log_dir)
        session.load()
        self.session = session
        self._sync_class_commands()
        self._sync_library(prefer_first_pending=True)

        overlaid = f" · {len(session.logs)} log(s) replayed" if session.logs else ""
        ctx.app.status(
            f"Triage: {len(session.pending)} pending, "
            f"{len(session.classified)} classified{overlaid}",
            6000,
        )

    # ── library <-> session ─────────────────────────────────────────────

    def _sync_library(self, prefer_first_pending: bool = False) -> None:
        """Rebuild the playlist from session state, pending first."""
        ctx = self._ctx
        if ctx is None or self.session is None:
            return

        self._order = [*self.session.pending, *self.session.classified]
        paths = [item.original_path for item in self._order]

        self._suppress_library_sync = True
        try:
            ctx.app.library.set_items(paths, keep_current=not prefer_first_pending)
        finally:
            self._suppress_library_sync = False

        self._refresh_explorer()
        self._on_library_current_changed(ctx.app.library.current)

    def _refresh_explorer(self) -> None:
        if self._explorer is not None and self.session is not None:
            self._explorer.set_items(self.session.pending, self.session.classified)

    @property
    def current_item(self) -> VideoItem | None:
        """The video on screen, resolved by path rather than by list position.

        ``_order`` mirrors the library only for as long as nothing else sets the
        playlist, and the library is shared — a command-line path list or
        another plugin can replace it. Indexing into ``_order`` with the
        library's index then silently returns *a different video than the one
        being shown*, and a keystroke would file a decision against the wrong
        file. Matching on the path cannot do that: worst case it finds nothing,
        the classify commands disable themselves, and the mismatch is visible.
        """
        ctx = self._ctx
        if ctx is None or self.session is None:
            return None
        current = ctx.app.library.current
        return self.session.find_by_path(current) if current is not None else None

    def _has_current(self) -> bool:
        return self.current_item is not None

    def _on_library_current_changed(self, _path: Path | None) -> None:
        """Keep the explorer cursor on whatever the library is showing."""
        if self._suppress_library_sync or self._explorer is None or self.session is None:
            return
        item = self.current_item
        if item is not None:
            self._explorer.select_item(item)

    def _on_explorer_selected(self, item: VideoItem) -> None:
        ctx = self._ctx
        if ctx is None or self.session is None:
            return
        if item in self._order:
            ctx.app.library.set_index(self._order.index(item))

    def _toggle_list_focus(self) -> None:
        if self._explorer is not None:
            self._explorer.toggle_focus()

    def _focus_search(self) -> None:
        """Show the file panel if it is hidden, then put the cursor in the filter."""
        ctx = self._ctx
        if ctx is None or ctx.app.window is None:
            return
        ctx.app.window.set_panel_visible("triage.explorer", True)
        if self._explorer is not None:
            self._explorer.focus_search()

    def _on_video_ended(self, _source_id: str) -> None:
        ctx = self._ctx
        if ctx is not None and ctx.app.playback.end_mode.value == "next":
            ctx.app.library.next()

    # ── actions ─────────────────────────────────────────────────────────

    def _classify(self, entry: ClassEntry) -> None:
        item = self.current_item
        ctx = self._ctx
        if item is None or ctx is None or self.session is None:
            return

        was_pending = item.is_pending
        self.session.classify(item, entry)
        ctx.app.status(f"{item.name} → {entry.name}", 2500)
        self._sync_library()
        if was_pending:
            self._go_to_next_pending()

    def _mark_error(self) -> None:
        item = self.current_item
        ctx = self._ctx
        if item is None or ctx is None or self.session is None or item.is_error:
            return

        self.session.mark_error(item)
        ctx.app.status(f"{item.name} → _errors", 2500)
        self._sync_library()
        self._go_to_next_pending()

    def _undo(self) -> None:
        ctx = self._ctx
        if ctx is None or self.session is None:
            return

        item = self.session.undo_last()
        self._sync_library()
        if item is None:
            ctx.app.status("Nothing to undo", 2000)
            return

        if item in self._order:
            ctx.app.library.set_index(self._order.index(item))
        ctx.app.status(f"Undid: {item.name}", 2500)

    def _skip(self) -> None:
        self._go_to_next_pending()

    def _go_to_next_pending(self) -> None:
        ctx = self._ctx
        if ctx is None or self.session is None:
            return
        pending = self.session.pending
        if not pending:
            ctx.app.status("All videos classified", 6000)
            return

        current = self.current_item
        start = self._order.index(current) + 1 if current in self._order else 0
        ordered = [*range(start, len(self._order)), *range(0, start)]
        for index in ordered:
            if self._order[index].is_pending:
                ctx.app.library.set_index(index)
                return

    # ── classes ─────────────────────────────────────────────────────────

    def _prompt_new_class(self, key: str) -> None:
        ctx = self._ctx
        if ctx is None or self.session is None or self.current_item is None:
            return
        name, accepted = QInputDialog.getText(
            ctx.app.window, "New Class", f"Name for key [{key}]:", text=f"class_{key}",
        )
        if not accepted or not name.strip():
            return
        entry = self.session.add_class(key, name.strip())
        self._sync_class_commands()
        self._classify(entry)

    def _change_classes(self) -> None:
        ctx = self._ctx
        if ctx is None or self.session is None:
            return
        text, accepted = QInputDialog.getMultiLineText(
            ctx.app.window, "VidTriage — Classes",
            f"One class per line, keys assigned 1-{MAX_CLASSES}:",
            "\n".join(c.name for c in self.session.classes),
        )
        if not accepted:
            return
        entries, errors = parse_classes(text)
        if errors:
            QMessageBox.warning(
                ctx.app.window, "Validation Error", "\n".join(f"• {e}" for e in errors),
            )
            return
        self.session.set_classes(entries)
        self._sync_class_commands()
        ctx.app.status(f"{len(entries)} classes", 3000)

    # ── reporting ───────────────────────────────────────────────────────

    def _show_summary(self) -> None:
        from collections import Counter

        ctx = self._ctx
        if ctx is None or self.session is None:
            return

        pending = self.session.pending
        classified = self.session.classified
        total = len(pending) + len(classified)
        counts = Counter(
            item.class_name for item in classified if not item.is_error and item.class_name
        )
        errors = sum(1 for item in classified if item.is_error)

        def row(name: str, count: int, color: str = "") -> str:
            percent = f"{count / total * 100:.1f}%" if total else "0%"
            style = f" style='color:{color};'" if color else ""
            return (
                f"<tr><td{style}>{name}</td>"
                f"<td align='right'>{count}</td>"
                f"<td align='right'>{percent}</td></tr>"
            )

        rows = "".join(row(name, n) for name, n in counts.most_common())
        if errors:
            rows += row("_errors", errors, "#ef5350")

        progress = f"{len(classified) / total * 100:.1f}%" if total else "0%"
        show_html(
            ctx.app.window, "Triage Summary",
            f"<h2>Triage Summary</h2>"
            f"<p><b>{len(classified)}</b> / {total} classified ({progress}) · "
            f"<b>{len(pending)}</b> pending</p>"
            f"<table width='100%' cellpadding='4'>"
            f"<tr><th align='left'>Class</th><th align='right'>Count</th>"
            f"<th align='right'>%</th></tr>{rows}</table>",
            (440, 420),
        )

    def _export_classifications(self) -> None:
        ctx = self._ctx
        if ctx is None or self.session is None:
            return
        default = str(self.session.output_dir / "classifications.csv")
        path, _filter = QFileDialog.getSaveFileName(
            ctx.app.window, "Export Classifications", default, "CSV Files (*.csv)",
        )
        if not path:
            return

        rows = [
            (item.name, item.class_name or "unclassified", str(item.original_path))
            for item in [*self.session.pending, *self.session.classified]
        ]
        try:
            with Path(path).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["video", "class", "path"])
                writer.writerows(rows)
        except OSError as exc:
            QMessageBox.warning(ctx.app.window, "Export failed", str(exc))
            return
        ctx.app.status(f"Exported {len(rows)} row(s) to {Path(path).name}", 6000)

    def _write_snapshot(self) -> None:
        """Copy the log's verdict out as folders of video."""
        ctx = self._ctx
        if ctx is None or self.session is None:
            return

        chosen = QFileDialog.getExistingDirectory(
            ctx.app.window, "Snapshot into a new, empty directory",
            str(self.session.output_dir),
        )
        if not chosen:
            return

        target = Path(chosen)
        items = self.session.classified
        plan = plan_snapshot(items, target)
        if plan.collisions:
            QMessageBox.warning(
                ctx.app.window, "Duplicate filenames",
                f"{len(plan.collisions)} filename(s) appear more than once, so a "
                f"snapshot cannot name them apart. Nothing was written.\n\n"
                + "\n".join(Path(d).name for d in list(plan.collisions)[:10]),
            )
            return

        link = QMessageBox.question(
            ctx.app.window, "Hardlink instead of copying?",
            f"Snapshot {len(plan.pairs)} video(s) into\n{target}\n\n"
            f"Hardlinking is instant and uses no extra disk, but only works on "
            f"the same filesystem — it falls back to copying per file.\n\n"
            f"Yes to hardlink, No to copy.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
        )
        if link == QMessageBox.StandardButton.Cancel:
            return

        try:
            result = write_snapshot(
                items, target, link=link == QMessageBox.StandardButton.Yes,
            )
        except VidTriageError as exc:
            QMessageBox.warning(ctx.app.window, "Snapshot failed", str(exc))
            return

        if result.warnings:
            show_html(
                ctx.app.window, "Snapshot warnings",
                f"<h3>{result.summary()}</h3><ul>"
                + "".join(f"<li>{warning}</li>" for warning in result.warnings)
                + "</ul>",
                (560, 380),
            )
        ctx.app.status(result.summary(), 8000)


PLUGIN = TriagePlugin
