"""Setup dialog: pick the directories, the class list, and the log stack.

Validation is the substance here. Input and output must exist, be readable and
writable, and must not overlap in either direction — a snapshot written inside
the input directory would be rescanned as source material on the next launch,
and the session would eat its own results.

The log stack is the other half. Every run appends its decisions to a new log,
so a corpus accumulates one per pass; the list here chooses which of them apply
and in what order. Order is the whole meaning of the list — it is replayed top
to bottom and the last decision for a file wins, so moving a log down makes it
override the ones above it.
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ....core.logging import get_logger
from ....view.theme import current_theme
from .config import load_all_sessions, parse_classes, save_session
from .io_ops import discover_videos, scan_output_subfolders
from .ledger import LOG_SUFFIX, default_log_dir, discover_logs, summarise
from .models import ERRORS_FOLDER, TriageConfig

__all__ = ["SetupWizard"]

_log = get_logger(__name__)

_DEBOUNCE_MS = 300
_NEW_SESSION = "+ New Session"


class SetupWizard(QDialog):
    """Choose or create a triage session."""

    def __init__(
        self,
        parent: QWidget | None = None,
        prefill_input: Path | None = None,
        prefill_output: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("VidTriage — Setup")
        self.setMinimumSize(560, 700)
        self.result_config: TriageConfig | None = None
        #: Logs to replay, in override order. ``None`` means auto-discover.
        self.result_logs: list[Path] | None = None

        self._sessions = load_all_sessions()
        self._last_output_text = ""
        self._last_input_text = ""

        layout = QVBoxLayout(self)

        session_row = QHBoxLayout()
        session_row.addWidget(QLabel("Session:"))
        self._session_combo = QComboBox()
        for config in self._sessions:
            self._session_combo.addItem(_session_label(config))
        self._session_combo.addItem(_NEW_SESSION)
        session_row.addWidget(self._session_combo, stretch=1)
        layout.addLayout(session_row)
        layout.addSpacing(8)

        self._input_edit = self._add_dir_row(layout, "Input directory (videos to classify):")
        layout.addSpacing(8)
        self._output_edit = self._add_dir_row(layout, "Output directory (classified videos):")

        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet(
            f"color: {current_theme().info_fg}; font-size: 12px; padding: 4px 0;",
        )
        layout.addWidget(self._info_label)

        layout.addSpacing(8)
        layout.addWidget(QLabel("Classes — one per line, keys assigned 1-9:"))
        self._build_class_widgets(layout)

        layout.addSpacing(8)
        self._build_log_widgets(layout)

        layout.addStretch()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Launch")
        buttons.accepted.connect(self._launch)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._update_info)
        self._input_edit.textChanged.connect(self._debounce.start)
        self._output_edit.textChanged.connect(self._debounce.start)

        self._select_initial_session(prefill_input, prefill_output)
        self._session_combo.currentIndexChanged.connect(self._on_session_changed)

    # ── construction helpers ────────────────────────────────────────────

    def _add_dir_row(self, layout: QVBoxLayout, label: str) -> QLineEdit:
        layout.addWidget(QLabel(label))
        row = QHBoxLayout()
        edit = QLineEdit()
        row.addWidget(edit)
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._browse(edit))
        row.addWidget(browse)
        layout.addLayout(row)
        return edit

    def _build_class_widgets(self, layout: QVBoxLayout) -> None:
        self._class_edit = QTextEdit()
        self._class_edit.setPlaceholderText("cat\ndog\nbird\nskip")
        self._class_edit.setMaximumHeight(130)
        self._class_edit.textChanged.connect(self._populate_table)
        layout.addWidget(self._class_edit)

        self._class_table = QTableWidget(0, 2)
        self._class_table.setHorizontalHeaderLabels(["Key", "Class"])
        self._class_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Fixed,
        )
        self._class_table.setColumnWidth(0, 60)
        self._class_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch,
        )
        self._class_table.verticalHeader().setVisible(False)
        self._class_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._class_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._class_table.setMaximumHeight(150)
        layout.addWidget(self._class_table)

    def _build_log_widgets(self, layout: QVBoxLayout) -> None:
        layout.addWidget(QLabel("Logs to replay — applied top to bottom, last one wins:"))

        row = QHBoxLayout()
        self._log_list = QListWidget()
        self._log_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._log_list.setMaximumHeight(120)
        row.addWidget(self._log_list, stretch=1)

        buttons = QVBoxLayout()
        for label, slot in (
            ("▲", lambda: self._move_log(-1)),
            ("▼", lambda: self._move_log(1)),
            ("Add…", self._add_log),
            ("Remove", self._remove_log),
        ):
            button = QPushButton(label)
            button.setMaximumWidth(80)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        buttons.addStretch()
        row.addLayout(buttons)
        layout.addLayout(row)

        self._log_hint = QLabel("")
        self._log_hint.setWordWrap(True)
        self._log_hint.setStyleSheet(
            f"color: {current_theme().info_fg}; font-size: 11px; padding: 2px 0;",
        )
        layout.addWidget(self._log_hint)

    def _refresh_log_list(self, input_dir: Path) -> None:
        """Reload the discovered stack for a corpus, oldest first."""
        self._log_list.clear()
        logs = discover_logs(default_log_dir(input_dir))
        for path in logs:
            self._log_list.addItem(_log_item(path))
        self._log_hint.setText(
            f"{len(logs)} previous run(s) in {default_log_dir(input_dir)}. "
            "This session appends to a new log of its own."
            if logs else
            "No previous runs for this input directory — this session starts a first log.",
        )

    def _selected_logs(self) -> list[Path]:
        return [
            Path(self._log_list.item(i).data(Qt.ItemDataRole.UserRole))
            for i in range(self._log_list.count())
        ]

    def _move_log(self, delta: int) -> None:
        row = self._log_list.currentRow()
        target = row + delta
        if row < 0 or not 0 <= target < self._log_list.count():
            return
        item = self._log_list.takeItem(row)
        self._log_list.insertItem(target, item)
        self._log_list.setCurrentRow(target)

    def _add_log(self) -> None:
        start = self._input_edit.text().strip() or str(Path.home())
        chosen, _filter = QFileDialog.getOpenFileNames(
            self, "Add triage logs", start, f"Triage logs (*{LOG_SUFFIX});;All files (*)",
        )
        existing = {str(p) for p in self._selected_logs()}
        for path in chosen:
            if path not in existing:
                self._log_list.addItem(_log_item(Path(path)))

    def _remove_log(self) -> None:
        row = self._log_list.currentRow()
        if row >= 0:
            self._log_list.takeItem(row)

    def _select_initial_session(
        self, prefill_input: Path | None, prefill_output: Path | None,
    ) -> None:
        index = len(self._sessions)  # "+ New Session"
        if prefill_output is not None:
            resolved = Path(prefill_output).resolve()
            for i, session in enumerate(self._sessions):
                if session.output_dir == resolved:
                    index = i
                    break
        elif self._sessions:
            index = 0

        self._session_combo.setCurrentIndex(index)
        self._apply_session(index)

        if prefill_input is not None:
            self._input_edit.setText(str(prefill_input))
        if prefill_output is not None and index == len(self._sessions):
            self._output_edit.setText(str(prefill_output))

        self._last_output_text = self._output_edit.text().strip()
        self._debounce.stop()
        self._update_info()

    # ── session switching ───────────────────────────────────────────────

    def _on_session_changed(self, index: int) -> None:
        self._apply_session(index)
        self._last_output_text = self._output_edit.text().strip()
        self._debounce.stop()
        self._update_info()

    def _apply_session(self, index: int) -> None:
        if index >= len(self._sessions):
            self._input_edit.clear()
            self._output_edit.clear()
            self._class_edit.clear()
            return
        config = self._sessions[index]
        self._input_edit.setText(str(config.input_dir) if config.input_dir else "")
        self._output_edit.setText(str(config.output_dir) if config.output_dir else "")
        self._class_edit.setPlainText("\n".join(c.name for c in config.classes))

    # ── live info ───────────────────────────────────────────────────────

    def _browse(self, target: QLineEdit) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Select Directory", target.text())
        if chosen:
            target.setText(chosen)

    def _populate_table(self) -> None:
        entries, _errors = parse_classes(self._class_edit.toPlainText())
        self._class_table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            key_item = QTableWidgetItem(entry.key)
            key_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._class_table.setItem(row, 0, key_item)
            self._class_table.setItem(row, 1, QTableWidgetItem(entry.name))

    def _update_info(self) -> None:
        parts: list[str] = []

        input_text = self._input_edit.text().strip()
        output_text = self._output_edit.text().strip()
        output_changed = output_text != self._last_output_text
        self._last_output_text = output_text

        if input_text and Path(input_text).is_dir():
            parts.append(f"Input: {len(discover_videos(Path(input_text)))} videos")
            if input_text != self._last_input_text:
                self._last_input_text = input_text
                self._refresh_log_list(Path(input_text))

        if output_text and Path(output_text).is_dir():
            counts: list[tuple[str, int]] = []
            errors = 0
            folder_classes: list[str] = []
            for name, videos in scan_output_subfolders(Path(output_text)):
                if name == ERRORS_FOLDER:
                    errors = len(videos)
                else:
                    counts.append((name, len(videos)))
                    folder_classes.append(name)

            if counts or errors:
                summary = [f"{name}: {n}" for name, n in counts]
                if errors:
                    summary.append(f"errors: {errors}")
                total = sum(n for _n, n in counts) + errors
                parts.append(f"Output: {total} videos ({', '.join(summary)})")
            else:
                parts.append("Output: no previous session")

            # Adopt the class list implied by existing folders, but only when the
            # user has just pointed at a different output directory.
            if folder_classes and output_changed and not self._class_edit.toPlainText().strip():
                self._class_edit.setPlainText("\n".join(folder_classes))

        self._info_label.setText("  |  ".join(parts))

    # ── validation ──────────────────────────────────────────────────────

    def _validate_dirs(self) -> tuple[Path | None, Path | None, list[str]]:
        errors: list[str] = []
        input_text = self._input_edit.text().strip()
        output_text = self._output_edit.text().strip()

        if not input_text:
            errors.append("Input directory is empty.")
        elif not Path(input_text).is_dir():
            errors.append("Input directory does not exist.")

        if not output_text:
            errors.append("Output directory is empty.")
        elif not Path(output_text).is_dir():
            reply = QMessageBox.question(
                self, "Create Directory?",
                f"Output directory does not exist:\n{output_text}\n\nCreate it?",
            )
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    Path(output_text).mkdir(parents=True, exist_ok=True)
                except OSError as exc:
                    errors.append(f"Could not create output directory: {exc}")
            else:
                errors.append("Output directory does not exist.")

        if errors:
            return None, None, errors

        input_dir = Path(input_text).resolve()
        output_dir = Path(output_text).resolve()

        if input_dir == output_dir:
            errors.append("Input and output directories must be different.")
        elif output_dir.is_relative_to(input_dir):
            errors.append(
                "Output cannot be inside the input directory — a snapshot written "
                "there would be rediscovered as source material on the next launch.",
            )
        elif input_dir.is_relative_to(output_dir):
            errors.append("Input directory cannot be inside the output directory.")

        if not errors:
            if not os.access(input_dir, os.R_OK):
                errors.append("Input directory is not readable.")
            if not os.access(output_dir, os.W_OK):
                errors.append("Output directory is not writable.")

        return input_dir, output_dir, errors

    def _launch(self) -> None:
        input_dir, output_dir, errors = self._validate_dirs()
        entries, class_errors = parse_classes(self._class_edit.toPlainText())
        errors.extend(class_errors)

        if errors:
            QMessageBox.warning(self, "Validation Error", "\n".join(f"• {e}" for e in errors))
            return

        config = TriageConfig(input_dir=input_dir, output_dir=output_dir, classes=entries)
        save_session(config)
        self.result_config = config
        self.result_logs = self._selected_logs()
        self.accept()


def _log_item(path: Path) -> QListWidgetItem:
    """One row: the log's name, how many decisions it holds, and when."""
    count, first, last = summarise(path)
    span = f"{first[:16]} → {last[:16]}" if first else "empty"
    item = QListWidgetItem(f"{path.name}   ({count} decision(s), {span})")
    item.setData(Qt.ItemDataRole.UserRole, str(path))
    item.setToolTip(str(path))
    return item


def _session_label(config: TriageConfig) -> str:
    if not config.output_dir:
        return "Unnamed"
    source = config.input_dir.name if config.input_dir else "?"
    return f"{config.output_dir.name}  ({source})"
