"""Setup dialog: pick the directories and the class list.

Validation is the substance here. Input and output must exist, be readable and
writable, and must not overlap in either direction — an output folder nested
inside the input would be rescanned as source material on the next launch, and
the session would eat its own results.
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
        self.setMinimumSize(560, 560)
        self.result_config: TriageConfig | None = None

        self._sessions = load_all_sessions()
        self._last_output_text = ""

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
                "Output cannot be inside the input directory — classified videos "
                "would be rediscovered as source material on the next launch.",
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
        self.accept()


def _session_label(config: TriageConfig) -> str:
    if not config.output_dir:
        return "Unnamed"
    source = config.input_dir.name if config.input_dir else "?"
    return f"{config.output_dir.name}  ({source})"
