from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QFileDialog, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMessageBox, QPushButton, QStackedWidget,
    QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from .config import load_all_sessions, save_config, parse_classes
from .io_ops import discover_videos, scan_output_subfolders
from .models import ERRORS_FOLDER, AppConfig


class _FocusOutEdit(QTextEdit):
    def __init__(self, on_focus_out: callable) -> None:
        super().__init__()
        self._on_focus_out = on_focus_out

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        self._on_focus_out()


class SetupWizard(QDialog):
    def __init__(
        self,
        parent: QWidget | None = None,
        prefill_input: Path | None = None,
        prefill_output: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("VidTriage — Setup")
        self.setMinimumSize(520, 540)
        self.result_config: AppConfig | None = None
        self._last_output_text: str = ""

        self._sessions = load_all_sessions()
        layout = QVBoxLayout(self)

        # --- Session dropdown ---
        session_row = QHBoxLayout()
        session_row.addWidget(QLabel("Session:"))
        self._session_combo = QComboBox()
        for cfg in self._sessions:
            self._session_combo.addItem(self._session_label(cfg))
        self._session_combo.addItem("+ New Session")
        session_row.addWidget(self._session_combo, stretch=1)
        layout.addLayout(session_row)
        layout.addSpacing(8)

        # --- Directory rows ---
        self._input_edit = self._add_dir_row(
            layout, "Input directory (videos to classify):", None,
        )
        layout.addSpacing(8)
        self._output_edit = self._add_dir_row(
            layout, "Output directory (classified videos):", None,
        )

        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        from .theme import current_theme
        self._info_label.setStyleSheet(f"color: {current_theme().info_fg}; font-size: 12px; padding: 4px 0;")
        layout.addWidget(self._info_label)

        layout.addSpacing(8)
        layout.addWidget(QLabel("Classes (keys auto-assigned 1-9):"))
        self._build_class_widgets(layout, [])

        layout.addStretch()

        btn_layout = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_launch = QPushButton("Launch")
        btn_launch.clicked.connect(self._launch)
        btn_layout.addWidget(btn_cancel)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_launch)
        layout.addLayout(btn_layout)

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(300)
        self._debounce_timer.timeout.connect(self._update_info)

        self._input_edit.textChanged.connect(lambda: self._debounce_timer.start())
        self._output_edit.textChanged.connect(lambda: self._debounce_timer.start())

        # --- Initial session selection ---
        initial_index = len(self._sessions)  # "New Session"
        prefill_resolved = Path(prefill_output).resolve() if prefill_output else None
        if prefill_resolved:
            for i, sess in enumerate(self._sessions):
                if sess.output_dir == prefill_resolved:
                    initial_index = i
                    break
        elif self._sessions:
            initial_index = 0

        self._session_combo.setCurrentIndex(initial_index)
        self._on_session_changed(initial_index)

        if initial_index == len(self._sessions):
            if prefill_input:
                self._input_edit.setText(str(prefill_input))
            if prefill_output:
                self._output_edit.setText(str(prefill_output))
        elif prefill_input:
            self._input_edit.setText(str(prefill_input))

        self._last_output_text = self._output_edit.text().strip()
        self._debounce_timer.stop()
        self._update_info()

        self._session_combo.currentIndexChanged.connect(self._on_session_changed)

    # --- Session helpers ---

    @staticmethod
    def _session_label(config: AppConfig) -> str:
        if not config.output_dir:
            return "Unnamed"
        out_name = config.output_dir.name
        in_name = config.input_dir.name if config.input_dir else "?"
        return f"{out_name}  ({in_name})"

    def _on_session_changed(self, index: int) -> None:
        if index < len(self._sessions):
            config = self._sessions[index]
            self._input_edit.setText(str(config.input_dir) if config.input_dir else "")
            self._output_edit.setText(str(config.output_dir) if config.output_dir else "")
            if config.classes:
                self._class_edit.setPlainText("\n".join(c.name for c in config.classes))
            else:
                self._class_edit.setPlainText("")
        else:
            self._input_edit.setText("")
            self._output_edit.setText("")
            self._class_edit.setPlainText("")

        self._populate_table()
        self._last_output_text = self._output_edit.text().strip()
        self._debounce_timer.stop()
        self._update_info()

    # --- UI builders ---

    def _add_dir_row(self, layout: QVBoxLayout, label: str, init: Path | None) -> QLineEdit:
        layout.addWidget(QLabel(label))
        row = QHBoxLayout()
        edit = QLineEdit(str(init) if init else "")
        row.addWidget(edit)
        btn = QPushButton("Browse…")
        btn.clicked.connect(lambda: self._browse(edit))
        row.addWidget(btn)
        layout.addLayout(row)
        return edit

    def _build_class_widgets(self, layout: QVBoxLayout, saved_classes) -> None:
        self._class_stack = QStackedWidget()
        self._class_stack.setMinimumHeight(140)

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
        self._class_table.mousePressEvent = lambda _: self._switch_to_edit()
        self._class_stack.addWidget(self._class_table)

        self._class_edit = _FocusOutEdit(self._switch_to_table)
        self._class_edit.setPlaceholderText("cat\ndog\nbird\nskip")
        self._class_stack.addWidget(self._class_edit)

        if saved_classes:
            user_classes = [c.name for c in saved_classes]
            self._class_edit.setPlainText("\n".join(user_classes))

        self._populate_table()
        self._class_stack.setCurrentIndex(0)
        layout.addWidget(self._class_stack)
        self._class_stack.setEnabled(False)

    # --- Class table/edit toggling ---

    def _populate_table(self) -> None:
        entries, _ = parse_classes(self._class_edit.toPlainText())
        self._class_table.setRowCount(len(entries))
        for i, entry in enumerate(entries):
            key_item = QTableWidgetItem(entry.key)
            key_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._class_table.setItem(i, 0, key_item)
            self._class_table.setItem(i, 1, QTableWidgetItem(entry.name))

    def _switch_to_edit(self) -> None:
        self._class_stack.setCurrentIndex(1)
        self._class_edit.setFocus()

    def _switch_to_table(self) -> None:
        self._populate_table()
        self._class_stack.setCurrentIndex(0)

    # --- Info / browse ---

    def _browse(self, target: QLineEdit) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Directory", target.text())
        if d:
            target.setText(d)

    def _update_info(self) -> None:
        parts: list[str] = []

        input_text = self._input_edit.text().strip()
        output_text = self._output_edit.text().strip()
        input_valid = bool(input_text) and Path(input_text).is_dir()
        output_valid = bool(output_text) and Path(output_text).is_dir()
        output_changed = output_text != self._last_output_text
        self._last_output_text = output_text

        if input_valid:
            videos = discover_videos(Path(input_text))
            parts.append(f"Input: {len(videos)} videos found")

        if output_valid:
            folder_stats: list[tuple[str, int]] = []
            total_classified = 0
            error_count = 0
            folder_classes: list[str] = []

            for name, vids in scan_output_subfolders(Path(output_text)):
                if name == ERRORS_FOLDER:
                    error_count = len(vids)
                else:
                    folder_stats.append((name, len(vids)))
                    total_classified += len(vids)
                    folder_classes.append(name)

            if folder_stats or error_count:
                summary_parts = [f"{name}: {n}" for name, n in folder_stats]
                if error_count:
                    summary_parts.append(f"errors: {error_count}")
                total = total_classified + error_count
                parts.append(f"Output: {total} videos ({', '.join(summary_parts)})")
            else:
                parts.append("Output: no previous session")

            if folder_classes and output_changed:
                self._class_edit.setPlainText("\n".join(folder_classes))
                self._populate_table()

        self._class_stack.setEnabled(bool(input_valid and output_valid))
        self._info_label.setText("  |  ".join(parts) if parts else "")

    # --- Validation ---

    def _validate_dirs(self) -> tuple[str, str, list[str]]:
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
            errors.append("Output directory does not exist.")

        if errors:
            return input_text, output_text, errors

        input_dir = Path(input_text).resolve()
        output_dir = Path(output_text).resolve()

        if input_dir == output_dir:
            errors.append("Input and output directories must be different.")
        elif output_dir.is_relative_to(input_dir):
            errors.append("Output directory cannot be inside the input directory.")
        elif input_dir.is_relative_to(output_dir):
            errors.append("Input directory cannot be inside the output directory.")

        if not errors:
            if not os.access(input_dir, os.R_OK):
                errors.append("Input directory is not readable.")
            if not os.access(output_dir, os.W_OK):
                errors.append("Output directory is not writable.")

        return input_text, output_text, errors

    def _launch(self) -> None:
        input_text, output_text, errors = self._validate_dirs()

        text = self._class_edit.toPlainText()
        entries, class_errors = parse_classes(text)
        errors.extend(class_errors)

        if errors:
            QMessageBox.warning(self, "Validation Error", "\n".join(errors))
            return

        config = AppConfig(
            input_dir=Path(input_text).resolve(),
            output_dir=Path(output_text).resolve(),
            classes=entries,
        )
        save_config(config)
        self.result_config = config
        self.accept()
