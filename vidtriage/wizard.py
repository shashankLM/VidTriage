from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QFileDialog, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMessageBox, QPushButton, QStackedWidget,
    QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from .config import load_config, save_config, parse_classes
from .io_ops import discover_videos, load_log
from .models import AppConfig


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
        self.setMinimumSize(520, 520)
        self.result_config: AppConfig | None = None
        self._prev_classes: list[str] | None = None

        saved = load_config()
        layout = QVBoxLayout(self)

        self._input_edit = self._add_dir_row(
            layout, "Input directory (videos to classify):",
            prefill_input or saved.input_dir,
        )
        layout.addSpacing(8)
        self._output_edit = self._add_dir_row(
            layout, "Output directory (classified videos):",
            prefill_output or saved.output_dir,
        )

        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("color: #aaa; font-size: 12px; padding: 4px 0;")
        layout.addWidget(self._info_label)

        layout.addSpacing(8)
        layout.addWidget(QLabel("Classes (keys auto-assigned 1-9):"))
        self._build_class_widgets(layout, saved.classes)

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

        self._input_edit.textChanged.connect(self._update_info)
        self._output_edit.textChanged.connect(self._update_info)
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
            user_classes = [c.name for c in saved_classes if c.name != c.key]
            self._class_edit.setPlainText("\n".join(user_classes))

        self._populate_table()
        self._class_stack.setCurrentIndex(0)
        layout.addWidget(self._class_stack)

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
        self._prev_classes = None

        input_text = self._input_edit.text().strip()
        if input_text and Path(input_text).is_dir():
            videos = discover_videos(Path(input_text))
            parts.append(f"Input: {len(videos)} videos found")

        output_text = self._output_edit.text().strip()
        if output_text and Path(output_text).is_dir():
            log = load_log(Path(output_text))
            if log:
                classified = self._reconcile_log(log)
                counts = Counter(r.get("class_name", "") for r in classified.values())
                summary = ", ".join(f"{name}: {n}" for name, n in counts.most_common())
                parts.append(f"Previous session: {len(classified)} classified ({summary})")

                prev_classes = list(dict.fromkeys(
                    r.get("class_name", "") for r in classified.values()
                    if r.get("class_name", "") and r.get("action") == "classify"
                ))
                if prev_classes:
                    self._prev_classes = prev_classes
            else:
                parts.append("Output: no previous session")

        self._info_label.setText("  |  ".join(parts) if parts else "")

    @staticmethod
    def _reconcile_log(log: list[dict[str, str]]) -> dict[str, dict[str, str]]:
        classified: dict[str, dict[str, str]] = {}
        for row in log:
            src = row.get("source_path", "")
            action = row.get("action", "")
            if action == "undo":
                classified.pop(src, None)
            elif action in ("classify", "error"):
                classified[src] = row
        return classified

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

        if self._prev_classes:
            user_names = [e.name for e in entries if e.name != e.key]
            if user_names != self._prev_classes:
                prev_str = ", ".join(self._prev_classes)
                reply = QMessageBox.question(
                    self,
                    "Previous session found",
                    f"The output directory has a previous session with classes:\n"
                    f"{prev_str}\n\n"
                    f"Use previous classes instead?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self._class_edit.setPlainText("\n".join(self._prev_classes))
                    entries, _ = parse_classes("\n".join(self._prev_classes))

        config = AppConfig(
            input_dir=Path(input_text).resolve(),
            output_dir=Path(output_text).resolve(),
            classes=entries,
        )
        save_config(config)
        self.result_config = config
        self.accept()
