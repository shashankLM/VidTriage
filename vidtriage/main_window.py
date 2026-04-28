from __future__ import annotations

from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QActionGroup, QKeyEvent, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QSlider, QSplitter, QMessageBox, QFrame, QDialog, QTextBrowser, QApplication,
    QFileDialog, QTextEdit, QInputDialog,
)

from .models import ClassEntry
from .session import Session
from .player import CvPlayerWidget
from .file_explorer import FileExplorerWidget


def _fmt_time(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


class MainWindow(QMainWindow):
    def __init__(self, session: Session) -> None:
        super().__init__()
        self.setWindowTitle("VidTriage")
        self.resize(1200, 750)

        self._session = session

        self._current_item = None
        self._current_list: str = "pending"
        self._current_row: int = -1
        self._pending_n: int = 0
        self._classified_n: int = 0

        self._build_ui()
        self._sync_explorer()

        QApplication.instance().installEventFilter(self)

        if self._session.pending:
            self._navigate_to("pending", 0)

    # ── UI construction ────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._build_menubar()

        self._splitter = QSplitter(Qt.Orientation.Horizontal)

        self._file_explorer = FileExplorerWidget()
        self._file_explorer.setMinimumWidth(200)
        self._file_explorer.setMaximumWidth(400)
        self._file_explorer.file_selected.connect(self._on_file_selected)
        self._splitter.addWidget(self._file_explorer)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._player = CvPlayerWidget()
        right_layout.addWidget(self._player, stretch=1)

        self._splitter.addWidget(right_panel)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        root_layout.addWidget(self._splitter, stretch=1)

        # --- Controls ---
        controls_frame = QFrame()
        controls_frame.setFrameShape(QFrame.Shape.StyledPanel)
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setContentsMargins(8, 4, 8, 4)

        seek_row = QHBoxLayout()
        self._btn_prev = QPushButton("◀◀ Prev")
        self._btn_prev.clicked.connect(self._prev_file)
        seek_row.addWidget(self._btn_prev)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderReleased.connect(self._on_slider_released)
        self._slider.sliderMoved.connect(self._on_slider_moved)
        seek_row.addWidget(self._slider, stretch=1)

        self._btn_play = QPushButton("▶")
        self._btn_play.setFixedWidth(40)
        self._btn_play.clicked.connect(self._toggle_play)
        seek_row.addWidget(self._btn_play)

        self._btn_next = QPushButton("Next ▶▶")
        self._btn_next.clicked.connect(self._next_file)
        seek_row.addWidget(self._btn_next)
        controls_layout.addLayout(seek_row)

        self._info_label = QLabel("No videos loaded")
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        controls_layout.addWidget(self._info_label)

        root_layout.addWidget(controls_frame)

        # --- Class legend ---
        self._legend_frame = QFrame()
        self._legend_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self._legend_layout = QHBoxLayout(self._legend_frame)
        self._legend_layout.setContentsMargins(8, 6, 8, 6)
        self._rebuild_legend()
        root_layout.addWidget(self._legend_frame)

        self._player.position_changed.connect(self._on_position_changed)
        self._player.duration_changed.connect(self._on_duration_changed)
        self._player.video_ended.connect(self._on_video_ended)
        self._slider_dragging = False
        self._duration = 0.0

    def _build_menubar(self) -> None:
        mb = self.menuBar()

        # ── File ──
        file_menu = mb.addMenu("&File")
        file_menu.addAction("Reopen Setup", self._reopen_setup)
        file_menu.addSeparator()
        file_menu.addAction("Export Annotations", self._export_annotations)
        file_menu.addSeparator()
        act_quit = file_menu.addAction("Quit")
        act_quit.setShortcut(QKeySequence("Ctrl+Q"))
        act_quit.triggered.connect(self.close)

        # ── Edit ──
        edit_menu = mb.addMenu("&Edit")
        act_undo = edit_menu.addAction("Undo")
        act_undo.setShortcut(QKeySequence("Ctrl+Z"))
        act_undo.triggered.connect(self._undo)
        edit_menu.addSeparator()
        edit_menu.addAction("Change Classes", self._change_classes)
        edit_menu.addAction("Skip", self._skip)

        # ── View ──
        view_menu = mb.addMenu("&View")
        self._act_frame_overlay = view_menu.addAction("Frame Number Overlay")
        self._act_frame_overlay.setCheckable(True)
        self._act_frame_overlay.toggled.connect(self._toggle_frame_overlay)

        self._act_explorer = view_menu.addAction("File Explorer")
        self._act_explorer.setCheckable(True)
        self._act_explorer.setChecked(True)
        self._act_explorer.toggled.connect(self._toggle_explorer)

        view_menu.addSeparator()
        act_fullscreen = view_menu.addAction("Fullscreen")
        act_fullscreen.setShortcut(QKeySequence("F11"))
        act_fullscreen.triggered.connect(self._toggle_fullscreen)

        view_menu.addSeparator()
        view_menu.addAction("Summary", self._show_summary)

        # ── Playback ──
        playback_menu = mb.addMenu("&Playback")

        speed_menu = playback_menu.addMenu("Speed")
        self._speed_group = QActionGroup(self)
        for mult in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0):
            act = speed_menu.addAction(f"{mult:g}x")
            act.setCheckable(True)
            act.setData(mult)
            self._speed_group.addAction(act)
            if mult == 1.0:
                act.setChecked(True)
        self._speed_group.triggered.connect(self._on_speed_changed)

        step_menu = playback_menu.addMenu("Frame Step")
        self._step_group = QActionGroup(self)
        for n in (1, 2, 5, 10):
            act = step_menu.addAction(f"{n} frame{'s' if n > 1 else ''}")
            act.setCheckable(True)
            act.setData(n)
            self._step_group.addAction(act)
            if n == 1:
                act.setChecked(True)
        self._step_group.triggered.connect(self._on_step_changed)

        end_menu = playback_menu.addMenu("End Behavior")
        self._end_group = QActionGroup(self)
        for mode, label in (("next", "Next Video"), ("loop", "Loop"), ("stop", "Stop")):
            act = end_menu.addAction(label)
            act.setCheckable(True)
            act.setData(mode)
            self._end_group.addAction(act)
            if mode == "next":
                act.setChecked(True)
        self._end_group.triggered.connect(self._on_end_mode_changed)

        # ── Help ──
        help_menu = mb.addMenu("&Help")
        help_menu.addAction("Keyboard Shortcuts", self._show_help)

    # ── Menu handlers ──────────────────────────────────────

    def _on_speed_changed(self, action) -> None:
        self._player.set_speed(action.data())

    def _on_step_changed(self, action) -> None:
        self._player.set_frame_step_size(action.data())

    def _on_end_mode_changed(self, action) -> None:
        self._player.set_end_mode(action.data())

    def _toggle_frame_overlay(self, checked: bool) -> None:
        self._player.set_show_frame_number(checked)

    def _toggle_explorer(self, checked: bool) -> None:
        self._file_explorer.setVisible(checked)

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _skip(self) -> None:
        pending = self._session.pending
        if not pending:
            return
        if self._current_list == "pending":
            next_row = self._current_row + 1
            if next_row < len(pending):
                self._navigate_to("pending", next_row)
        else:
            self._navigate_to("pending", 0)

    def _on_video_ended(self) -> None:
        if self._player.end_mode == "next":
            prev = self._current_item
            self._next_file()
            if self._current_item is not prev:
                self._player.play()
        self._btn_play.setText("▶" if self._player.is_paused() else "⏸")

    # ── Explorer sync ──────────────────────────────────────

    def _sync_explorer(self) -> None:
        self._file_explorer.set_items(self._session.pending, self._session.classified)

    # ── Navigation ─────────────────────────────────────────

    def _navigate_to(self, list_name: str, row: int) -> None:
        items = self._session.pending if list_name == "pending" else self._session.classified
        if not items or row < 0 or row >= len(items):
            return

        self._current_list = list_name
        self._current_row = row
        self._current_item = items[row]

        if list_name == "pending":
            self._file_explorer.select_pending(row)
        else:
            self._file_explorer.select_classified(row)

        path = self._session.playback_path_of(self._current_item)
        if path.exists():
            self._player.load(path)
        else:
            self._player.show_error("File not found")

        self._btn_play.setText("▶" if self._player.is_paused() else "⏸")
        self._update_info()

    def _on_file_selected(self, list_name: str, row: int) -> None:
        self._navigate_to(list_name, row)

    def _next_file(self) -> None:
        active = self._file_explorer.get_active_list()
        row = self._file_explorer.current_row()
        count = self._file_explorer.active_list_count()
        if row < count - 1:
            self._navigate_to(active, row + 1)

    def _prev_file(self) -> None:
        active = self._file_explorer.get_active_list()
        row = self._file_explorer.current_row()
        if row > 0:
            self._navigate_to(active, row - 1)

    def _advance_to_next_pending(self) -> None:
        pending = self._session.pending
        if not pending:
            self._update_info()
            QMessageBox.information(self, "Done", "All videos have been classified!")
            return
        row = min(self._current_row, len(pending) - 1)
        row = max(0, row)
        self._navigate_to("pending", row)

    def _update_info(self) -> None:
        self._pending_n = len(self._session.pending)
        self._classified_n = len(self._session.classified)
        self._update_info_text()

    def _update_info_text(self) -> None:
        if not self._current_item:
            self._info_label.setText("No videos loaded")
            self.setWindowTitle("VidTriage")
            return

        item = self._current_item
        pos = _fmt_time(self._player.get_position())
        dur = _fmt_time(self._player.get_duration())
        name = item.original_path.name

        status = ""
        if item.is_error:
            status = " [error]"
        elif item.class_name:
            status = f" [{item.class_name}]"

        total = self._pending_n + self._classified_n
        idx = self._current_row + 1
        list_count = self._pending_n if self._current_list == "pending" else self._classified_n

        self._info_label.setText(
            f"{name}{status} · {pos} / {dur} · {self._pending_n} pending / {self._classified_n} classified"
        )
        self.setWindowTitle(
            f"[{idx}/{list_count}] {name}{status} — VidTriage ({self._classified_n}/{total} done)"
        )

    # ── Slider ─────────────────────────────────────────────

    def _on_position_changed(self, pos: float) -> None:
        if not self._slider_dragging and self._duration > 0:
            self._slider.setValue(int(pos / self._duration * 1000))
        self._update_info_text()

    def _on_duration_changed(self, dur: float) -> None:
        self._duration = dur

    def _on_slider_pressed(self) -> None:
        self._slider_dragging = True

    def _on_slider_released(self) -> None:
        self._slider_dragging = False
        if self._duration > 0:
            self._player.seek(self._slider.value() / 1000.0 * self._duration)

    def _on_slider_moved(self, value: int) -> None:
        if self._duration > 0:
            self._player.seek(value / 1000.0 * self._duration)

    # ── Playback ───────────────────────────────────────────

    def _toggle_play(self) -> None:
        self._player.toggle_pause()
        self._btn_play.setText("▶" if self._player.is_paused() else "⏸")

    # ── Classification ─────────────────────────────────────

    def _classify(self, class_entry: ClassEntry) -> None:
        if not self._current_item:
            return

        item = self._current_item
        was_pending = item.is_pending

        self._player.stop()
        try:
            self._session.classify(item, class_entry)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to classify: {e}")
            return

        self._sync_explorer()

        if was_pending:
            self._advance_to_next_pending()
        else:
            classified = self._session.classified
            try:
                row = classified.index(item)
            except ValueError:
                row = max(0, len(classified) - 1)
            self._navigate_to("classified", row)

    def _mark_error(self) -> None:
        if not self._current_item or not self._current_item.is_pending:
            return

        self._player.stop()
        try:
            self._session.mark_error(self._current_item)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to move file: {e}")
            return

        self._sync_explorer()
        self._advance_to_next_pending()

    def _undo(self) -> None:
        self._player.stop()
        item = self._session.undo_last()
        if item is None:
            return

        self._sync_explorer()

        if item.is_pending:
            pending = self._session.pending
            try:
                row = pending.index(item)
            except ValueError:
                row = 0
            self._navigate_to("pending", row)
        else:
            classified = self._session.classified
            try:
                row = classified.index(item)
            except ValueError:
                row = 0
            self._navigate_to("classified", row)

    # ── Summary ────────────────────────────────────────────

    def _show_summary(self) -> None:
        from collections import Counter
        pending = self._session.pending
        classified = self._session.classified
        pending_n = len(pending)
        classified_n = len(classified)
        total = pending_n + classified_n

        class_counts = Counter(
            item.class_name for item in classified if not item.is_error
        )
        error_count = sum(1 for item in classified if item.is_error)

        rows_html = ""
        for name, count in class_counts.most_common():
            pct = f"{count / total * 100:.1f}%" if total else "0%"
            rows_html += (
                f"<tr><td style='padding:4px 8px;'>{name}</td>"
                f"<td style='padding:4px 8px; text-align:right;'>{count}</td>"
                f"<td style='padding:4px 8px; text-align:right;'>{pct}</td></tr>"
            )
        if error_count:
            pct = f"{error_count / total * 100:.1f}%" if total else "0%"
            rows_html += (
                f"<tr><td style='padding:4px 8px; color:#ef5350;'>_errors</td>"
                f"<td style='padding:4px 8px; text-align:right;'>{error_count}</td>"
                f"<td style='padding:4px 8px; text-align:right;'>{pct}</td></tr>"
            )

        progress_pct = f"{classified_n / total * 100:.1f}%" if total else "0%"

        dlg = QDialog(self)
        dlg.setWindowTitle("VidTriage — Summary")
        dlg.resize(420, 380)
        layout = QVBoxLayout(dlg)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml(
            f"<h2 style='text-align:center;'>Session Summary</h2>"
            f"<p style='text-align:center; font-size:16px;'>"
            f"<b>{classified_n}</b> / {total} classified ({progress_pct})"
            f"&nbsp;&nbsp;·&nbsp;&nbsp;<b>{pending_n}</b> pending</p>"
            f"<table cellpadding='4' cellspacing='0' width='100%' "
            f"style='border-collapse:collapse; font-size:14px; margin-top:12px;'>"
            f"<tr style='background:#2a2a2a;'>"
            f"<th style='text-align:left; padding:6px 8px; border-bottom:1px solid #555;'>Class</th>"
            f"<th style='text-align:right; padding:6px 8px; border-bottom:1px solid #555;'>Count</th>"
            f"<th style='text-align:right; padding:6px 8px; border-bottom:1px solid #555;'>%</th></tr>"
            f"{rows_html}"
            f"</table>"
        )
        layout.addWidget(browser)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        layout.addWidget(btn_close)
        dlg.exec()

    # ── Export ─────────────────────────────────────────────

    def _export_annotations(self) -> None:
        import csv
        default_path = str(self._session.output_dir / "annotations.csv")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Annotations", default_path, "CSV Files (*.csv)",
        )
        if not path:
            return

        rows: list[tuple[str, str, str]] = []
        for item in self._session.pending:
            name = item.original_path.name
            rows.append((name, "unclassified", name))
        for item in self._session.classified:
            name = item.original_path.name
            if item.is_error:
                rows.append((name, "unclassified", name))
            elif item.class_name:
                rows.append((name, item.class_name, f"{item.class_name}/{name}"))
            else:
                rows.append((name, "unclassified", name))

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video", "class", "path"])
            writer.writerows(rows)

        QMessageBox.information(
            self, "Export", f"Exported {len(rows)} entries to\n{path}",
        )

    # ── Help ───────────────────────────────────────────────

    def _show_help(self) -> None:
        shortcuts = [
            ("1 &ndash; 9", "Classify / reclassify with mapped class"),
            ("Space", "Play / Pause"),
            ("&rarr;", "Step forward"),
            ("&larr;", "Step backward"),
            ("&darr;", "Next file"),
            ("&uarr;", "Previous file"),
            ("Tab", "Toggle Pending / Classified lists"),
            ("Ctrl+Z", "Undo last action"),
            ("x", "Move to <code>_errors/</code>"),
            ("s", "Skip to next pending"),
            ("e", "Toggle file explorer"),
            ("h / ?", "Open this help"),
            ("F11", "Toggle fullscreen"),
            ("Ctrl+Q", "Quit"),
        ]

        rows_html = ""
        for i, (key, desc) in enumerate(shortcuts):
            bg = " style='background:#1e1e1e;'" if i % 2 else ""
            rows_html += (
                f"<tr{bg}>"
                f"<td style='padding:6px 8px;'><code>{key}</code></td>"
                f"<td style='padding:6px 8px;'>{desc}</td></tr>"
            )

        classes_html = "".join(
            f"<tr><td style='padding:4px 8px;'><code>{c.key}</code></td>"
            f"<td style='padding:4px 8px;'>{c.name}</td></tr>"
            for c in self._session.classes
        )

        dlg = QDialog(self)
        dlg.setWindowTitle("VidTriage — Help")
        dlg.resize(480, 600)
        layout = QVBoxLayout(dlg)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml(
            "<h2 style='text-align:center;'>Keyboard Shortcuts</h2>"
            "<table cellpadding='6' cellspacing='0' width='100%' "
            "style='border-collapse:collapse; font-size:14px;'>"
            "<tr style='background:#2a2a2a;'>"
            "<th style='text-align:left; padding:8px; border-bottom:1px solid #555;'>Key</th>"
            "<th style='text-align:left; padding:8px; border-bottom:1px solid #555;'>Action</th></tr>"
            f"{rows_html}"
            "</table>"
            "<h3 style='margin-top:18px;'>Current Classes</h3>"
            "<table cellpadding='4' cellspacing='0' width='100%' "
            "style='border-collapse:collapse; font-size:14px;'>"
            f"{classes_html}"
            "</table>"
        )
        layout.addWidget(browser)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        layout.addWidget(btn_close)
        dlg.exec()

    # ── Class management ───────────────────────────────────

    def _rebuild_legend(self) -> None:
        layout = self._legend_layout
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            else:
                del item
        layout.addWidget(QLabel("Classes:"))
        for entry in self._session.classes:
            layout.addWidget(QLabel(f"  [{entry.key}] {entry.name}"))
        layout.addStretch()

    def _change_classes(self) -> None:
        from .config import parse_classes

        dlg = QDialog(self)
        dlg.setWindowTitle("VidTriage — Change Classes")
        dlg.resize(400, 340)
        layout = QVBoxLayout(dlg)

        layout.addWidget(QLabel("Classes (one per line, keys auto-assigned 1-9):"))
        edit = QTextEdit()
        user_classes = [c.name for c in self._session.classes]
        edit.setPlainText("\n".join(user_classes))
        edit.setPlaceholderText("cat\ndog\nbird\nskip")
        layout.addWidget(edit)

        btn_layout = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(dlg.reject)
        btn_apply = QPushButton("Apply")
        btn_layout.addWidget(btn_cancel)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_apply)
        layout.addLayout(btn_layout)

        def on_apply():
            entries, errors = parse_classes(edit.toPlainText())
            if errors:
                QMessageBox.warning(dlg, "Validation Error", "\n".join(errors))
                return
            self._session.set_classes(entries)
            self._rebuild_legend()
            dlg.accept()

        btn_apply.clicked.connect(on_apply)
        dlg.exec()

    def _prompt_new_class(self, key: str) -> None:
        if not self._current_item or len(self._session.classes) >= 9:
            return

        name, ok = QInputDialog.getText(
            self, "New Class",
            f"Name for key [{key}]:",
            text=f"class_{key}",
        )
        if not ok or not name.strip():
            return

        entry = self._session.add_class(key, name.strip())
        self._rebuild_legend()
        self._classify(entry)

    def _reopen_setup(self) -> None:
        from .wizard import SetupWizard
        wizard = SetupWizard(
            self,
            prefill_input=self._session.input_dir,
            prefill_output=self._session.output_dir,
        )
        if wizard.exec() and wizard.result_config:
            rc = wizard.result_config
            self._session = Session(rc.input_dir, rc.output_dir, rc.classes)
            self._session.load()
            self._rebuild_legend()
            self._sync_explorer()
            self._current_item = None
            self._current_row = -1
            if self._session.pending:
                self._navigate_to("pending", 0)

    # ── Key handling (app-level event filter) ──────────────

    def eventFilter(self, obj: object, event: QEvent) -> bool:
        if event.type() != QEvent.Type.KeyPress:
            return False

        if QApplication.activeWindow() is not self or QApplication.activePopupWidget():
            return False

        key_event: QKeyEvent = event  # type: ignore[assignment]
        key = key_event.key()
        text = key_event.text()

        if key == Qt.Key.Key_Tab:
            self._file_explorer.toggle_focus()
            active = self._file_explorer.get_active_list()
            row = self._file_explorer.current_row()
            if row >= 0:
                self._navigate_to(active, row)
            return True
        if key == Qt.Key.Key_Space:
            self._toggle_play()
            return True
        if key == Qt.Key.Key_Right:
            self._player.frame_step()
            return True
        if key == Qt.Key.Key_Left:
            self._player.frame_back_step()
            return True
        if key == Qt.Key.Key_Down:
            self._next_file()
            return True
        if key == Qt.Key.Key_Up:
            self._prev_file()
            return True
        if text == "x":
            self._mark_error()
            return True
        if text == "s":
            self._skip()
            return True
        if text == "e":
            self._act_explorer.toggle()
            return True
        if text in ("h", "?"):
            self._show_help()
            return True

        class_map = self._session.class_map
        if text in class_map:
            self._classify(class_map[text])
            return True
        if text.isdigit() and text != "0":
            self._prompt_new_class(text)
            return True

        return False

    def closeEvent(self, event) -> None:
        QApplication.instance().removeEventFilter(self)
        self._player.cleanup()
        super().closeEvent(event)
