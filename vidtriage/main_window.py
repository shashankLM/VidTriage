from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QEvent
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QSlider, QSplitter, QMessageBox, QFrame, QDialog, QTextBrowser, QApplication,
)

from .models import AppConfig, ClassEntry, FileStatus, VideoItem
from .player import CvPlayerWidget
from .file_explorer import FileExplorerWidget
from .io_ops import (
    discover_videos, move_to_class, move_to_errors, undo_move, log_action, load_log,
    setup_logger,
)


def _fmt_time(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.setWindowTitle("VidTriage")
        self.resize(1200, 750)

        self._config = config
        self._class_map: dict[str, ClassEntry] = {c.key: c for c in config.classes}

        self._pending: list[VideoItem] = []
        self._classified: list[VideoItem] = []
        self._current_item: VideoItem | None = None
        self._current_list: str = "pending"
        self._current_row: int = -1

        self._undo_stack: list[tuple[str, int, VideoItem, Path, ClassEntry | None, str]] = []

        assert config.output_dir is not None
        setup_logger(config.output_dir)

        self._build_ui()
        self._load_videos()
        self._reconcile_log()
        self._sync_explorer()

        QApplication.instance().installEventFilter(self)

        if self._pending:
            self._navigate_to("pending", 0)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # --- Toolbar ---
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(8, 4, 8, 4)
        btn_setup = QPushButton("Reopen Setup")
        btn_setup.clicked.connect(self._reopen_setup)
        btn_summary = QPushButton("Summary")
        btn_summary.clicked.connect(self._show_summary)
        btn_help = QPushButton("Help (h)")
        btn_help.clicked.connect(self._show_help)
        btn_quit = QPushButton("Quit")
        btn_quit.clicked.connect(self.close)
        toolbar.addWidget(btn_setup)
        toolbar.addWidget(btn_summary)
        toolbar.addWidget(btn_help)
        toolbar.addStretch()
        toolbar.addWidget(btn_quit)
        root_layout.addLayout(toolbar)

        # --- Main splitter ---
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._file_explorer = FileExplorerWidget()
        self._file_explorer.setMinimumWidth(200)
        self._file_explorer.setMaximumWidth(400)
        self._file_explorer.file_selected.connect(self._on_file_selected)
        splitter.addWidget(self._file_explorer)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._player = CvPlayerWidget()
        right_layout.addWidget(self._player, stretch=1)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root_layout.addWidget(splitter, stretch=1)

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

        self._btn_play = QPushButton("⏸")
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
        legend_frame = QFrame()
        legend_frame.setFrameShape(QFrame.Shape.StyledPanel)
        legend_layout = QHBoxLayout(legend_frame)
        legend_layout.setContentsMargins(8, 6, 8, 6)
        legend_layout.addWidget(QLabel("Classes:"))
        for entry in self._config.classes:
            legend_layout.addWidget(QLabel(f"  [{entry.key}] {entry.name}"))
        legend_layout.addStretch()
        root_layout.addWidget(legend_frame)

        self._player.position_changed.connect(self._on_position_changed)
        self._player.duration_changed.connect(self._on_duration_changed)
        self._slider_dragging = False
        self._duration = 0.0

    # --- Data loading ---

    def _load_videos(self) -> None:
        assert self._config.input_dir is not None
        self._pending = [VideoItem(original_path=p) for p in discover_videos(self._config.input_dir)]
        self._classified = []

    def _reconcile_log(self) -> None:
        assert self._config.output_dir is not None
        rows = load_log(self._config.output_dir)
        classified_sources: dict[str, dict[str, str]] = {}
        for row in rows:
            action = row.get("action", "")
            src = row.get("source_path", "")
            if action == "undo":
                classified_sources.pop(src, None)
            elif action in ("classify", "error", "skip"):
                classified_sources[src] = row

        # Match pending items against log
        still_pending: list[VideoItem] = []
        for item in self._pending:
            key = str(item.original_path)
            if key in classified_sources:
                row = classified_sources.pop(key)
                dest = Path(row["destination_path"])
                if dest.exists():
                    action = row["action"]
                    if action == "error":
                        item.status = FileStatus.ERROR
                    else:
                        item.status = FileStatus.CLASSIFIED
                        item.class_entry = ClassEntry(
                            key=row.get("class_key", ""),
                            name=row.get("class_name", ""),
                        )
                    item.destination_path = dest
                    self._classified.append(item)
                else:
                    still_pending.append(item)
            else:
                still_pending.append(item)
        self._pending = still_pending

        # Load remaining log entries not found in input dir (from previous sessions)
        for src, row in classified_sources.items():
            dest = Path(row["destination_path"])
            if not dest.exists():
                continue
            action = row["action"]
            item = VideoItem(original_path=Path(src))
            if action == "error":
                item.status = FileStatus.ERROR
            else:
                item.status = FileStatus.CLASSIFIED
                item.class_entry = ClassEntry(
                    key=row.get("class_key", ""),
                    name=row.get("class_name", ""),
                )
            item.destination_path = dest
            self._classified.append(item)

    def _sync_explorer(self) -> None:
        self._file_explorer.set_items(self._pending, self._classified)

    # --- Navigation ---

    def _navigate_to(self, list_name: str, row: int) -> None:
        items = self._pending if list_name == "pending" else self._classified
        if not items or row < 0 or row >= len(items):
            return

        self._current_list = list_name
        self._current_row = row
        self._current_item = items[row]
        item = self._current_item

        if list_name == "pending":
            self._file_explorer.select_pending(row)
        else:
            self._file_explorer.select_classified(row)

        path = item.destination_path if item.destination_path and item.destination_path.exists() else item.original_path
        if path.exists():
            self._player.load(path)
        else:
            self._player.show_error("File not found")

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
        if not self._pending:
            self._update_info()
            QMessageBox.information(self, "Done", "All videos have been classified!")
            return
        row = min(self._current_row, len(self._pending) - 1)
        row = max(0, row)
        self._navigate_to("pending", row)

    def _update_info(self) -> None:
        pending_n = len(self._pending)
        classified_n = len(self._classified)
        total = pending_n + classified_n

        if not self._current_item:
            self._info_label.setText("No videos loaded")
            self.setWindowTitle("VidTriage")
            return

        item = self._current_item
        pos = _fmt_time(self._player.get_position())
        dur = _fmt_time(self._player.get_duration())
        name = item.original_path.name

        status = ""
        if item.status == FileStatus.CLASSIFIED and item.class_entry:
            status = f" [{item.class_entry.name}]"
        elif item.status == FileStatus.ERROR:
            status = " [error]"

        idx = self._current_row + 1
        list_count = len(self._pending) if self._current_list == "pending" else len(self._classified)

        self._info_label.setText(
            f"{name}{status} · {pos} / {dur} · {pending_n} pending / {classified_n} classified"
        )
        self.setWindowTitle(
            f"[{idx}/{list_count}] {name}{status} — VidTriage ({classified_n}/{total} done)"
        )

    # --- Slider ---

    def _on_position_changed(self, pos: float) -> None:
        if not self._slider_dragging and self._duration > 0:
            self._slider.setValue(int(pos / self._duration * 1000))
        self._update_info()

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

    # --- Playback ---

    def _toggle_play(self) -> None:
        self._player.toggle_pause()
        self._btn_play.setText("▶" if self._player.is_paused() else "⏸")

    # --- Classification ---

    def _classify(self, class_entry: ClassEntry) -> None:
        if not self._current_item:
            return

        item = self._current_item
        assert self._config.output_dir is not None

        if item.status == FileStatus.PENDING:
            self._classify_pending(item, class_entry)
        elif item.status in (FileStatus.CLASSIFIED, FileStatus.ERROR):
            self._reclassify(item, class_entry)

    def _classify_pending(self, item: VideoItem, class_entry: ClassEntry) -> None:
        if not item.original_path.exists():
            return

        self._player.stop()
        try:
            dest = move_to_class(item.original_path, self._config.output_dir, class_entry)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to move file: {e}")
            return

        item.status = FileStatus.CLASSIFIED
        item.class_entry = class_entry
        item.destination_path = dest

        log_action(
            self._config.output_dir, item.original_path, dest,
            class_entry.key, class_entry.name, "classify",
        )

        row = self._current_row
        self._pending.remove(item)
        self._classified.append(item)

        self._undo_stack.append(("pending", row, item, item.original_path, class_entry, "classify"))

        self._sync_explorer()
        self._advance_to_next_pending()

    def _reclassify(self, item: VideoItem, class_entry: ClassEntry) -> None:
        if item.class_entry and item.class_entry.key == class_entry.key:
            return

        old_dest = item.destination_path
        if not old_dest or not old_dest.exists():
            return

        self._player.stop()
        try:
            new_dest = move_to_class(old_dest, self._config.output_dir, class_entry)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to reclassify: {e}")
            return

        old_class = item.class_entry
        row = self._current_row

        item.status = FileStatus.CLASSIFIED
        item.class_entry = class_entry
        item.destination_path = new_dest

        log_action(
            self._config.output_dir, item.original_path, new_dest,
            class_entry.key, class_entry.name, "classify",
        )

        self._undo_stack.append(("classified", row, item, old_dest, old_class, "reclassify"))

        self._sync_explorer()
        if row < len(self._classified):
            self._navigate_to("classified", row)
        elif self._classified:
            self._navigate_to("classified", len(self._classified) - 1)

    def _mark_error(self) -> None:
        if not self._current_item:
            return
        item = self._current_item
        if item.status not in (FileStatus.PENDING,):
            return
        if not item.original_path.exists():
            return

        assert self._config.output_dir is not None
        self._player.stop()
        try:
            dest = move_to_errors(item.original_path, self._config.output_dir)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to move file: {e}")
            return

        row = self._current_row
        item.status = FileStatus.ERROR
        item.destination_path = dest

        log_action(
            self._config.output_dir, item.original_path, dest,
            "x", "_errors", "error",
        )

        self._pending.remove(item)
        self._classified.append(item)

        self._undo_stack.append(("pending", row, item, item.original_path, None, "error"))

        self._sync_explorer()
        self._advance_to_next_pending()

    def _undo(self) -> None:
        if not self._undo_stack:
            return

        prev_list, prev_row, item, original_path, old_class, action = self._undo_stack.pop()

        assert self._config.output_dir is not None
        self._player.stop()

        if action == "reclassify":
            # Move back from new dest to old dest
            current_dest = item.destination_path
            if not current_dest or not current_dest.exists():
                return
            try:
                from .io_ops import _unique_dest
                import shutil
                old_dest = _unique_dest(original_path)
                shutil.move(str(current_dest), str(old_dest))
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to undo: {e}")
                return

            item.class_entry = old_class
            item.destination_path = old_dest

            log_action(
                self._config.output_dir, item.original_path, old_dest,
                old_class.key if old_class else "", old_class.name if old_class else "",
                "undo",
            )

            self._sync_explorer()
            idx = self._classified.index(item) if item in self._classified else 0
            self._navigate_to("classified", idx)

        else:
            # Undo classify or error: move back to input dir
            current_dest = item.destination_path
            if not current_dest or not current_dest.exists():
                return
            try:
                undo_move(current_dest, original_path)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to undo: {e}")
                return

            item.status = FileStatus.PENDING
            item.class_entry = None
            item.destination_path = None

            log_action(
                self._config.output_dir, original_path, current_dest,
                old_class.key if old_class else "x",
                old_class.name if old_class else "_errors",
                "undo",
            )

            if item in self._classified:
                self._classified.remove(item)
            insert_at = min(prev_row, len(self._pending))
            self._pending.insert(insert_at, item)

            self._sync_explorer()
            self._navigate_to("pending", insert_at)

    # --- Summary ---

    def _show_summary(self) -> None:
        from collections import Counter
        pending_n = len(self._pending)
        classified_n = len(self._classified)
        total = pending_n + classified_n

        class_counts = Counter(
            item.class_entry.name
            for item in self._classified
            if item.class_entry and item.status == FileStatus.CLASSIFIED
        )
        error_count = sum(1 for item in self._classified if item.status == FileStatus.ERROR)

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

    # --- Help ---

    def _show_help(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("VidTriage — Help")
        dlg.resize(480, 540)
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
            "<tr><td style='padding:6px 8px;'><code>1</code> – <code>9</code></td>"
            "<td style='padding:6px 8px;'>Classify / reclassify with mapped class</td></tr>"
            "<tr style='background:#1e1e1e;'><td style='padding:6px 8px;'><code>Space</code></td>"
            "<td style='padding:6px 8px;'>Play / Pause</td></tr>"
            "<tr><td style='padding:6px 8px;'><code>&rarr;</code></td>"
            "<td style='padding:6px 8px;'>Next frame (step forward)</td></tr>"
            "<tr style='background:#1e1e1e;'><td style='padding:6px 8px;'><code>&larr;</code></td>"
            "<td style='padding:6px 8px;'>Previous frame</td></tr>"
            "<tr><td style='padding:6px 8px;'><code>&darr;</code></td>"
            "<td style='padding:6px 8px;'>Next file</td></tr>"
            "<tr style='background:#1e1e1e;'><td style='padding:6px 8px;'><code>&uarr;</code></td>"
            "<td style='padding:6px 8px;'>Previous file</td></tr>"
            "<tr><td style='padding:6px 8px;'><code>Tab</code></td>"
            "<td style='padding:6px 8px;'>Toggle focus between Pending / Classified lists</td></tr>"
            "<tr style='background:#1e1e1e;'><td style='padding:6px 8px;'><code>Ctrl+Z</code></td>"
            "<td style='padding:6px 8px;'>Undo last action</td></tr>"
            "<tr><td style='padding:6px 8px;'><code>x</code></td>"
            "<td style='padding:6px 8px;'>Move to <code>_errors/</code></td></tr>"
            "<tr style='background:#1e1e1e;'><td style='padding:6px 8px;'><code>h</code> / <code>?</code></td>"
            "<td style='padding:6px 8px;'>Open this help</td></tr>"
            "<tr><td style='padding:6px 8px;'><code>Ctrl+Q</code></td>"
            "<td style='padding:6px 8px;'>Quit</td></tr>"
            "</table>"
            "<h3 style='margin-top:18px;'>Current Classes</h3>"
            "<table cellpadding='4' cellspacing='0' width='100%' "
            "style='border-collapse:collapse; font-size:14px;'>"
            + "".join(
                f"<tr><td style='padding:4px 8px;'><code>{c.key}</code></td>"
                f"<td style='padding:4px 8px;'>{c.name}</td></tr>"
                for c in self._config.classes
            )
            + "</table>"
        )
        layout.addWidget(browser)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        layout.addWidget(btn_close)
        dlg.exec()

    def _reopen_setup(self) -> None:
        from .wizard import SetupWizard
        wizard = SetupWizard(self)
        if wizard.exec() and wizard.result_config:
            self._config = wizard.result_config
            self._class_map = {c.key: c for c in self._config.classes}
            self._load_videos()
            self._reconcile_log()
            self._sync_explorer()
            if self._pending:
                self._navigate_to("pending", 0)

    # --- Key handling (app-level event filter) ---

    def eventFilter(self, obj: object, event: QEvent) -> bool:
        if event.type() != QEvent.Type.KeyPress:
            return False

        key_event: QKeyEvent = event  # type: ignore[assignment]
        key = key_event.key()
        text = key_event.text()
        ctrl = bool(key_event.modifiers() & Qt.KeyboardModifier.ControlModifier)

        if ctrl and key == Qt.Key.Key_Q:
            self.close()
            return True
        if ctrl and key == Qt.Key.Key_Z:
            self._undo()
            return True
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
        if text in ("h", "?"):
            self._show_help()
            return True
        if text in self._class_map:
            self._classify(self._class_map[text])
            return True

        return False

    def closeEvent(self, event) -> None:
        QApplication.instance().removeEventFilter(self)
        self._player.cleanup()
        super().closeEvent(event)
