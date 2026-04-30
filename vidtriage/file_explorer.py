from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QLabel, QSplitter,
)

from .models import VideoItem
from .theme import current_theme


class FileExplorerWidget(QWidget):
    file_selected = Signal(str, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # --- Pending panel ---
        pending_panel = QWidget()
        pending_layout = QVBoxLayout(pending_panel)
        pending_layout.setContentsMargins(0, 0, 0, 0)
        pending_layout.setSpacing(2)
        self._pending_header = QLabel("Pending (0)")
        self._pending_header.setStyleSheet("font-weight: bold;")
        pending_layout.addWidget(self._pending_header)
        self._pending_list = QListWidget()
        self._pending_list.currentRowChanged.connect(self._on_pending_row_changed)
        self._pending_list.clicked.connect(lambda: self._set_focus("pending"))
        pending_layout.addWidget(self._pending_list)
        splitter.addWidget(pending_panel)

        # --- Classified panel ---
        classified_panel = QWidget()
        classified_layout = QVBoxLayout(classified_panel)
        classified_layout.setContentsMargins(0, 0, 0, 0)
        classified_layout.setSpacing(2)
        self._classified_header = QLabel("Classified (0)")
        self._classified_header.setStyleSheet("font-weight: bold;")
        classified_layout.addWidget(self._classified_header)
        self._classified_list = QListWidget()
        self._classified_list.currentRowChanged.connect(self._on_classified_row_changed)
        self._classified_list.clicked.connect(lambda: self._set_focus("classified"))
        classified_layout.addWidget(self._classified_list)
        splitter.addWidget(classified_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        self._pending_items: list[VideoItem] = []
        self._classified_items: list[VideoItem] = []
        self._active_list: str = "pending"

        self._apply_theme()

    def apply_theme(self) -> None:
        self._apply_theme()
        self._refresh()

    def _apply_theme(self) -> None:
        t = current_theme()
        self._pending_list.setStyleSheet(t.focused_list_style())
        self._classified_list.setStyleSheet(t.unfocused_list_style())
        self._update_focus_style()

    def set_items(
        self,
        pending: list[VideoItem],
        classified: list[VideoItem],
    ) -> None:
        self._pending_items = pending
        self._classified_items = classified
        self._refresh()

    def _refresh(self) -> None:
        t = current_theme()

        self._pending_list.blockSignals(True)
        self._pending_list.clear()
        for item in self._pending_items:
            li = QListWidgetItem(item.original_path.name)
            li.setForeground(QColor(t.pending_fg))
            self._pending_list.addItem(li)
        self._pending_list.blockSignals(False)
        self._pending_header.setText(f"Pending ({len(self._pending_items)})")

        self._classified_list.blockSignals(True)
        self._classified_list.clear()
        for item in self._classified_items:
            if item.is_error:
                label = f"[error] {item.original_path.name}"
                color = QColor(t.error_item_fg)
            elif item.class_name:
                label = f"[{item.class_name}] {item.original_path.name}"
                color = QColor(t.classified_fg)
            else:
                label = item.original_path.name
                color = QColor(t.pending_fg)
            li = QListWidgetItem(label)
            li.setForeground(color)
            self._classified_list.addItem(li)
        self._classified_list.blockSignals(False)
        self._classified_header.setText(f"Classified ({len(self._classified_items)})")

        self._update_focus_style()

    def _set_focus(self, which: str) -> None:
        if which == self._active_list:
            return
        self._active_list = which
        if which == "pending":
            self._classified_list.blockSignals(True)
            self._classified_list.clearSelection()
            self._classified_list.setCurrentRow(-1)
            self._classified_list.blockSignals(False)
        else:
            self._pending_list.blockSignals(True)
            self._pending_list.clearSelection()
            self._pending_list.setCurrentRow(-1)
            self._pending_list.blockSignals(False)
        self._update_focus_style()

    def _update_focus_style(self) -> None:
        t = current_theme()
        if self._active_list == "pending":
            self._pending_list.setStyleSheet(t.focused_list_style())
            self._classified_list.setStyleSheet(t.unfocused_list_style())
        else:
            self._pending_list.setStyleSheet(t.unfocused_list_style())
            self._classified_list.setStyleSheet(t.focused_list_style())

    def _on_pending_row_changed(self, row: int) -> None:
        if row >= 0:
            self._set_focus("pending")
            self.file_selected.emit("pending", row)

    def _on_classified_row_changed(self, row: int) -> None:
        if row >= 0:
            self._set_focus("classified")
            self.file_selected.emit("classified", row)

    def get_active_list(self) -> str:
        return self._active_list

    def select_pending(self, row: int) -> None:
        self._set_focus("pending")
        self._pending_list.blockSignals(True)
        self._pending_list.setCurrentRow(row)
        self._pending_list.blockSignals(False)

    def select_classified(self, row: int) -> None:
        self._set_focus("classified")
        self._classified_list.blockSignals(True)
        self._classified_list.setCurrentRow(row)
        self._classified_list.blockSignals(False)

    def current_row(self) -> int:
        if self._active_list == "pending":
            return self._pending_list.currentRow()
        return self._classified_list.currentRow()

    def active_list_count(self) -> int:
        if self._active_list == "pending":
            return len(self._pending_items)
        return len(self._classified_items)

    def toggle_focus(self) -> None:
        if self._active_list == "pending":
            self._set_focus("classified")
            if self._classified_list.count() > 0 and self._classified_list.currentRow() < 0:
                self._classified_list.setCurrentRow(0)
        else:
            self._set_focus("pending")
            if self._pending_list.count() > 0 and self._pending_list.currentRow() < 0:
                self._pending_list.setCurrentRow(0)
