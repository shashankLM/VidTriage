"""The two-list file explorer: pending above, classified below."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ....view.theme import Theme, ThemedMixin
from .models import VideoItem

__all__ = ["CLASSIFIED", "PENDING", "FileExplorerWidget"]

PENDING = "pending"
CLASSIFIED = "classified"


class FileExplorerWidget(QWidget, ThemedMixin):
    """Pending and classified videos, with the focused list outlined."""

    file_selected = Signal(str, int)  # list name, row

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self._pending_header, self._pending_list = self._make_panel(splitter, "Pending")
        self._classified_header, self._classified_list = self._make_panel(splitter, "Classified")
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        self._pending_list.currentRowChanged.connect(
            lambda row: self._on_row_changed(PENDING, row),
        )
        self._classified_list.currentRowChanged.connect(
            lambda row: self._on_row_changed(CLASSIFIED, row),
        )

        self._pending_items: list[VideoItem] = []
        self._classified_items: list[VideoItem] = []
        self._active = PENDING

        self.init_theme()

    def _make_panel(self, splitter: QSplitter, title: str) -> tuple[QLabel, QListWidget]:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        header = QLabel(f"{title} (0)")
        header.setStyleSheet("font-weight: bold;")
        layout.addWidget(header)

        listing = QListWidget()
        listing.setUniformItemSizes(True)
        layout.addWidget(listing)

        splitter.addWidget(panel)
        return header, listing

    # ── theming ─────────────────────────────────────────────────────────

    def apply_theme(self, theme: Theme) -> None:
        self._update_focus_style()
        self._refresh()

    # ── contents ────────────────────────────────────────────────────────

    def set_items(self, pending: list[VideoItem], classified: list[VideoItem]) -> None:
        self._pending_items = pending
        self._classified_items = classified
        self._refresh()

    def _refresh(self) -> None:
        from ....view.theme import current_theme

        theme = current_theme()

        self._fill(
            self._pending_list,
            [(item.name, QColor(theme.pending_fg)) for item in self._pending_items],
        )
        self._pending_header.setText(f"Pending ({len(self._pending_items)})")

        rows = []
        for item in self._classified_items:
            if item.is_error:
                rows.append((f"[error] {item.name}", QColor(theme.error_item_fg)))
            elif item.class_name:
                rows.append((f"[{item.class_name}] {item.name}", QColor(theme.classified_fg)))
            else:
                rows.append((item.name, QColor(theme.pending_fg)))
        self._fill(self._classified_list, rows)
        self._classified_header.setText(f"Classified ({len(self._classified_items)})")

        self._update_focus_style()

    @staticmethod
    def _fill(listing: QListWidget, rows: list[tuple[str, QColor]]) -> None:
        # Preserve the cursor across a rebuild so classifying does not scroll
        # the user back to the top of a thousand-file list.
        previous = listing.currentRow()
        listing.blockSignals(True)
        listing.clear()
        for text, color in rows:
            entry = QListWidgetItem(text)
            entry.setForeground(color)
            listing.addItem(entry)
        if 0 <= previous < listing.count():
            listing.setCurrentRow(previous)
        listing.blockSignals(False)

    # ── focus ───────────────────────────────────────────────────────────

    @property
    def active_list(self) -> str:
        return self._active

    def _set_active(self, which: str) -> None:
        if which == self._active:
            return
        self._active = which
        other = self._classified_list if which == PENDING else self._pending_list
        other.blockSignals(True)
        other.clearSelection()
        other.setCurrentRow(-1)
        other.blockSignals(False)
        self._update_focus_style()

    def _update_focus_style(self) -> None:
        from ....view.theme import current_theme

        theme = current_theme()
        focused = theme.focused_list_style()
        unfocused = theme.unfocused_list_style()
        pending_focused = self._active == PENDING
        self._pending_list.setStyleSheet(focused if pending_focused else unfocused)
        self._classified_list.setStyleSheet(unfocused if pending_focused else focused)

    def toggle_focus(self) -> None:
        target = CLASSIFIED if self._active == PENDING else PENDING
        self._set_active(target)
        listing = self._pending_list if target == PENDING else self._classified_list
        if listing.count() and listing.currentRow() < 0:
            listing.setCurrentRow(0)

    # ── selection ───────────────────────────────────────────────────────

    def _on_row_changed(self, which: str, row: int) -> None:
        if row < 0:
            return
        self._set_active(which)
        self.file_selected.emit(which, row)

    def select(self, which: str, row: int) -> None:
        """Move the cursor without re-emitting :attr:`file_selected`."""
        self._set_active(which)
        listing = self._pending_list if which == PENDING else self._classified_list
        listing.blockSignals(True)
        listing.setCurrentRow(row)
        listing.blockSignals(False)
        if 0 <= row < listing.count():
            listing.scrollToItem(listing.item(row))

    def current_row(self) -> int:
        listing = self._pending_list if self._active == PENDING else self._classified_list
        return listing.currentRow()

    def count(self, which: str | None = None) -> int:
        target = which or self._active
        return len(self._pending_items if target == PENDING else self._classified_items)

    def items(self, which: str) -> list[VideoItem]:
        return self._pending_items if which == PENDING else self._classified_items
