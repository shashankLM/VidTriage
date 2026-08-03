"""The file panel: a filter box, pending above, classified below.

**The signal carries the item, not a row.** It used to emit ``(list name, row)``
and the plugin re-derived the item with ``session.pending[row]`` — which is only
correct while the widget shows every video in the same order the session holds
them. A filter breaks that assumption immediately: row 0 of a filtered list is
not video 0 of the session, so a click would have selected a different file than
the one clicked. Emitting the item makes the mapping impossible to get wrong.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ....view.theme import Theme, ThemedMixin
from .models import MediaItem

__all__ = ["CLASSIFIED", "PENDING", "FileExplorerWidget"]

PENDING = "pending"
CLASSIFIED = "classified"


class FileExplorerWidget(QWidget, ThemedMixin):
    """Pending and classified media, filterable, with the focused list outlined."""

    file_selected = Signal(object)  # MediaItem

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter…  (Ctrl+F)")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._on_filter_changed)
        layout.addWidget(self._search)

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

        self._pending_items: list[MediaItem] = []
        self._classified_items: list[MediaItem] = []
        self._visible_pending: list[MediaItem] = []
        self._visible_classified: list[MediaItem] = []
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

    # ── filtering ───────────────────────────────────────────────────────

    def focus_search(self) -> None:
        """Put the cursor in the filter box, ready to replace what is there."""
        self._search.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self._search.selectAll()

    def clear_filter(self) -> None:
        self._search.clear()

    def _on_filter_changed(self, _text: str) -> None:
        self._refresh()

    def keyPressEvent(self, event) -> None:
        """Escape clears the filter and hands focus back to the list."""
        if event.key() == Qt.Key.Key_Escape and self._search.text():
            self.clear_filter()
            self._active_list().setFocus()
            return
        super().keyPressEvent(event)

    def _matches(self, text: str) -> bool:
        """Every whitespace-separated term must appear, case-insensitively.

        Terms are ANDed so that ``doorway single`` narrows to one clip out of a
        set whose names share long prefixes — which is the case this box exists
        for.
        """
        haystack = text.lower()
        return all(term in haystack for term in self._search.text().lower().split())

    # ── contents ────────────────────────────────────────────────────────

    def set_items(self, pending: list[MediaItem], classified: list[MediaItem]) -> None:
        self._pending_items = pending
        self._classified_items = classified
        self._refresh()

    def _label_for(self, item: MediaItem) -> tuple[str, str]:
        """``(display text, colour key)``. The class prefix is searchable too."""
        if item.is_error:
            return f"[error] {item.name}", "error_item_fg"
        if item.class_name:
            return f"[{item.class_name}] {item.name}", "classified_fg"
        return item.name, "pending_fg"

    def _refresh(self) -> None:
        from ....view.theme import current_theme

        theme = current_theme()
        selected = self.selected_item()

        self._visible_pending = []
        pending_rows: list[tuple[str, QColor]] = []
        for item in self._pending_items:
            text, _key = self._label_for(item)
            if self._matches(text):
                self._visible_pending.append(item)
                pending_rows.append((text, QColor(theme.pending_fg)))

        self._visible_classified = []
        classified_rows: list[tuple[str, QColor]] = []
        for item in self._classified_items:
            text, key = self._label_for(item)
            if self._matches(text):
                self._visible_classified.append(item)
                classified_rows.append((text, QColor(getattr(theme, key))))

        self._fill(self._pending_list, pending_rows)
        self._fill(self._classified_list, classified_rows)
        self._pending_header.setText(
            self._count_label("Pending", len(pending_rows), len(self._pending_items)),
        )
        self._classified_header.setText(
            self._count_label("Classified", len(classified_rows), len(self._classified_items)),
        )

        if selected is not None:
            self.select_item(selected)
        self._update_focus_style()

    def _count_label(self, title: str, shown: int, total: int) -> str:
        """Say how much the filter is hiding, or the count would look like loss."""
        if shown == total:
            return f"{title} ({total})"
        return f"{title} ({shown} of {total})"

    @staticmethod
    def _fill(listing: QListWidget, rows: list[tuple[str, QColor]]) -> None:
        listing.blockSignals(True)
        listing.clear()
        for text, color in rows:
            entry = QListWidgetItem(text)
            entry.setForeground(color)
            listing.addItem(entry)
        listing.blockSignals(False)

    # ── focus ───────────────────────────────────────────────────────────

    def _active_list(self) -> QListWidget:
        return self._pending_list if self._active == PENDING else self._classified_list

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

    def _visible(self, which: str) -> list[MediaItem]:
        return self._visible_pending if which == PENDING else self._visible_classified

    def _on_row_changed(self, which: str, row: int) -> None:
        rows = self._visible(which)
        if not 0 <= row < len(rows):
            return
        self._set_active(which)
        self.file_selected.emit(rows[row])

    def selected_item(self) -> MediaItem | None:
        rows = self._visible(self._active)
        row = self._active_list().currentRow()
        return rows[row] if 0 <= row < len(rows) else None

    def select_item(self, item: MediaItem) -> None:
        """Move the cursor to ``item`` without re-emitting :attr:`file_selected`.

        A no-op when the filter is hiding it — the cursor stays where it is
        rather than jumping to an unrelated row.
        """
        which = PENDING if item.is_pending else CLASSIFIED
        rows = self._visible(which)
        if item not in rows:
            return

        self._set_active(which)
        listing = self._pending_list if which == PENDING else self._classified_list
        row = rows.index(item)
        listing.blockSignals(True)
        listing.setCurrentRow(row)
        listing.blockSignals(False)
        listing.scrollToItem(listing.item(row))

    def items(self, which: str) -> list[MediaItem]:
        """Everything in this list, filter or no filter."""
        return self._pending_items if which == PENDING else self._classified_items

    def visible_items(self, which: str) -> list[MediaItem]:
        """Only what the filter is currently letting through."""
        return list(self._visible(which))
