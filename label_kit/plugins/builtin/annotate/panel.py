"""The annotation side panel: current label, what is on this frame, quick edits."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ....core.annotations import MANUAL_SOURCE, Annotation
from ....view.theme import Theme, ThemedMixin

if TYPE_CHECKING:
    from ....app.context import AppContext
    from . import AnnotatePlugin

__all__ = ["AnnotationPanel"]


class AnnotationPanel(QWidget, ThemedMixin):
    """Shows the annotations on the current frame and edits them in place.

    Selection is two-way with the canvas: clicking a row highlights the shape,
    and clicking a shape highlights the row.
    """

    def __init__(self, app: AppContext, plugin: AnnotatePlugin) -> None:
        super().__init__()
        self._app = app
        self._plugin = plugin
        self._syncing = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("Label:"))
        self._label_combo = QComboBox()
        self._label_combo.setEditable(True)
        self._label_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._label_combo.setToolTip("Applied to new annotations and model results")
        self._label_combo.currentTextChanged.connect(self._on_label_typed)
        label_row.addWidget(self._label_combo, stretch=1)
        layout.addLayout(label_row)

        self._prompt_label = QLabel()
        self._prompt_label.setWordWrap(True)
        layout.addWidget(self._prompt_label)

        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._list.itemSelectionChanged.connect(self._on_list_selection)
        layout.addWidget(self._list, stretch=1)

        button_row = QHBoxLayout()
        self._btn_relabel = QPushButton("Relabel")
        self._btn_relabel.setToolTip("Apply the label above to the selected annotations")
        self._btn_relabel.clicked.connect(self._relabel_selected)
        button_row.addWidget(self._btn_relabel)

        self._btn_delete = QPushButton("Delete")
        self._btn_delete.clicked.connect(self._delete_selected)
        button_row.addWidget(self._btn_delete)
        layout.addLayout(button_row)

        self._btn_promote = QPushButton("Accept All Predictions")
        self._btn_promote.setToolTip(
            "Mark every model prediction on this frame as confirmed, so it "
            "renders solid and exports as a reviewed annotation",
        )
        self._btn_promote.clicked.connect(plugin.promote_predictions)
        layout.addWidget(self._btn_promote)

        self._summary = QLabel()
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        app.frame_changed.connect(lambda _f: self.refresh())
        app.annotations.changed.connect(lambda _c: self.refresh())
        app.canvas.selection_changed.connect(self._on_canvas_selection)

        self.init_theme()
        self.refresh()

    def apply_theme(self, theme: Theme) -> None:
        self._summary.setStyleSheet(f"color: {theme.info_fg}; font-size: 11px;")
        self._prompt_label.setStyleSheet(f"color: {theme.info_fg}; font-size: 11px;")

    # ── refresh ─────────────────────────────────────────────────────────

    def refresh(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self._refresh_labels()
            self._refresh_list()
            self._refresh_prompt_hint()
        finally:
            self._syncing = False

    def _refresh_labels(self) -> None:
        known = sorted({*self._app.annotations.labels(), *self._plugin.recent_labels})
        current = self._plugin.current_label
        self._label_combo.blockSignals(True)
        self._label_combo.clear()
        self._label_combo.addItems(known)
        self._label_combo.setCurrentText(current)
        self._label_combo.blockSignals(False)

    def _refresh_list(self) -> None:
        frame = self._app.canvas.frame
        self._list.blockSignals(True)
        self._list.clear()

        annotations: list[Annotation] = (
            self._app.annotations.for_frame(frame.ref) if frame else []
        )
        for annotation in annotations:
            item = QListWidgetItem(_describe(annotation))
            item.setData(Qt.ItemDataRole.UserRole, annotation.id)
            item.setForeground(self._app.canvas.palette_map.color(annotation.label))
            if annotation.is_prediction:
                font = item.font()
                font.setItalic(True)
                item.setFont(font)
            self._list.addItem(item)

        self._list.blockSignals(False)

        predictions = sum(1 for a in annotations if a.is_prediction)
        total_frames = len(self._app.annotations.frames_with_annotations())
        self._summary.setText(
            f"{len(annotations)} on this frame ({predictions} predicted) · "
            f"{len(self._app.annotations)} total across {total_frames} frame(s)",
        )
        self._btn_promote.setEnabled(predictions > 0)

    def _refresh_prompt_hint(self) -> None:
        model = self._plugin.prompt_model
        if model is None:
            self._prompt_label.setText(
                "Prompt target: <b>manual</b> — shapes you draw are saved as-is.",
            )
        else:
            self._prompt_label.setText(
                f"Prompt target: <b>{model.display_name}</b> — "
                "drawing a box or clicking a point runs the model there.",
            )

    # ── selection sync ──────────────────────────────────────────────────

    def _on_list_selection(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self._app.canvas.select_annotations(self._selected_ids())
        finally:
            self._syncing = False

    def _on_canvas_selection(self, ids: tuple[str, ...]) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            wanted = set(ids)
            for row in range(self._list.count()):
                item = self._list.item(row)
                item.setSelected(item.data(Qt.ItemDataRole.UserRole) in wanted)
        finally:
            self._syncing = False

    def _selected_ids(self) -> list[str]:
        return [
            item.data(Qt.ItemDataRole.UserRole) for item in self._list.selectedItems()
        ]

    # ── actions ─────────────────────────────────────────────────────────

    def _on_label_typed(self, text: str) -> None:
        if not self._syncing:
            self._plugin.set_current_label(text.strip())

    def _relabel_selected(self) -> None:
        label = self._plugin.current_label
        store = self._app.annotations
        updated = [
            store.by_id(annotation_id).with_label(label)
            for annotation_id in self._selected_ids()
            if store.by_id(annotation_id) is not None
        ]
        if updated:
            store.update(updated)
            self._app.status(f"Relabelled {len(updated)} to '{label}'", 3000)

    def _delete_selected(self) -> None:
        store = self._app.annotations
        removed = store.remove([
            a for a in (store.by_id(i) for i in self._selected_ids()) if a is not None
        ])
        if removed:
            self._app.status(f"Deleted {len(removed)} annotation(s)", 2500)


def _describe(annotation: Annotation) -> str:
    label = annotation.label or "(unlabelled)"
    box = annotation.bounding_rect
    parts = [f"{annotation.kind.value:>7} · {label}"]
    if annotation.score is not None:
        parts.append(f"{annotation.score:.2f}")
    parts.append(f"{int(box.x1)},{int(box.y1)} {int(box.width)}×{int(box.height)}")
    if annotation.source != MANUAL_SOURCE:
        parts.append(f"[{annotation.source}]")
    return "  ".join(parts)
