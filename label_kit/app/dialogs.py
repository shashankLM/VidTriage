"""Shell dialogs: plugins, export, shortcuts, about.

The plugin dialog is the one that earns its keep. When a model does not appear
in the Models menu the question is always "why", and the answer is here in
plain text — not installed, weights missing, failed to import with a traceback,
or simply switched off — together with the command that fixes it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.logging import get_logger
from ..persistence.exporters import (
    ExportItem,
    ExportRequest,
    items_from_library,
    items_from_store,
)
from ..view.theme import current_theme

if TYPE_CHECKING:
    from .context import AppContext

__all__ = [
    "ExportDialog",
    "PluginDialog",
    "show_about",
    "show_html",
    "show_shortcuts",
]

_log = get_logger(__name__)

_SCOPE_LIBRARY = "library"
_SCOPE_CURRENT = "current"

_STATUS_HINT = {
    "active": "Running.",
    "inactive": "Enabled but not started.",
    "disabled": "Switched off. Tick to enable.",
    "unavailable": "Prerequisites are missing.",
    "error": "Failed to load — see the details below.",
}


class PluginDialog(QDialog):
    """Lists every discovered plugin with its status, and toggles them live."""

    def __init__(self, app: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._app = app
        self.setWindowTitle("label-kit — Plugins")
        self.resize(720, 480)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Plugins contribute tools, overlays, models and panels. "
            "Changes apply immediately.",
        ))

        self._tree = QTreeWidget()
        self._tree.setColumnCount(4)
        self._tree.setHeaderLabels(["Plugin", "Status", "Source", "Detail"])
        self._tree.setRootIsDecorated(False)
        self._tree.setAlternatingRowColors(True)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        header = self._tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._tree, stretch=1)

        self._detail = QTextBrowser()
        self._detail.setMaximumHeight(150)
        self._detail.setPlaceholderText("Select a plugin to see details.")
        layout.addWidget(self._detail)

        self._tree.currentItemChanged.connect(self._show_detail)
        self._tree.itemChanged.connect(self._on_item_changed)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self._populate()

    def _populate(self) -> None:
        self._tree.blockSignals(True)
        self._tree.clear()
        for plugin_id in sorted(self._app.plugins.states):
            state = self._app.plugins.states[plugin_id]
            plugin = state.plugin
            item = QTreeWidgetItem([
                plugin.name or plugin_id,
                state.status,
                state.origin,
                plugin.description or state.availability.reason,
            ])
            item.setData(0, Qt.ItemDataRole.UserRole, plugin_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                0,
                Qt.CheckState.Checked if state.enabled else Qt.CheckState.Unchecked,
            )
            if plugin.essential or state.error or not state.availability.ok:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)

            theme = current_theme()
            if state.error:
                item.setForeground(1, theme.color("error_item_fg"))
            elif state.active:
                item.setForeground(1, theme.color("classified_fg"))
            else:
                item.setForeground(1, theme.color("info_fg"))
            self._tree.addTopLevelItem(item)
        self._tree.blockSignals(False)

    def _show_detail(self, item: QTreeWidgetItem | None, _previous=None) -> None:
        if item is None:
            self._detail.clear()
            return
        plugin_id = item.data(0, Qt.ItemDataRole.UserRole)
        state = self._app.plugins.states.get(plugin_id)
        if state is None:
            return

        lines = [
            f"<b>{state.plugin.name or plugin_id}</b> &nbsp; "
            f"<code>{plugin_id}</code> &nbsp; v{state.plugin.version}",
            f"<p>{state.plugin.description}</p>" if state.plugin.description else "",
            f"<p><i>{_STATUS_HINT.get(state.status, '')}</i></p>",
        ]
        if state.plugin.requires:
            lines.append(f"<p>Requires: <code>{', '.join(state.plugin.requires)}</code></p>")
        if not state.availability.ok:
            lines.append(f"<p><b>Unavailable:</b> {state.availability.reason}</p>")
            if state.availability.remedy:
                lines.append(f"<pre>{state.availability.remedy}</pre>")
        if state.error:
            lines.append(f"<pre>{state.error}</pre>")
        self._detail.setHtml("".join(lines))

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0:
            return
        plugin_id = item.data(0, Qt.ItemDataRole.UserRole)
        enabled = item.checkState(0) == Qt.CheckState.Checked
        self._app.plugins.set_enabled(plugin_id, enabled, self._app)
        self._populate()
        self._app.status(
            f"Plugin {plugin_id} {'enabled' if enabled else 'disabled'}", 3000,
        )


class ExportDialog(QDialog):
    """Pick a scope, a format and a destination, then run the exporter."""

    def __init__(self, app: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._app = app
        self.setWindowTitle("label-kit — Export Annotations")
        self.resize(560, 300)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._scope = QComboBox()
        self._scope.addItem("Every annotated file in the playlist", _SCOPE_LIBRARY)
        self._scope.addItem("Current file only", _SCOPE_CURRENT)
        self._scope.setToolTip(
            "A dataset usually spans the whole session. Exporting one file at a "
            "time into the same folder is what the label formats handle worst.",
        )
        self._scope.currentIndexChanged.connect(lambda _i: self._update_summary())
        form.addRow("Export:", self._scope)

        self._format = QComboBox()
        for exporter in app.exporters:
            self._format.addItem(exporter.title, exporter.id)
        form.addRow("Format:", self._format)

        destination_row = QHBoxLayout()
        self._destination = QLabel("<i>not chosen</i>")
        self._destination.setWordWrap(True)
        browse = QPushButton("Choose…")
        browse.clicked.connect(self._choose_destination)
        destination_row.addWidget(self._destination, stretch=1)
        destination_row.addWidget(browse)
        form.addRow("Destination:", destination_row)

        self._write_frames = QCheckBox("Also extract the annotated frames as images")
        self._write_frames.setToolTip(
            "COCO and YOLO reference image files. Without this the labels have "
            "nothing to point at, so a training run cannot use them.",
        )
        form.addRow("", self._write_frames)
        layout.addLayout(form)

        self._summary = QLabel()
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)
        layout.addStretch()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Export")
        buttons.accepted.connect(self._run)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok_button.setEnabled(False)

        self._chosen: Path | None = None
        self._update_summary()

    def _current_exporter(self):
        return self._app.exporters.get(self._format.currentData())

    def _choose_destination(self) -> None:
        exporter = self._current_exporter()
        if exporter is None:
            return
        start = str(self._app.settings.get("export.last_dir", str(Path.home())))
        if exporter.writes_directory:
            chosen = QFileDialog.getExistingDirectory(self, "Export into folder", start)
        else:
            suggested = str(
                Path(start) / f"annotations{exporter.default_extension}",
            )
            chosen, _ = QFileDialog.getSaveFileName(
                self, "Export to", suggested, exporter.file_filter,
            )
        if not chosen:
            return
        self._chosen = Path(chosen)
        self._destination.setText(str(self._chosen))
        self._app.settings.set("export.last_dir", str(self._chosen.parent))
        self._ok_button.setEnabled(True)

    def _current_items(self) -> list[ExportItem]:
        media = self._app.current_media
        info = self._app.media_info
        if media is None or info is None:
            return []
        return items_from_store(media, info.size, self._app.annotations.all())

    def _library_items(self) -> tuple[list[ExportItem], list[str]]:
        """Every annotated file in the playlist.

        The open file is taken from the live store rather than from its sidecar,
        so an export includes edits that autosave has not written out yet —
        without this the last few annotations drawn would be missing, which is
        the kind of loss nobody notices until training.
        """
        app = self._app
        current = app.current_media
        items, warnings = items_from_library(p for p in app.library.items if p != current)
        return self._current_items() + items, warnings

    def _gather(self) -> tuple[list[ExportItem], list[str]]:
        if self._scope.currentData() == _SCOPE_CURRENT:
            return self._current_items(), []
        return self._library_items()

    def _build_request(self, items: list[ExportItem]) -> ExportRequest | None:
        if self._chosen is None or not items:
            return None
        return ExportRequest(
            destination=self._chosen,
            items=items,
            write_frames=self._write_frames.isChecked(),
        )

    def _update_summary(self) -> None:
        items, warnings = self._gather()
        annotations = sum(len(item.annotations) for item in items)
        files = len({item.media_path for item in items})
        note = f"  ({len(warnings)} file(s) skipped)" if warnings else ""
        self._summary.setText(
            f"{annotations} annotation(s) across {len(items)} frame(s) "
            f"of {files} file(s) will be exported.{note}",
        )

    def _run(self) -> None:
        exporter = self._current_exporter()
        items, skipped = self._gather()
        request = self._build_request(items)
        if exporter is None or request is None:
            self._app.status(
                "Nothing to export — no annotations in scope" if exporter else "Nothing to export",
                4000,
            )
            self.reject()
            return

        try:
            report = exporter.export(request)
        except Exception as exc:  # noqa: BLE001 - surface the failure, keep the app alive
            _log.exception("Export via %s failed", exporter.id)
            self._app.status(f"Export failed: {exc}", 8000)
            self.reject()
            return

        for warning in (*skipped, *report.warnings):
            _log.warning("Export: %s", warning)
        total = len(skipped) + len(report.warnings)
        note = f" · {total} warning(s), see the log" if total else ""
        self._app.status(f"Exported — {report.summary}{note}", 8000)
        self.accept()


def show_shortcuts(app: AppContext, parent: QWidget | None = None) -> None:
    """List every bound shortcut, grouped by menu — generated from the registry.

    Because it is generated, it cannot drift out of date the way a hand-written
    help table does.
    """
    by_menu: dict[str, list[tuple[str, str, str]]] = {}
    for command in app.commands.values():
        if not command.shortcut:
            continue
        group = command.menu_path[0] if command.menu_path else "Other"
        by_menu.setdefault(group, []).append(
            (command.shortcut, command.title, command.description),
        )

    rows = []
    for group in sorted(by_menu):
        rows.append(
            f"<tr><td colspan='2' style='padding:12px 6px 4px;'><b>{group}</b></td></tr>",
        )
        for shortcut, title, description in sorted(by_menu[group]):
            detail = f" <span style='opacity:.6'>— {description}</span>" if description else ""
            rows.append(
                f"<tr><td style='padding:3px 12px 3px 18px; white-space:nowrap;'>"
                f"<code>{shortcut}</code></td>"
                f"<td style='padding:3px 6px;'>{title}{detail}</td></tr>",
            )

    extra = (
        "<p style='margin-top:14px;'><b>Mouse</b><br>"
        "Wheel — zoom at the cursor &nbsp;·&nbsp; Middle-drag — pan &nbsp;·&nbsp; "
        "Right-click — context menu</p>"
    )
    show_html(
        parent, "Keyboard Shortcuts",
        f"<h2>Keyboard Shortcuts</h2><table width='100%'>{''.join(rows)}</table>{extra}",
        (620, 640),
    )


def show_about(app: AppContext, parent: QWidget | None = None) -> None:
    from .. import __version__

    active = [s for s in app.plugins.states.values() if s.active]
    plugin_rows = "".join(
        f"<li><code>{s.id}</code> — {s.plugin.name}</li>" for s in sorted(active, key=lambda s: s.id)
    )
    show_html(
        parent, "About label-kit",
        f"<h2>label-kit {__version__}</h2>"
        "<p>Video and image triage, frame annotation, and model-assisted labelling.</p>"
        f"<p><b>{len(app.commands)}</b> commands · "
        f"<b>{len(app.canvas.tools)}</b> tools · "
        f"<b>{len(app.canvas.layers)}</b> overlays · "
        f"<b>{len(app.models)}</b> models · "
        f"<b>{len(app.exporters)}</b> exporters</p>"
        f"<p><b>Active plugins</b></p><ul>{plugin_rows or '<li>none</li>'}</ul>",
        (520, 440),
    )


def show_html(
    parent: QWidget | None, title: str, html: str, size: tuple[int, int] = (560, 480),
) -> None:
    """Read-only HTML in a modal box. Shared by every informational dialog."""
    dialog = QDialog(parent)
    dialog.setWindowTitle(f"label-kit — {title}")
    dialog.resize(*size)
    layout = QVBoxLayout(dialog)
    browser = QTextBrowser()
    browser.setOpenExternalLinks(False)
    browser.setHtml(html)
    layout.addWidget(browser)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dialog.accept)
    layout.addWidget(buttons)
    dialog.exec()
