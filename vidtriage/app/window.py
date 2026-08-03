"""The main window — a thin shell over :class:`AppContext`.

What it does: lay out the canvas and transport bar, host dock panels, build the
menu bar and shortcuts from the registries, restore and save window state.

What it deliberately does *not* do: know about classification, annotation,
models, or any particular feature. Those arrive as plugin contributions. The
predecessor was an 847-line class holding the key map, the classification logic,
the CSV export, the summary and help dialogs, and the theme wiring — so every
new feature meant editing it, and everything in it could break everything else.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from ..core.logging import get_logger
from ..media.source import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from ..view.theme import THEMES, Theme, ThemedMixin, current_theme, set_theme, theme_manager
from .commands import register_core_commands
from .context import AppContext
from .dialogs import ExportDialog, PluginDialog, show_about, show_shortcuts
from .library import discover_media
from .menus import MenuBuilder, ShortcutBinder
from .transport import TransportBar

__all__ = ["MainWindow"]

_log = get_logger(__name__)

_DOCK_AREAS = {
    "left": Qt.DockWidgetArea.LeftDockWidgetArea,
    "right": Qt.DockWidgetArea.RightDockWidgetArea,
    "top": Qt.DockWidgetArea.TopDockWidgetArea,
    "bottom": Qt.DockWidgetArea.BottomDockWidgetArea,
}


class MainWindow(QMainWindow, ThemedMixin):
    """Hosts the canvas, the transport bar, and whatever plugins contribute."""

    def __init__(self, app: AppContext) -> None:
        super().__init__()
        self.setWindowTitle("VidTriage")
        self.app = app
        app.window = self

        self._docks: dict[str, QDockWidget] = {}
        self._built_panels: set[str] = set()

        self._build_central()
        self.setStatusBar(self.statusBar())

        app.status_message.connect(self._on_status_message)
        app.media_opened.connect(self._on_media_opened)
        app.library.current_changed.connect(lambda _p: self._update_title())
        app.canvas.cursor_moved.connect(self._on_cursor_moved)
        app.panels.changed.connect(lambda _c: self.sync_panels())

        # Registry mirrors must outlive this call, hence the attribute.
        self._registry_commands = register_core_commands(app, self)
        self._menus = MenuBuilder(self, app.commands)
        self._shortcuts = ShortcutBinder(self, app.commands)

        self.setAcceptDrops(True)
        self.init_theme()
        self._restore_state()

    # ── layout ──────────────────────────────────────────────────────────

    def _build_central(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self.app.canvas, stretch=1)

        self._transport = TransportBar(self.app.playback, self.app.library)
        self._transport.next_requested.connect(self.app.library.next)
        self._transport.previous_requested.connect(self.app.library.previous)
        layout.addWidget(self._transport)

        self.setCentralWidget(central)

    def apply_theme(self, theme: Theme) -> None:
        application = QApplication.instance()
        if application is not None:
            application.setStyleSheet(theme.app_stylesheet())

    # ── panels ──────────────────────────────────────────────────────────

    def sync_panels(self) -> None:
        """Add docks for new panel specs and drop those whose plugin went away.

        Called on every ``panels`` registry change, and once after startup
        activation so plugins loaded before the window existed get their docks.
        """
        wanted = set(self.app.panels.keys())
        for panel_id in list(self._docks):
            if panel_id not in wanted:
                dock = self._docks.pop(panel_id)
                self.removeDockWidget(dock)
                dock.deleteLater()
                self._built_panels.discard(panel_id)

        for panel_id, spec in self.app.panels.items():
            if panel_id in self._docks:
                continue
            dock = QDockWidget(spec.title, self)
            dock.setObjectName(f"dock_{panel_id}")
            dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
            self.addDockWidget(_DOCK_AREAS.get(spec.area, Qt.DockWidgetArea.RightDockWidgetArea), dock)
            self._docks[panel_id] = dock

            remembered = self.app.settings.get(f"panels.{panel_id}.visible")
            visible = spec.visible_by_default if remembered is None else bool(remembered)
            self.set_panel_visible(panel_id, visible)

    def set_panel_visible(self, panel_id: str, visible: bool) -> None:
        dock = self._docks.get(panel_id)
        spec = self.app.panels.get(panel_id)
        if dock is None or spec is None:
            return

        # Build the widget on first show, so an expensive panel costs nothing
        # for a user who never opens it.
        if visible and panel_id not in self._built_panels:
            try:
                dock.setWidget(spec.factory())
            except Exception:  # noqa: BLE001 - a broken panel must not kill the window
                _log.exception("Panel %r failed to build", panel_id)
                self.app.status(f"Panel '{spec.title}' failed to open", 6000)
                return
            self._built_panels.add(panel_id)

        dock.setVisible(visible)
        self.app.settings.set(f"panels.{panel_id}.visible", visible)

    def is_panel_visible(self, panel_id: str) -> bool:
        dock = self._docks.get(panel_id)
        return bool(dock and dock.isVisible())

    def panel_widget(self, panel_id: str) -> QWidget | None:
        dock = self._docks.get(panel_id)
        return dock.widget() if dock else None

    # ── file commands ───────────────────────────────────────────────────

    def prompt_open_file(self) -> None:
        patterns = " ".join(
            f"*{ext}" for ext in sorted(VIDEO_EXTENSIONS | IMAGE_EXTENSIONS)
        )
        start = str(self.app.settings.get("io.last_open_dir", str(Path.home())))
        path, _ = QFileDialog.getOpenFileName(
            self, "Open media", start, f"Media ({patterns});;All Files (*)",
        )
        if path:
            self.app.settings.set("io.last_open_dir", str(Path(path).parent))
            self.app.open_media(Path(path))

    def prompt_open_folder(self) -> None:
        start = str(self.app.settings.get("io.last_open_dir", str(Path.home())))
        directory = QFileDialog.getExistingDirectory(self, "Open folder", start)
        if not directory:
            return
        found = discover_media(Path(directory))
        if not found:
            self.app.status(f"No media files in {Path(directory).name}", 5000)
            return
        self.app.settings.set("io.last_open_dir", directory)
        self.app.library.set_items(found, keep_current=False)
        self.app.status(f"Loaded {len(found)} file(s)", 4000)

    def prompt_export(self) -> None:
        ExportDialog(self.app, self).exec()

    def show_plugins_dialog(self) -> None:
        PluginDialog(self.app, self).exec()

    def show_shortcuts_dialog(self) -> None:
        show_shortcuts(self.app, self)

    def show_about_dialog(self) -> None:
        show_about(self.app, self)

    def set_fullscreen(self, enabled: bool) -> None:
        self.showFullScreen() if enabled else self.showNormal()

    # ── status / title ──────────────────────────────────────────────────

    def _on_status_message(self, text: str, timeout: int) -> None:
        self.statusBar().showMessage(text, timeout)

    def _on_cursor_moved(self, point) -> None:
        if point is None or not self.app.canvas.is_inside_image(point):
            return
        self.statusBar().showMessage(f"x={int(point.x)}  y={int(point.y)}", 1500)

    def _on_media_opened(self, _path: Path) -> None:
        self._update_title()

    def _update_title(self) -> None:
        current = self.app.library.current
        if current is None:
            self.setWindowTitle("VidTriage")
            return
        position = ""
        if len(self.app.library) > 1:
            position = f"[{self.app.library.index + 1}/{len(self.app.library)}] "
        self.setWindowTitle(f"{position}{current.name} — VidTriage")

    # ── drag and drop ───────────────────────────────────────────────────

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls()]
        files: list[Path] = []
        for path in paths:
            if path.is_dir():
                files.extend(discover_media(path))
            elif path.suffix.lower() in (VIDEO_EXTENSIONS | IMAGE_EXTENSIONS):
                files.append(path)
        if not files:
            self.app.status("Nothing openable in that drop", 3000)
            return
        self.app.library.set_items(files, keep_current=False)
        self.app.status(f"Loaded {len(files)} file(s)", 4000)
        event.acceptProposedAction()

    # ── window state ────────────────────────────────────────────────────

    def _restore_state(self) -> None:
        settings = self.app.settings

        theme_name = settings.get("ui.theme", current_theme().name)
        if theme_name in THEMES:
            set_theme(theme_name)
        else:
            # Force a broadcast so widgets pick up the default on first launch.
            theme_manager().changed.emit(current_theme())

        self.resize(
            int(settings.get("window.width", 1280)),
            int(settings.get("window.height", 800)),
        )
        geometry = settings.get("window.geometry")
        if isinstance(geometry, str):
            self.restoreGeometry(QByteArray.fromBase64(geometry.encode("ascii")))
        layout = settings.get("window.layout")
        if isinstance(layout, str):
            self.restoreState(QByteArray.fromBase64(layout.encode("ascii")))
        if settings.get("window.fullscreen"):
            self.showFullScreen()

        self.app.canvas.set_style_option(
            show_labels=bool(settings.get("canvas.show_labels", True)),
            show_scores=bool(settings.get("canvas.show_scores", True)),
            fill_shapes=bool(settings.get("canvas.fill_shapes", True)),
        )
        for layer_id in ("core.frame_info", "core.crosshair"):
            remembered = settings.get(f"overlays.{layer_id}")
            if remembered is not None:
                self.app.canvas.set_layer_visible(layer_id, bool(remembered))

    def _save_state(self) -> None:
        canvas = self.app.canvas
        values = {
            "ui.theme": current_theme().name,
            "window.width": self.width(),
            "window.height": self.height(),
            "window.fullscreen": self.isFullScreen(),
            "window.geometry": bytes(self.saveGeometry().toBase64()).decode("ascii"),
            "window.layout": bytes(self.saveState().toBase64()).decode("ascii"),
            "canvas.show_labels": canvas.style.show_labels,
            "canvas.show_scores": canvas.style.show_scores,
            "canvas.fill_shapes": canvas.style.fill_shapes,
        }
        for layer in canvas.layers:
            if layer.user_toggleable:
                values[f"overlays.{layer.id}"] = layer.visible
        self.app.settings.update(values)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_state()
        self.app.shutdown()
        super().closeEvent(event)
