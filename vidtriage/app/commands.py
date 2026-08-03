"""Commands the shell itself owns, plus the machinery that mirrors registries.

Two kinds live here:

*Static* commands — open, quit, zoom, play/pause — are registered once.

*Derived* commands are generated from a registry by :class:`RegistryCommands`
and re-synced whenever that registry changes. Registering a tool, an overlay
layer, a model or a dock panel therefore makes a working, correctly-checked menu
entry appear on its own. A plugin never writes menu code, and disabling a plugin
removes its entries because its contributions left the registry.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..core.commands import Command, CommandRegistry
from ..core.logging import get_logger
from ..core.registry import Registry
from ..media.clock import EndMode
from ..plugins.models import Capability
from ..view.theme import THEMES, current_theme, set_theme

if TYPE_CHECKING:
    from .context import AppContext
    from .window import MainWindow

__all__ = ["RegistryCommands", "register_core_commands"]

_log = get_logger(__name__)

_SPEEDS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0)
_FRAME_STEPS = (1, 2, 5, 10, 30)
_END_MODES = ((EndMode.NEXT, "Next File"), (EndMode.LOOP, "Loop"), (EndMode.STOP, "Stop"))


class RegistryCommands:
    """Keeps a set of commands in one-to-one correspondence with a registry.

    ``build`` maps a registry entry to a :class:`Command`, or to ``None`` to skip
    it. The sync is idempotent and coalesced by the menu builder downstream, so
    a plugin registering ten layers costs one menu rebuild.
    """

    def __init__(
        self,
        commands: CommandRegistry,
        registry: Registry,
        build: Callable[[str, object], Command | None],
    ) -> None:
        self._commands = commands
        self._registry = registry
        self._build = build
        self._owned: set[str] = set()

        registry.changed.connect(self._on_registry_changed)
        self.sync()

    def _on_registry_changed(self, _change: object) -> None:
        self.sync()

    def sync(self) -> None:
        wanted: dict[str, Command] = {}
        for key, value in self._registry.items():
            command = self._build(key, value)
            if command is not None:
                wanted[command.id] = command

        for stale in self._owned - set(wanted):
            self._commands.unregister(stale)
        for command_id, command in wanted.items():
            self._commands.add(command, replace=command_id in self._owned)
        self._owned = set(wanted)

    def dispose(self) -> None:
        for command_id in self._owned:
            self._commands.unregister(command_id)
        self._owned.clear()


def register_core_commands(app: AppContext, window: MainWindow) -> list[RegistryCommands]:
    """Register the shell's own commands. Returns the live registry mirrors."""
    add = app.add_command
    canvas = app.canvas
    playback = app.playback
    library = app.library
    store = app.annotations

    # ── File ────────────────────────────────────────────────────────────
    add(
        id="file.open", title="Open File…", shortcut="Ctrl+O", menu="File", section="0",
        order=10, handler=window.prompt_open_file,
        description="Open a single video or image",
    )
    add(
        id="file.open_folder", title="Open Folder…", shortcut="Ctrl+Shift+O", menu="File",
        section="0", order=20, handler=window.prompt_open_folder,
        description="Load every video and image in a folder",
    )
    add(
        id="file.export", title="Export Annotations…", shortcut="Ctrl+E", menu="File",
        section="1", order=10, handler=window.prompt_export,
        is_enabled=lambda: len(app.exporters) > 0,
        description="Write annotations as COCO, YOLO or CSV",
    )
    add(
        id="file.save_annotations", title="Save Annotations Now", shortcut="Ctrl+S",
        menu="File", section="1", order=20, handler=app.flush_annotations,
        is_enabled=lambda: store.is_dirty,
    )
    add(
        id="file.autosave", title="Autosave Annotations", menu="File", section="1", order=30,
        checkable=True, is_checked=lambda: app.autosave_annotations,
        handler=lambda checked: setattr(app, "autosave_annotations", checked),
        description="Write a .vidtriage.json sidecar when leaving a file",
    )
    add(
        id="file.quit", title="Quit", shortcut="Ctrl+Q", menu="File", section="9",
        handler=window.close,
    )

    # ── Edit ────────────────────────────────────────────────────────────
    add(
        id="edit.undo", title="Undo", shortcut="Ctrl+Z", menu="Edit", section="0", order=10,
        handler=lambda: _undo(app), is_enabled=lambda: store.can_undo,
    )
    add(
        id="edit.redo", title="Redo", shortcut="Ctrl+Shift+Z", menu="Edit", section="0",
        order=20, handler=lambda: _redo(app), is_enabled=lambda: store.can_redo,
    )
    add(
        id="edit.delete", title="Delete Selected", shortcut="Del", menu="Edit", section="1",
        order=10, handler=lambda: _delete_selected(app),
        is_enabled=lambda: bool(canvas.selected_ids()),
    )
    add(
        id="edit.select_all", title="Select All On Frame", shortcut="Ctrl+A", menu="Edit",
        section="1", order=20, handler=lambda: _select_all(app),
    )
    add(
        id="edit.clear_frame", title="Clear This Frame", menu="Edit", section="1", order=30,
        handler=lambda: _clear_frame(app),
        is_enabled=lambda: bool(canvas.frame and store.count_for_frame(canvas.frame.ref)),
    )

    # ── View ────────────────────────────────────────────────────────────
    add(
        id="view.zoom_in", title="Zoom In", shortcut="Ctrl+=", menu="View", section="0",
        order=10, handler=canvas.zoom_in,
    )
    add(
        id="view.zoom_out", title="Zoom Out", shortcut="Ctrl+-", menu="View", section="0",
        order=20, handler=canvas.zoom_out,
    )
    add(
        id="view.zoom_fit", title="Fit To Window", shortcut="Ctrl+0", menu="View", section="0",
        order=30, handler=canvas.fit_to_window,
    )
    add(
        id="view.zoom_actual", title="Actual Size", shortcut="Ctrl+1", menu="View", section="0",
        order=40, handler=canvas.zoom_reset,
    )
    add(
        id="view.fullscreen", title="Fullscreen", shortcut="F11", menu="View", section="1",
        checkable=True, is_checked=window.isFullScreen, handler=window.set_fullscreen,
    )
    add(
        id="view.labels", title="Show Labels", menu="View/Annotations", section="0", order=10,
        checkable=True, is_checked=lambda: canvas.style.show_labels,
        handler=lambda checked: canvas.set_style_option(show_labels=checked),
    )
    add(
        id="view.scores", title="Show Scores", menu="View/Annotations", section="0", order=20,
        checkable=True, is_checked=lambda: canvas.style.show_scores,
        handler=lambda checked: canvas.set_style_option(show_scores=checked),
    )
    add(
        id="view.fill", title="Fill Shapes", menu="View/Annotations", section="0", order=30,
        checkable=True, is_checked=lambda: canvas.style.fill_shapes,
        handler=lambda checked: canvas.set_style_option(fill_shapes=checked),
    )
    add(
        id="view.plugins", title="Plugins…", menu="View", section="8",
        handler=window.show_plugins_dialog,
        description="See which plugins loaded, and enable or disable them",
    )

    for name in THEMES:
        add(
            id=f"view.theme.{name.lower().replace(' ', '_')}", title=name,
            menu="View/Theme", section="0", checkable=True,
            is_checked=lambda n=name: current_theme().name == n,
            handler=lambda checked, n=name: _apply_theme(app, n),
        )

    # ── Playback ────────────────────────────────────────────────────────
    add(
        id="playback.toggle", title="Play / Pause", shortcut="Space", menu="Playback",
        section="0", order=10, handler=playback.toggle_play,
        is_enabled=lambda: playback.is_loaded,
    )
    add(
        id="playback.step_forward", title="Step Forward", shortcut="Right", menu="Playback",
        section="0", order=20, handler=playback.step_forward,
        is_enabled=lambda: playback.is_loaded,
    )
    add(
        id="playback.step_back", title="Step Backward", shortcut="Left", menu="Playback",
        section="0", order=30, handler=playback.step_backward,
        is_enabled=lambda: playback.is_loaded,
    )
    add(
        id="playback.next_file", title="Next File", shortcut="Down", menu="Playback",
        section="1", order=10, handler=library.next, is_enabled=lambda: library.has_next,
    )
    add(
        id="playback.prev_file", title="Previous File", shortcut="Up", menu="Playback",
        section="1", order=20, handler=library.previous,
        is_enabled=lambda: library.has_previous,
    )

    for speed in _SPEEDS:
        add(
            id=f"playback.speed.{speed}", title=f"{speed:g}×", menu="Playback/Speed",
            order=int(speed * 100), checkable=True,
            is_checked=lambda s=speed: abs(playback.speed - s) < 1e-6,
            handler=lambda checked, s=speed: playback.set_speed(s),
        )
    for step in _FRAME_STEPS:
        add(
            id=f"playback.step.{step}", title=f"{step} frame{'s' if step > 1 else ''}",
            menu="Playback/Frame Step", order=step, checkable=True,
            is_checked=lambda s=step: playback.step_size == s,
            handler=lambda checked, s=step: playback.set_step_size(s),
        )
    for mode, label in _END_MODES:
        add(
            id=f"playback.end.{mode.value}", title=label, menu="Playback/At End Of File",
            checkable=True, is_checked=lambda m=mode: playback.end_mode is m,
            handler=lambda checked, m=mode: playback.set_end_mode(m),
        )

    # ── Help ────────────────────────────────────────────────────────────
    add(
        id="help.shortcuts", title="Keyboard Shortcuts", shortcut="F1", menu="Help",
        handler=window.show_shortcuts_dialog,
    )
    add(
        id="help.about", title="About VidTriage", menu="Help", section="9",
        handler=window.show_about_dialog,
    )

    # ── derived from registries ─────────────────────────────────────────
    return [
        RegistryCommands(app.commands, canvas.tools, lambda key, tool: Command(
            id=f"tool.{key}",
            title=tool.title or key,
            description=tool.hint,
            menu="Tools",
            section="0",
            shortcut=tool.shortcut,
            checkable=True,
            is_checked=lambda t=key: canvas.active_tool.id == t,
            handler=lambda checked, t=key: _activate_tool(app, t),
        )),
        RegistryCommands(app.commands, canvas.layers, lambda key, layer: None
            if not layer.user_toggleable else Command(
                id=f"overlay.{key}",
                title=layer.title or key,
                menu="View/Overlays",
                shortcut=layer.shortcut,
                checkable=True,
                is_checked=lambda ly=layer: ly.visible,
                handler=lambda checked, ly=layer: setattr(ly, "visible", checked),
            )),
        # Only models that can run unprompted get a "Run …" entry. A
        # point-or-box-only model like SAM has nothing to do without a gesture,
        # so listing it here would only ever show a permanently disabled item.
        RegistryCommands(app.commands, app.models, lambda key, model: None
            if not (model.capabilities & Capability.WHOLE_FRAME) else Command(
                id=f"model.run.{key}",
                title=f"Run {model.display_name or key}",
                description=model.description or model.availability().message,
                menu="Models",
                section="0",
                handler=lambda m=key: app.run_model(m),
                is_enabled=lambda m=model: bool(app.current_frame) and m.availability().ok,
            )),
        RegistryCommands(app.commands, app.panels, lambda key, panel: Command(
            id=f"panel.{key}",
            title=panel.title,
            menu="View/Panels",
            shortcut=panel.shortcut,
            checkable=True,
            is_checked=lambda p=key: window.is_panel_visible(p),
            handler=lambda checked, p=key: window.set_panel_visible(p, checked),
        )),
    ]


# ── handlers ────────────────────────────────────────────────────────────


def _activate_tool(app: AppContext, tool_id: str) -> None:
    tool = app.canvas.set_tool(tool_id)
    if tool.hint:
        app.status(tool.hint, 4000)


def _apply_theme(app: AppContext, name: str) -> None:
    set_theme(name)
    app.settings.set("ui.theme", name)


def _undo(app: AppContext) -> None:
    description = app.annotations.undo_description()
    if app.annotations.undo():
        app.status(f"Undo: {description}", 2500)


def _redo(app: AppContext) -> None:
    if app.annotations.redo():
        app.status("Redo", 2000)


def _delete_selected(app: AppContext) -> None:
    ids = app.canvas.selected_ids()
    if not ids:
        return
    removed = app.annotations.remove([
        a for a in (app.annotations.by_id(i) for i in ids) if a is not None
    ])
    app.status(f"Deleted {len(removed)} annotation(s)", 2500)


def _select_all(app: AppContext) -> None:
    frame = app.canvas.frame
    if frame is None:
        return
    app.canvas.select_annotations([a.id for a in app.annotations.for_frame(frame.ref)])


def _clear_frame(app: AppContext) -> None:
    frame = app.canvas.frame
    if frame is None:
        return
    removed = app.annotations.remove_frame(frame.ref)
    app.status(f"Cleared {len(removed)} annotation(s) from frame {frame.index}", 3000)
