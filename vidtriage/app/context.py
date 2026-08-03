"""AppContext — the object every plugin is handed.

It owns the services (canvas, playback, annotation store, inference runner,
media library, settings) and the extension registries (commands, models, panels,
exporters). A plugin reads services off it and contributes through
``PluginContext``, which writes into these same registries.

There is deliberately no back-reference from services to the window. The window
is a *view* of this context, built from its registries, and can be rebuilt or
absent — which is what makes the whole stack drivable from a test with no
window at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, Signal

from ..core.annotations import AnnotationStore
from ..core.commands import Command, CommandRegistry
from ..core.frames import Frame, MediaInfo, source_id_for
from ..core.logging import get_logger
from ..core.registry import Registry
from ..media.controller import PlaybackController
from ..persistence.exporters import Exporter, builtin_exporters
from ..persistence.settings import Settings
from ..persistence.sidecar import load_into_store, save_store
from ..plugins.manager import PluginManager
from ..plugins.models import (
    InferenceModel,
    InferenceRequest,
    Prompt,
    WholeFramePrompt,
)
from ..plugins.runner import InferenceRunner
from ..view.canvas import ImageCanvas
from .library import MediaLibrary

if TYPE_CHECKING:
    from ..plugins.api import PanelSpec
    from .window import MainWindow

__all__ = ["AppContext"]

_log = get_logger(__name__)


class AppContext(QObject):
    """Services and extension registries for one running application."""

    media_opened = Signal(object)        # Path
    media_failed = Signal(object, str)   # Path, message
    frame_changed = Signal(object)       # Frame
    inference_finished = Signal(object)  # InferenceResult
    status_message = Signal(str, int)    # text, timeout ms

    def __init__(
        self,
        settings: Settings | None = None,
        plugin_manager: PluginManager | None = None,
        parent: QObject | None = None,
        launch_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(parent)

        self.settings = settings or Settings()
        self.plugins = plugin_manager or PluginManager()
        # How this process was launched, for plugins that take command-line
        # arguments. Distinct from ``settings`` (persisted user preference) and
        # from ``PluginContext.settings`` (persisted per-plugin state): these
        # apply to this run only and are never written back.
        self.launch_options: dict[str, Any] = dict(launch_options or {})

        # ── extension registries ────────────────────────────────────────
        self.commands = CommandRegistry()
        self.models: Registry[InferenceModel] = Registry(
            "model", owner_key=lambda m: m.owner,
        )
        self.panels: Registry[PanelSpec] = Registry(
            "panel", owner_key=lambda p: p.owner,
        )
        self.exporters: Registry[Exporter] = Registry(
            "exporter", owner_key=lambda e: e.owner,
        )
        for exporter in builtin_exporters():
            self.exporters.register(exporter.id, exporter)

        # ── services ────────────────────────────────────────────────────
        self.canvas = ImageCanvas()
        self.playback = PlaybackController(self)
        self.runner = InferenceRunner(self)
        self.library = MediaLibrary(self)
        self.annotations = AnnotationStore()

        self.canvas.bind_store(self.annotations)
        self.window: MainWindow | None = None

        self._media_info: MediaInfo | None = None
        # The file the annotation store currently holds. Tracked separately from
        # ``library.current`` because the library's cursor has already moved by
        # the time the change handler runs — flushing against it would write one
        # file's annotations into the next file's sidecar.
        self._store_path: Path | None = None
        self._autosave = bool(self.settings.get("annotations.autosave", True))

        self.playback.frame_changed.connect(self._on_frame)
        self.playback.opened.connect(self._on_media_opened)
        self.playback.open_failed.connect(self._on_media_open_failed)
        self.playback.error.connect(lambda msg: self.status(msg, 6000))
        self.library.current_changed.connect(self._on_library_current_changed)
        self.runner.finished.connect(self._on_inference_finished)
        self.runner.failed.connect(self._on_inference_failed)
        self.runner.started.connect(self._on_inference_started)
        self.runner.superseded.connect(lambda model_id, _rid: self._clear_busy(model_id))

    # ── media ───────────────────────────────────────────────────────────

    @property
    def current_media(self) -> Path | None:
        return self.library.current

    @property
    def media_info(self) -> MediaInfo | None:
        return self._media_info

    @property
    def current_frame(self) -> Frame | None:
        return self.playback.current_frame

    def open_media(self, path: Path) -> None:
        """Open ``path``, routing through the library so navigation stays consistent."""
        self.library.open(Path(path))

    def _on_library_current_changed(self, path: Path | None) -> None:
        # Save against the outgoing file before rebinding the store.
        self.flush_annotations()
        self._store_path = path
        if path is None:
            self.playback.close()
            self.canvas.clear()
            self.annotations.reset([], source_id="")
            return

        self.annotations.reset([], source_id=source_id_for(path))
        try:
            count = load_into_store(self.annotations, path)
        except Exception:  # noqa: BLE001 - a bad sidecar must not block opening the video
            _log.exception("Failed to load annotations for %s", path)
            count = 0
        if count:
            _log.debug("Loaded %d annotation(s) for %s", count, path.name)

        self.playback.open(path)

    def _on_media_opened(self, info: MediaInfo) -> None:
        self._media_info = info
        self.canvas.frame_info_layer.set_media(info.frame_count, info.fps)
        self.media_opened.emit(info.path)

    def _on_media_open_failed(self, path: str, message: str) -> None:
        self._media_info = None
        self.canvas.show_error(message)
        self.media_failed.emit(Path(path), message)
        self.status(message, 6000)

    def _on_frame(self, frame: Frame) -> None:
        self.canvas.set_frame(frame)
        self.frame_changed.emit(frame)

    # ── annotations ─────────────────────────────────────────────────────

    def flush_annotations(self) -> None:
        """Persist the store to the sidecar of the file it belongs to."""
        if not self._autosave or not self.annotations.is_dirty:
            return
        path = self._store_path
        if path is None:
            return
        try:
            save_store(self.annotations, path, self._media_info.size if self._media_info else None)
        except Exception:  # noqa: BLE001 - report, never lose the session over a save
            _log.exception("Failed to save annotations for %s", path)
            self.status(f"Could not save annotations for {path.name}", 6000)

    @property
    def annotated_media(self) -> Path | None:
        """The file the annotation store currently describes."""
        return self._store_path

    @property
    def autosave_annotations(self) -> bool:
        return self._autosave

    @autosave_annotations.setter
    def autosave_annotations(self, value: bool) -> None:
        self._autosave = bool(value)
        self.settings.set("annotations.autosave", self._autosave)

    # ── inference ───────────────────────────────────────────────────────

    def available_models(self, prompt: Prompt | None = None) -> list[InferenceModel]:
        """Registered models, optionally filtered to those accepting ``prompt``."""
        models = self.models.values()
        if prompt is None:
            return models
        return [m for m in models if m.supports(prompt)]

    def run_model(
        self,
        model: InferenceModel | str,
        prompt: Prompt | None = None,
        params: dict[str, Any] | None = None,
        frame: Frame | None = None,
    ) -> str | None:
        """Queue inference on the current frame. Returns the request id.

        The one call a plugin needs: it resolves the model, checks availability
        and prompt support, and reports the reason in the status bar rather than
        failing silently or raising into a Qt slot.
        """
        resolved = self.models.get(model) if isinstance(model, str) else model
        if resolved is None:
            self.status(f"No such model: {model}", 4000)
            return None

        target = frame or self.current_frame
        if target is None:
            self.status("Open a video first", 3000)
            return None

        availability = resolved.availability()
        if not availability.ok:
            self.status(f"{resolved.display_name}: {availability.message}", 8000)
            return None

        actual_prompt = prompt or WholeFramePrompt()
        if not resolved.supports(actual_prompt):
            self.status(
                f"{resolved.display_name} does not accept "
                f"{type(actual_prompt).__name__.replace('Prompt', '').lower()} prompts",
                5000,
            )
            return None

        request = InferenceRequest(
            frame=target,
            prompt=actual_prompt,
            params=resolved.resolved_params(params),
        )
        return self.runner.submit(resolved, request)

    def _on_inference_started(self, model_id: str, _request_id: str) -> None:
        model = self.models.get(model_id)
        self.canvas.busy_layer.set_running(
            model_id, model.display_name if model else model_id,
        )

    def _on_inference_finished(self, result) -> None:
        self._clear_busy(result.model_id)
        self.inference_finished.emit(result)
        model = self.models.get(result.model_id)
        name = model.display_name if model else result.model_id
        self.status(f"{name}: {result.count} result(s) in {result.elapsed_ms:.0f}ms", 4000)

    def _on_inference_failed(self, model_id: str, _request_id: str, message: str) -> None:
        self._clear_busy(model_id)
        model = self.models.get(model_id)
        name = model.display_name if model else model_id
        self.status(f"{name} failed: {message}", 8000)

    def _clear_busy(self, model_id: str) -> None:
        self.canvas.busy_layer.clear_running(model_id)

    # ── misc ────────────────────────────────────────────────────────────

    def status(self, message: str, timeout: int = 3000) -> None:
        """Show a transient status-bar message. Safe with no window attached."""
        self.status_message.emit(message, timeout)

    def add_command(self, command: Command | None = None, **kwargs: Any) -> Command:
        """Register a shell-owned command. Plugins use ``PluginContext`` instead."""
        return self.commands.add(command or Command(**kwargs))

    def shutdown(self) -> None:
        """Tear everything down in dependency order. Idempotent."""
        self.flush_annotations()
        self.plugins.deactivate_all()
        self.plugins.save_state()
        self.runner.shutdown()
        self.playback.shutdown()
        self.settings.save()
