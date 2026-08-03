"""The plugin contract.

A plugin is a bundle of *contributions*: commands, tools, overlay layers,
inference models, dock panels, exporters. It declares them in
:meth:`Plugin.activate` and never touches the main window, the menu bar or the
key handler — those are generated from the registries.

That inversion is the point. Adding a feature is:

.. code-block:: python

    class MyPlugin(Plugin):
        id = "myplugin"
        name = "My Feature"

        def activate(self, ctx: PluginContext) -> None:
            ctx.add_layer(MyOverlay())
            ctx.add_tool(MyTool())
            ctx.add_model(MyModel())
            ctx.add_command(id="myplugin.go", title="Go", shortcut="G",
                            menu="Tools", handler=self._go)

    PLUGIN = MyPlugin

Drop that in ``~/.labelkit/plugins/`` and it is picked up on next launch.

Everything registered through :class:`PluginContext` is tagged with the plugin
id and torn down automatically on deactivate, so disabling a plugin genuinely
removes its menu entries, shortcuts, overlays and models rather than leaving
half of them wired up.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ..core.commands import Command
from ..core.events import Event, Subscription
from ..core.logging import get_logger
from .models import Availability, InferenceModel

if TYPE_CHECKING:  # avoids a cycle: app imports plugins, plugins type-hint app
    from ..app.context import AppContext
    from ..persistence.exporters import Exporter
    from ..view.layers import Layer
    from ..view.tools import Tool

__all__ = ["PanelArea", "PanelSpec", "Plugin", "PluginContext"]

_log = get_logger(__name__)

PanelArea = Literal["left", "right", "bottom", "top"]


@dataclass
class PanelSpec:
    """A dock panel a plugin wants in the window.

    The widget is built lazily, on first show, so a plugin with an expensive
    panel does not slow down startup for users who never open it.
    """

    id: str
    title: str
    factory: Callable[[], Any]
    area: PanelArea = "right"
    visible_by_default: bool = False
    shortcut: str | None = None
    owner: str | None = None


class PluginContext:
    """Handed to :meth:`Plugin.activate`; the only way a plugin talks to the app.

    Every ``add_*`` call tags the contribution with the plugin id and remembers
    it, so :meth:`dispose` can undo the lot. Plugins should not register into
    the app's registries directly.
    """

    def __init__(self, plugin_id: str, app: AppContext, settings: dict[str, Any]) -> None:
        self.plugin_id = plugin_id
        self.app = app
        self.settings = settings
        self.log = get_logger(f"plugin.{plugin_id}")

        self._commands: list[str] = []
        self._tools: list[str] = []
        self._layers: list[str] = []
        self._models: list[str] = []
        self._panels: list[str] = []
        self._exporters: list[str] = []
        self._subscriptions: list[Subscription] = []
        self._qt_connections: list[tuple[Any, Any]] = []

    # ── contributions ───────────────────────────────────────────────────

    def add_command(self, command: Command | None = None, **kwargs: Any) -> Command:
        """Register a command. Pass a :class:`Command` or its fields as kwargs."""
        if command is None:
            command = Command(**kwargs)
        command = _with_owner(command, self.plugin_id)
        self.app.commands.add(command)
        self._commands.append(command.id)
        return command

    def add_tool(self, tool: Tool) -> Tool:
        tool.owner = self.plugin_id
        self.app.canvas.register_tool(tool)
        self._tools.append(tool.id)
        return tool

    def add_layer(self, layer: Layer) -> Layer:
        layer.owner = self.plugin_id
        self.app.canvas.add_layer(layer)
        self._layers.append(layer.id)
        return layer

    def add_model(self, model: InferenceModel) -> InferenceModel:
        model.owner = self.plugin_id
        self.app.models.register(model.id, model)
        self._models.append(model.id)
        return model

    def add_panel(self, panel: PanelSpec | None = None, **kwargs: Any) -> PanelSpec:
        if panel is None:
            panel = PanelSpec(**kwargs)
        panel.owner = self.plugin_id
        self.app.panels.register(panel.id, panel)
        self._panels.append(panel.id)
        return panel

    def add_exporter(self, exporter: Exporter) -> Exporter:
        exporter.owner = self.plugin_id
        self.app.exporters.register(exporter.id, exporter)
        self._exporters.append(exporter.id)
        return exporter

    # ── subscriptions ───────────────────────────────────────────────────

    def subscribe(self, event: Event, handler: Callable) -> Subscription:
        """Connect to a core :class:`Event`, auto-disconnected on deactivate."""
        subscription = event.connect(handler)
        self._subscriptions.append(subscription)
        return subscription

    def connect(self, signal: Any, handler: Callable) -> None:
        """Connect to a Qt signal, auto-disconnected on deactivate."""
        signal.connect(handler)
        self._qt_connections.append((signal, handler))

    # ── teardown ────────────────────────────────────────────────────────

    def dispose(self) -> None:
        """Remove every contribution and subscription this plugin made."""
        for subscription in self._subscriptions:
            subscription.disconnect()
        self._subscriptions.clear()

        for signal, handler in self._qt_connections:
            # The emitting object may already have been destroyed by Qt, in
            # which case the connection is gone and there is nothing to undo.
            with suppress(RuntimeError, TypeError):
                signal.disconnect(handler)
        self._qt_connections.clear()

        for tool_id in self._tools:
            self.app.canvas.unregister_tool(tool_id)
        for layer_id in self._layers:
            self.app.canvas.remove_layer(layer_id)
        for command_id in self._commands:
            self.app.commands.unregister(command_id)
        for model_id in self._models:
            self.app.models.unregister(model_id)
        for panel_id in self._panels:
            self.app.panels.unregister(panel_id)
        for exporter_id in self._exporters:
            self.app.exporters.unregister(exporter_id)

        for names in (
            self._tools, self._layers, self._commands,
            self._models, self._panels, self._exporters,
        ):
            names.clear()


class Plugin(ABC):
    """Base class for every plugin."""

    #: Unique id. Also the namespace for this plugin's contribution ids.
    id: str = ""
    #: Name shown in the plugin manager.
    name: str = ""
    #: One-line description.
    description: str = ""
    version: str = "1.0"
    #: Ids of plugins that must activate first.
    requires: tuple[str, ...] = ()
    #: Whether the plugin is on for a fresh install.
    default_enabled: bool = True
    #: A built-in the app depends on cannot be disabled from the UI.
    essential: bool = False

    def __init__(self) -> None:
        self._context: PluginContext | None = None

    @property
    def context(self) -> PluginContext | None:
        return self._context

    @property
    def is_active(self) -> bool:
        return self._context is not None

    def availability(self) -> Availability:
        """Whether the plugin can run. Must be cheap — no heavy imports."""
        return Availability.available()

    @abstractmethod
    def activate(self, ctx: PluginContext) -> None:
        """Register contributions. Called once when the plugin is enabled."""

    def deactivate(self) -> None:
        """Release resources. Contributions are removed automatically."""

    # ── used by the manager ─────────────────────────────────────────────

    def _bind(self, ctx: PluginContext) -> None:
        self._context = ctx

    def _unbind(self) -> None:
        self._context = None

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.id!r}>"


def _with_owner(command: Command, owner: str) -> Command:
    from dataclasses import replace

    return command if command.owner == owner else replace(command, owner=owner)
