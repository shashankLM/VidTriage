"""Plugin discovery, ordering, activation and persistence.

Three discovery sources, all equal citizens:

1. **Built-ins** — every subpackage of :mod:`label_kit.plugins.builtin`.
2. **Installed packages** — anything advertising the ``label_kit.plugins``
   entry-point group, so ``pip install label_kit-sam3`` is enough to add a model.
3. **Drop-ins** — ``.py`` files and packages under ``~/.labelkit/plugins/``,
   for one-off local extensions with no packaging ceremony.

Each source is scanned defensively. A plugin that fails to import, reports
itself unavailable, or raises during activation is recorded with its error and
skipped — the app still starts, and the plugin manager shows why that one is
missing. A broken third-party plugin must never be able to prevent the user from
opening their videos.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import pkgutil
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.errors import PluginLoadError
from ..core.logging import get_logger
from ..core.registry import Registry
from .api import Plugin, PluginContext
from .models import Availability

if TYPE_CHECKING:
    from ..app.context import AppContext

__all__ = ["ENTRY_POINT_GROUP", "PLUGIN_ATTR", "PluginManager", "PluginState"]

_log = get_logger(__name__)

#: Module attribute a plugin module must expose: a :class:`Plugin` subclass.
PLUGIN_ATTR = "PLUGIN"
ENTRY_POINT_GROUP = "labelkit.plugins"
_MAX_DEPENDENCY_PASSES = 32


@dataclass
class PluginState:
    """Bookkeeping for one discovered plugin."""

    plugin: Plugin
    origin: str
    enabled: bool = True
    active: bool = False
    error: str = ""
    availability: Availability = field(default_factory=Availability.available)
    context: PluginContext | None = None

    @property
    def id(self) -> str:
        return self.plugin.id

    @property
    def can_activate(self) -> bool:
        return self.enabled and self.availability.ok and not self.error

    @property
    def status(self) -> str:
        if self.active:
            return "active"
        if self.error:
            return "error"
        if not self.availability.ok:
            return "unavailable"
        return "disabled" if not self.enabled else "inactive"


class PluginManager:
    """Owns the plugin lifecycle for one application instance."""

    def __init__(self, state_file: Path | None = None) -> None:
        self.plugins: Registry[Plugin] = Registry("plugin")
        self.states: dict[str, PluginState] = {}
        self._state_file = state_file
        self._persisted = self._load_persisted()

    # ── discovery ───────────────────────────────────────────────────────

    def discover(self, user_dir: Path | None = None) -> list[PluginState]:
        """Find plugins from all three sources. Safe to call once at startup."""
        self._discover_builtin()
        self._discover_entry_points()
        if user_dir is not None:
            self._discover_user_dir(user_dir)
        # Discovery and activation are lifecycle bookkeeping, logged at debug
        # throughout this class: a host that wants to show the user what loaded
        # reads `states` and renders it, rather than relying on the log. Failures
        # stay at warning or error.
        _log.debug(
            "Discovered %d plugin(s): %s",
            len(self.states), ", ".join(sorted(self.states)) or "none",
        )
        return list(self.states.values())

    def _discover_builtin(self) -> None:
        from . import builtin

        for module_info in pkgutil.iter_modules(builtin.__path__):
            name = f"{builtin.__name__}.{module_info.name}"
            try:
                module = importlib.import_module(name)
            except Exception:  # noqa: BLE001 - a plugin may raise anything on import
                self._record_import_failure(module_info.name, "builtin", name)
                continue
            self._register_from_module(module, origin="builtin")

    def _discover_entry_points(self) -> None:
        try:
            from importlib.metadata import entry_points

            found = entry_points(group=ENTRY_POINT_GROUP)
        except Exception:  # noqa: BLE001 - metadata backends vary across envs
            _log.debug("Entry-point discovery unavailable", exc_info=True)
            return

        for entry in found:
            try:
                loaded = entry.load()
            except Exception:  # noqa: BLE001 - third-party code; contain the fault
                self._record_import_failure(entry.name, "package", entry.value)
                continue
            module_or_class = loaded
            if isinstance(module_or_class, type) and issubclass(module_or_class, Plugin):
                self._register_class(module_or_class, origin=f"package:{entry.value}")
            else:
                self._register_from_module(module_or_class, origin=f"package:{entry.value}")

    def _discover_user_dir(self, directory: Path) -> None:
        if not directory.is_dir():
            return
        for path in sorted(directory.iterdir()):
            is_module = path.suffix == ".py" and not path.name.startswith("_")
            is_package = path.is_dir() and (path / "__init__.py").exists()
            if not (is_module or is_package):
                continue

            target = path / "__init__.py" if is_package else path
            module_name = f"labelkit_userplugin_{path.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, target)
                if spec is None or spec.loader is None:
                    raise PluginLoadError(f"cannot build import spec for {target}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            except Exception:  # noqa: BLE001 - user drop-in; contain the fault
                self._record_import_failure(path.stem, "user", str(path))
                continue
            self._register_from_module(module, origin=f"user:{path.name}")

    def _register_from_module(self, module: Any, origin: str) -> None:
        candidate = getattr(module, PLUGIN_ATTR, None)
        if candidate is None:
            _log.debug("%s exposes no %s attribute; skipping", module, PLUGIN_ATTR)
            return
        if not (isinstance(candidate, type) and issubclass(candidate, Plugin)):
            _log.warning("%s.%s is not a Plugin subclass; skipping", module, PLUGIN_ATTR)
            return
        self._register_class(candidate, origin)

    def _register_class(self, cls: type[Plugin], origin: str) -> None:
        try:
            plugin = cls()
        except Exception:  # noqa: BLE001 - a plugin constructor may raise anything
            self._record_import_failure(getattr(cls, "id", cls.__name__), origin, cls.__name__)
            return

        if not plugin.id:
            _log.warning("Plugin class %s has no id; skipping", cls.__name__)
            return
        if plugin.id in self.states:
            _log.warning(
                "Plugin id %r already provided by %s; ignoring %s",
                plugin.id, self.states[plugin.id].origin, origin,
            )
            return

        try:
            availability = plugin.availability()
        except Exception as exc:  # noqa: BLE001 - a bad check must not stop discovery
            _log.exception("availability() failed for plugin %r", plugin.id)
            availability = Availability(False, reason=f"availability check failed: {exc}")

        self.plugins.register(plugin.id, plugin)
        self.states[plugin.id] = PluginState(
            plugin=plugin,
            origin=origin,
            enabled=self._initial_enabled(plugin),
            availability=availability,
        )

    def _initial_enabled(self, plugin: Plugin) -> bool:
        """Remembered choice wins; otherwise the plugin's own default."""
        if plugin.essential:
            return True
        if plugin.id in set(self._persisted.get("disabled", [])):
            return False
        if plugin.id in set(self._persisted.get("enabled", [])):
            return True
        return plugin.default_enabled

    def _record_import_failure(self, plugin_id: str, origin: str, target: str) -> None:
        message = traceback.format_exc(limit=6)
        _log.error("Failed to load plugin %r from %s:\n%s", plugin_id, target, message)
        placeholder = _BrokenPlugin(plugin_id, target)
        if plugin_id in self.states:
            return
        self.plugins.register(plugin_id, placeholder, replace=True)
        self.states[plugin_id] = PluginState(
            plugin=placeholder, origin=origin, enabled=False, error=message,
        )

    # ── activation ──────────────────────────────────────────────────────

    def activation_order(self) -> list[str]:
        """Plugin ids ordered so dependencies come first.

        A dependency cycle or a missing requirement is reported and the affected
        plugins are appended at the end rather than dropped, so the failure
        surfaces as one clear activation error instead of a silent absence.
        """
        remaining = {
            state.id: set(state.plugin.requires) & set(self.states)
            for state in self.states.values()
        }
        ordered: list[str] = []
        for _ in range(_MAX_DEPENDENCY_PASSES):
            ready = sorted(pid for pid, deps in remaining.items() if not deps - set(ordered))
            if not ready:
                break
            ordered.extend(ready)
            for pid in ready:
                remaining.pop(pid)
        if remaining:
            _log.error(
                "Unresolvable plugin dependencies for: %s", ", ".join(sorted(remaining)),
            )
            ordered.extend(sorted(remaining))
        return ordered

    def activate_all(self, app: AppContext) -> None:
        for plugin_id in self.activation_order():
            state = self.states[plugin_id]
            if state.can_activate:
                self.activate(plugin_id, app)

    def activate(self, plugin_id: str, app: AppContext) -> bool:
        state = self.states.get(plugin_id)
        if state is None or state.active:
            return False
        if not state.availability.ok:
            _log.debug("Skipping unavailable plugin %r: %s", plugin_id, state.availability.reason)
            return False

        missing = [
            dep for dep in state.plugin.requires
            if dep not in self.states or not self.states[dep].active
        ]
        if missing:
            state.error = f"requires inactive plugin(s): {', '.join(missing)}"
            _log.error("Cannot activate %r — %s", plugin_id, state.error)
            return False

        settings = self._persisted.setdefault("settings", {}).setdefault(plugin_id, {})
        context = PluginContext(plugin_id, app, settings)
        try:
            state.plugin._bind(context)
            state.plugin.activate(context)
        except Exception:  # noqa: BLE001 - one bad plugin must not stop the app
            state.error = traceback.format_exc(limit=6)
            _log.error("Plugin %r failed to activate:\n%s", plugin_id, state.error)
            # Roll back whatever it managed to register before blowing up.
            context.dispose()
            state.plugin._unbind()
            return False

        state.context = context
        state.active = True
        state.error = ""
        _log.debug("Activated plugin %r", plugin_id)
        return True

    def deactivate(self, plugin_id: str) -> bool:
        state = self.states.get(plugin_id)
        if state is None or not state.active:
            return False
        try:
            state.plugin.deactivate()
        except Exception:  # noqa: BLE001 - tear down regardless; leaks beat a hang
            _log.exception("Plugin %r raised during deactivate; removing anyway", plugin_id)
        if state.context is not None:
            state.context.dispose()
        state.plugin._unbind()
        state.context = None
        state.active = False
        _log.debug("Deactivated plugin %r", plugin_id)
        return True

    def deactivate_all(self) -> None:
        for plugin_id in reversed(self.activation_order()):
            self.deactivate(plugin_id)

    # ── enable / disable ────────────────────────────────────────────────

    def is_enabled(self, plugin_id: str) -> bool:
        state = self.states.get(plugin_id)
        return bool(state and state.enabled)

    def set_enabled(self, plugin_id: str, enabled: bool, app: AppContext) -> bool:
        """Turn a plugin on or off immediately, and remember the choice."""
        state = self.states.get(plugin_id)
        if state is None:
            return False
        if state.plugin.essential and not enabled:
            _log.warning("Plugin %r is essential and cannot be disabled", plugin_id)
            return False

        state.enabled = enabled
        if enabled and not state.active:
            self.activate(plugin_id, app)
        elif not enabled and state.active:
            # Anything depending on this must go down first.
            for other in reversed(self.activation_order()):
                if other != plugin_id and plugin_id in self.states[other].plugin.requires:
                    self.deactivate(other)
            self.deactivate(plugin_id)

        self._persist_enabled_flags()
        self.save_state()
        return True

    def active_plugins(self) -> list[Plugin]:
        return [s.plugin for s in self.states.values() if s.active]

    def settings_for(self, plugin_id: str) -> dict[str, Any]:
        return self._persisted.setdefault("settings", {}).setdefault(plugin_id, {})

    # ── persistence ─────────────────────────────────────────────────────

    def _load_persisted(self) -> dict[str, Any]:
        if self._state_file is None or not self._state_file.exists():
            return {}
        try:
            data = json.loads(self._state_file.read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            _log.warning("Ignoring unreadable plugin state %s: %s", self._state_file, exc)
            return {}

    def _persist_enabled_flags(self) -> None:
        self._persisted["disabled"] = sorted(
            pid for pid, state in self.states.items() if not state.enabled
        )
        self._persisted["enabled"] = sorted(
            pid for pid, state in self.states.items() if state.enabled
        )

    def save_state(self) -> None:
        if self._state_file is None:
            return
        self._persist_enabled_flags()
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            self._state_file.write_text(json.dumps(self._persisted, indent=2))
        except OSError as exc:
            _log.warning("Could not save plugin state to %s: %s", self._state_file, exc)


class _BrokenPlugin(Plugin):
    """Stand-in for a plugin that could not even be imported.

    Keeping a placeholder means the plugin manager can show *why* something the
    user installed is missing, instead of it silently not existing.
    """

    def __init__(self, plugin_id: str, target: str) -> None:
        super().__init__()
        self.id = plugin_id
        self.name = f"{plugin_id} (failed to load)"
        self.description = f"Could not be imported from {target}"
        self.default_enabled = False

    def availability(self) -> Availability:
        return Availability(False, reason="failed to import", remedy="see the log for the traceback")

    def activate(self, ctx: PluginContext) -> None:
        raise PluginLoadError(f"plugin {self.id!r} failed to import")
