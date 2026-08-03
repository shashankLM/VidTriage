"""Commands: the single declaration point for anything the user can trigger.

A command carries its own title, shortcut and menu location.  The app shell
builds the menu bar and the keyboard map *from the registry*, so a plugin adds a
feature by registering a command — it never touches the window, the menu code,
or a key-handling ``if`` chain.

That is the whole reason the old ``MainWindow.eventFilter`` key ladder is gone:
it was a list every new feature had to edit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .logging import get_logger
from .registry import Registry

__all__ = ["Command", "CommandRegistry"]

_log = get_logger(__name__)


@dataclass(frozen=True)
class Command:
    """A user-triggerable action.

    Args:
        id: Dotted unique id, e.g. ``"playback.toggle"``. Namespace it with your
            plugin id so two plugins cannot collide.
        title: Menu text.
        handler: Called on trigger. Takes no arguments, unless ``checkable`` is
            set, in which case it receives the new checked state.
        shortcut: Qt key sequence string, e.g. ``"Space"``, ``"Ctrl+Z"``, ``"1"``.
        menu: Slash-separated menu path, e.g. ``"View/Overlays"``. ``None`` keeps
            the command out of the menu bar (shortcut-only).
        section: Commands sharing a section are grouped between separators.
        checkable: Renders as a toggle.
        is_checked: Supplies the current toggle state when the menu opens.
        is_enabled: Consulted when the menu opens, and before a shortcut fires.
        owner: Plugin id, filled in automatically by ``PluginContext``.
        order: Sort key within a menu section.
    """

    id: str
    title: str
    handler: Callable[..., None]
    shortcut: str | None = None
    menu: str | None = None
    section: str = ""
    checkable: bool = False
    is_checked: Callable[[], bool] | None = None
    is_enabled: Callable[[], bool] | None = None
    owner: str | None = None
    order: int = 100
    description: str = ""

    @property
    def menu_path(self) -> tuple[str, ...]:
        return tuple(p for p in (self.menu or "").split("/") if p)

    @property
    def enabled(self) -> bool:
        if self.is_enabled is None:
            return True
        try:
            return bool(self.is_enabled())
        except Exception:  # noqa: BLE001 - a plugin predicate must not break the menu
            _log.exception("is_enabled for command %r raised; treating as disabled", self.id)
            return False

    @property
    def checked(self) -> bool:
        if self.is_checked is None:
            return False
        try:
            return bool(self.is_checked())
        except Exception:  # noqa: BLE001 - a plugin predicate must not break the menu
            _log.exception("is_checked for command %r raised; treating as unchecked", self.id)
            return False

    def invoke(self, checked: bool | None = None) -> None:
        """Run the handler, guarded by ``is_enabled``.

        Handler exceptions are logged rather than propagated: a Qt slot that
        raises would otherwise unwind through the C++ event loop, where the
        traceback is lost and the app may abort.
        """
        if not self.enabled:
            _log.debug("Command %r invoked while disabled; ignored", self.id)
            return
        try:
            if self.checkable:
                self.handler(bool(checked) if checked is not None else not self.checked)
            else:
                self.handler()
        except Exception:  # noqa: BLE001 - see the docstring: never unwind into Qt
            _log.exception("Command %r failed", self.id)


@dataclass
class _MenuNode:
    title: str
    children: dict[str, _MenuNode] = field(default_factory=dict)
    commands: list[Command] = field(default_factory=list)

    def sorted_commands(self) -> list[Command]:
        return sorted(self.commands, key=lambda c: (c.section, c.order, c.title))

    def sections(self) -> list[list[Command]]:
        """Commands grouped by section, in order — one separator between groups."""
        groups: dict[str, list[Command]] = {}
        for command in self.sorted_commands():
            groups.setdefault(command.section, []).append(command)
        return [groups[key] for key in sorted(groups)]


class CommandRegistry(Registry[Command]):
    """Every command in the app, plus the derived menu tree and key map."""

    def __init__(self) -> None:
        super().__init__("command", owner_key=lambda c: c.owner)

    def add(self, command: Command, *, replace: bool = False) -> Command:
        return self.register(command.id, command, replace=replace)

    def execute(self, command_id: str, checked: bool | None = None) -> None:
        self.require(command_id).invoke(checked)

    def shortcut_map(self) -> dict[str, Command]:
        """``shortcut -> command``.

        A later registration wins a contested shortcut and the loser is logged,
        because silently dropping one of two identical bindings makes for a
        genuinely baffling bug report.
        """
        mapping: dict[str, Command] = {}
        for command in self.values():
            if not command.shortcut:
                continue
            previous = mapping.get(command.shortcut)
            if previous is not None:
                _log.warning(
                    "Shortcut %r claimed by both %r and %r; %r wins",
                    command.shortcut, previous.id, command.id, command.id,
                )
            mapping[command.shortcut] = command
        return mapping

    def menu_tree(self) -> _MenuNode:
        """Nested menu structure built from every command's ``menu`` path."""
        root = _MenuNode("")
        for command in self.values():
            path = command.menu_path
            if not path:
                continue
            node = root
            for part in path:
                node = node.children.setdefault(part, _MenuNode(part))
            node.commands.append(command)
        return root
