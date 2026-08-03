"""Menu bar and keyboard map, generated from the command registry.

Nothing here knows what any particular command *does*. Menus and shortcuts are a
pure projection of :class:`~vidtriage.core.commands.CommandRegistry`, rebuilt
whenever it changes — which is what makes "register a command" the entire cost
of adding a feature, and what replaced the old hand-maintained ``eventFilter``
key ladder in ``MainWindow``.

Enabled and checked state is evaluated when a menu is about to show, not when it
is built, so a command's ``is_enabled`` sees current state without anyone having
to remember to refresh the UI.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import QMainWindow, QMenu, QWidget

from ..core.commands import Command, CommandRegistry, _MenuNode
from ..core.logging import get_logger

__all__ = ["TOP_LEVEL_ORDER", "MenuBuilder", "ShortcutBinder"]

_log = get_logger(__name__)

#: Preferred left-to-right order of top-level menus. Anything not listed is
#: appended alphabetically, so a plugin adding "Tools" lands somewhere sensible
#: without having to be told about this list.
TOP_LEVEL_ORDER = ("File", "Edit", "View", "Playback", "Annotate", "Models", "Tools", "Help")


class MenuBuilder:
    """Keeps a window's menu bar in sync with the command registry."""

    def __init__(self, window: QMainWindow, commands: CommandRegistry) -> None:
        self._window = window
        self._commands = commands
        self._rebuild_scheduled = False
        self._actions: dict[str, QAction] = {}

        commands.changed.connect(self._schedule_rebuild)
        self.rebuild()

    def _schedule_rebuild(self, _change: object = None) -> None:
        """Coalesce: a plugin registering twelve commands rebuilds the bar once."""
        if self._rebuild_scheduled:
            return
        self._rebuild_scheduled = True
        QTimer.singleShot(0, self._do_scheduled_rebuild)

    def _do_scheduled_rebuild(self) -> None:
        self._rebuild_scheduled = False
        self.rebuild()

    def rebuild(self) -> None:
        menu_bar = self._window.menuBar()
        menu_bar.clear()
        self._actions.clear()

        root = self._commands.menu_tree()
        for title in self._ordered_titles(list(root.children)):
            node = root.children[title]
            menu = menu_bar.addMenu(f"&{title}" if len(title) > 1 else title)
            self._populate(menu, node)

    @staticmethod
    def _ordered_titles(titles: list[str]) -> list[str]:
        known = [t for t in TOP_LEVEL_ORDER if t in titles]
        rest = sorted(t for t in titles if t not in TOP_LEVEL_ORDER)
        return known + rest

    def _populate(self, menu: QMenu, node: _MenuNode) -> None:
        menu.aboutToShow.connect(lambda m=menu, n=node: self._refresh_states(n))

        sections = node.sections()
        for i, section in enumerate(sections):
            if i:
                menu.addSeparator()
            for command in section:
                menu.addAction(self._action_for(command, menu))

        if node.children:
            if sections:
                menu.addSeparator()
            for title in sorted(node.children):
                submenu = menu.addMenu(title)
                self._populate(submenu, node.children[title])

    def _action_for(self, command: Command, parent: QWidget) -> QAction:
        action = QAction(command.title, parent)
        if command.description:
            action.setStatusTip(command.description)
            action.setToolTip(command.description)
        if command.shortcut:
            action.setShortcut(QKeySequence(command.shortcut))
            # The shortcut itself is owned by ShortcutBinder; showing it here but
            # not letting the action fire avoids the command running twice.
            action.setShortcutVisibleInContextMenu(True)
            action.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
        if command.checkable:
            action.setCheckable(True)
            action.setChecked(command.checked)
        action.setEnabled(command.enabled)
        action.triggered.connect(
            lambda checked=False, c=command: c.invoke(checked if c.checkable else None),
        )
        self._actions[command.id] = action
        return action

    def _refresh_states(self, node: _MenuNode) -> None:
        for command in node.commands:
            action = self._actions.get(command.id)
            if action is None:
                continue
            action.setEnabled(command.enabled)
            if command.checkable:
                action.setChecked(command.checked)


class ShortcutBinder:
    """Owns the window's ``QShortcut`` objects, derived from the registry.

    Shortcuts are bound to the window with application context rather than
    filtered out of a global event stream. That means a text field keeps its own
    keystrokes: typing "s" into a label box no longer skips to the next video,
    which the old application-wide ``eventFilter`` could not prevent.
    """

    def __init__(self, window: QWidget, commands: CommandRegistry) -> None:
        self._window = window
        self._commands = commands
        self._shortcuts: list[QShortcut] = []
        self._rebuild_scheduled = False

        commands.changed.connect(self._schedule_rebuild)
        self.rebuild()

    def _schedule_rebuild(self, _change: object = None) -> None:
        if self._rebuild_scheduled:
            return
        self._rebuild_scheduled = True
        QTimer.singleShot(0, self._do_scheduled_rebuild)

    def _do_scheduled_rebuild(self) -> None:
        self._rebuild_scheduled = False
        self.rebuild()

    def rebuild(self) -> None:
        for shortcut in self._shortcuts:
            shortcut.setParent(None)
            shortcut.deleteLater()
        self._shortcuts.clear()

        for key, command in self._commands.shortcut_map().items():
            sequence = QKeySequence(key)
            if sequence.isEmpty():
                _log.warning("Command %r has an unparseable shortcut %r", command.id, key)
                continue
            shortcut = QShortcut(sequence, self._window)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(lambda c=command: c.invoke())
            self._shortcuts.append(shortcut)

    @property
    def count(self) -> int:
        return len(self._shortcuts)
