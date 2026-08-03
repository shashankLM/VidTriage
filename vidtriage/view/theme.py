"""Themes and the manager that broadcasts changes.

The previous version required :class:`MainWindow` to know every themed widget
and call ``apply_theme()`` on each one by hand — so a new widget silently kept
the old colours until someone remembered to add it to that list. Here widgets
subscribe themselves via :class:`ThemedMixin` and are unsubscribed
automatically when Qt destroys them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from PySide6.QtGui import QColor

from ..core.events import Event
from ..core.logging import get_logger

try:  # pragma: no cover - present in every supported PySide6 build
    from shiboken6 import isValid as _is_valid
except ImportError:  # pragma: no cover
    def _is_valid(_obj: object) -> bool:
        return True

__all__ = [
    "THEMES",
    "Theme",
    "ThemeManager",
    "ThemedMixin",
    "current_theme",
    "set_theme",
    "theme_manager",
]

_log = get_logger(__name__)


@dataclass(frozen=True)
class Theme:
    """A complete colour scheme.

    Roles are named for what they *are used for*, not what they look like, so a
    light and a dark theme can both satisfy the same set.
    """

    name: str
    dark: bool

    # Window / global
    window_bg: str
    window_fg: str
    panel_bg: str

    # Canvas
    canvas_bg: str
    error_fg: str

    # Lists and focus
    focus_border: str
    unfocus_border: str
    pending_fg: str
    classified_fg: str
    error_item_fg: str

    # Misc chrome
    table_alt_bg: str
    info_fg: str

    # Canvas overlays. Defaulted for dark themes; light themes override.
    hud_fg: str = "#f0f0f0"
    hud_bg: str = "#000000b4"
    selection_fg: str = "#ffd54f"
    handle_fg: str = "#ffffff"
    handle_bg: str = "#1e1e1e"
    crosshair_fg: str = "#80ffffff"
    guide_fg: str = "#ff5252"

    # ── colour helpers ──────────────────────────────────────────────────

    def color(self, role: str) -> QColor:
        """``QColor`` for a role name, e.g. ``theme.color("focus_border")``."""
        value = getattr(self, role, None)
        if not isinstance(value, str):
            raise KeyError(f"unknown theme role {role!r}")
        return QColor(value)

    # ── stylesheets ─────────────────────────────────────────────────────

    def app_stylesheet(self) -> str:
        return "".join((
            f"QMainWindow, QDialog {{ background: {self.window_bg}; color: {self.window_fg}; }}",
            f"QMenuBar {{ background: {self.panel_bg}; color: {self.window_fg}; }}",
            f"QMenuBar::item:selected {{ background: {self.focus_border}; }}",
            f"QMenu {{ background: {self.panel_bg}; color: {self.window_fg}; }}",
            f"QMenu::item:selected {{ background: {self.focus_border}; }}",
            f"QMenu::item:disabled {{ color: {self.info_fg}; }}",
            f"QLabel {{ color: {self.window_fg}; background: transparent; }}",
            f"QPushButton {{ background: {self.panel_bg}; color: {self.window_fg};"
            f" border: 1px solid {self.unfocus_border}; padding: 4px 8px; }}",
            f"QPushButton:hover {{ background: {self.focus_border}; }}",
            f"QPushButton:disabled {{ color: {self.info_fg}; border-color: {self.unfocus_border}; }}",
            f"QToolButton {{ background: {self.panel_bg}; color: {self.window_fg};"
            f" border: 1px solid {self.unfocus_border}; padding: 3px; }}",
            f"QToolButton:checked {{ background: {self.focus_border}; }}",
            f"QSlider::groove:horizontal {{ background: {self.unfocus_border}; height: 6px; }}",
            f"QSlider::handle:horizontal {{ background: {self.focus_border}; width: 12px; margin: -4px 0; }}",
            f"QListWidget, QTreeWidget, QTableWidget {{ background: {self.panel_bg};"
            f" color: {self.window_fg}; alternate-background-color: {self.table_alt_bg}; }}",
            f"QHeaderView::section {{ background: {self.table_alt_bg}; color: {self.window_fg};"
            f" border: none; padding: 4px; }}",
            f"QSplitter::handle {{ background: {self.unfocus_border}; }}",
            f"QFrame {{ background: {self.panel_bg}; }}",
            f"QTextEdit, QTextBrowser, QLineEdit, QPlainTextEdit {{ background: {self.panel_bg};"
            f" color: {self.window_fg}; border: 1px solid {self.unfocus_border}; }}",
            f"QComboBox, QSpinBox, QDoubleSpinBox {{ background: {self.panel_bg};"
            f" color: {self.window_fg}; border: 1px solid {self.unfocus_border}; padding: 2px 4px; }}",
            f"QStatusBar {{ background: {self.panel_bg}; color: {self.window_fg}; }}",
            f"QDockWidget {{ color: {self.window_fg}; titlebar-close-icon: none; }}",
            f"QDockWidget::title {{ background: {self.table_alt_bg}; padding: 4px; }}",
            f"QScrollBar:vertical, QScrollBar:horizontal {{ background: {self.panel_bg}; }}",
            f"QScrollBar::handle {{ background: {self.unfocus_border}; border-radius: 3px; }}",
            f"QToolTip {{ background: {self.panel_bg}; color: {self.window_fg};"
            f" border: 1px solid {self.focus_border}; }}",
        ))

    def focused_list_style(self) -> str:
        return f"QListWidget {{ border: 2px solid {self.focus_border}; background: {self.panel_bg}; }}"

    def unfocused_list_style(self) -> str:
        return f"QListWidget {{ border: 2px solid {self.unfocus_border}; background: {self.panel_bg}; }}"


DARK = Theme(
    name="Dark", dark=True,
    window_bg="#1e1e1e", window_fg="#d4d4d4", panel_bg="#252526",
    canvas_bg="#141414", error_fg="#ff6666",
    focus_border="#42a5f5", unfocus_border="#444444",
    pending_fg="#cccccc", classified_fg="#66bb6a", error_item_fg="#ef5350",
    table_alt_bg="#2a2a2a", info_fg="#aaaaaa",
)

LIGHT = Theme(
    name="Light", dark=False,
    window_bg="#f5f5f5", window_fg="#1e1e1e", panel_bg="#ffffff",
    canvas_bg="#d8d8d8", error_fg="#c62828",
    focus_border="#1976d2", unfocus_border="#bdbdbd",
    pending_fg="#424242", classified_fg="#2e7d32", error_item_fg="#c62828",
    table_alt_bg="#eeeeee", info_fg="#757575",
    hud_fg="#ffffff", hud_bg="#000000a0",
    selection_fg="#e65100", handle_fg="#1e1e1e", handle_bg="#ffffff",
    crosshair_fg="#90000000",
)

WARM_GRAY = replace(
    LIGHT, name="Warm Gray",
    window_bg="#f0eee9", window_fg="#3b3735", panel_bg="#fafaf8",
    canvas_bg="#ddd9d2", error_fg="#bf360c",
    focus_border="#8d6e63", unfocus_border="#c8c3bc",
    pending_fg="#5d5652", classified_fg="#558b2f", error_item_fg="#bf360c",
    table_alt_bg="#edeae4", info_fg="#8a8279",
    selection_fg="#bf360c",
)

SOLARIZED_LIGHT = replace(
    LIGHT, name="Solarized Light",
    window_bg="#fdf6e3", window_fg="#657b83", panel_bg="#eee8d5",
    canvas_bg="#ded7c0", error_fg="#dc322f",
    focus_border="#268bd2", unfocus_border="#b8b0a0",
    pending_fg="#586e75", classified_fg="#859900", error_item_fg="#dc322f",
    table_alt_bg="#f5eed8", info_fg="#93a1a1",
    selection_fg="#cb4b16",
)

NORD_LIGHT = replace(
    LIGHT, name="Nord Light",
    window_bg="#eceff4", window_fg="#2e3440", panel_bg="#e5e9f0",
    canvas_bg="#ccd3de", error_fg="#bf616a",
    focus_border="#5e81ac", unfocus_border="#b4bcc8",
    pending_fg="#3b4252", classified_fg="#a3be8c", error_item_fg="#bf616a",
    table_alt_bg="#e8ecf1", info_fg="#7b88a1",
    selection_fg="#d08770",
)

NORD_DARK = replace(
    DARK, name="Nord Dark",
    window_bg="#2e3440", window_fg="#d8dee9", panel_bg="#3b4252",
    canvas_bg="#242933", error_fg="#bf616a",
    focus_border="#88c0d0", unfocus_border="#4c566a",
    pending_fg="#d8dee9", classified_fg="#a3be8c", error_item_fg="#bf616a",
    table_alt_bg="#434c5e", info_fg="#8f9bb3",
    selection_fg="#ebcb8b",
)

THEMES: dict[str, Theme] = {
    t.name: t for t in (DARK, NORD_DARK, LIGHT, WARM_GRAY, SOLARIZED_LIGHT, NORD_LIGHT)
}


class ThemeManager:
    """Holds the active theme and notifies subscribers when it changes."""

    def __init__(self, initial: Theme = DARK) -> None:
        self._theme = initial
        self.changed: Event[Theme] = Event("theme.changed")

    @property
    def theme(self) -> Theme:
        return self._theme

    @property
    def names(self) -> list[str]:
        return list(THEMES)

    def set(self, name_or_theme: str | Theme) -> Theme:
        theme = (
            name_or_theme if isinstance(name_or_theme, Theme)
            else THEMES.get(name_or_theme, self._theme)
        )
        if theme is self._theme:
            return theme
        self._theme = theme
        _log.debug("Theme -> %s", theme.name)
        self.changed.emit(theme)
        return theme


_manager = ThemeManager()


def theme_manager() -> ThemeManager:
    return _manager


def current_theme() -> Theme:
    return _manager.theme


def set_theme(name: str) -> Theme:
    return _manager.set(name)


class ThemedMixin:
    """Mix into a ``QWidget`` to receive theme updates automatically.

    Call :meth:`init_theme` once from ``__init__`` and implement
    :meth:`apply_theme`. The subscription is dropped when the C++ object dies,
    so a widget closed mid-session never receives a call on a dangling pointer.
    """

    def init_theme(self) -> None:
        theme_manager().changed.connect(self._dispatch_theme)
        self.apply_theme(current_theme())

    def _dispatch_theme(self, theme: Theme) -> None:
        if not _is_valid(self):
            return
        self.apply_theme(theme)

    def apply_theme(self, theme: Theme) -> None:  # pragma: no cover - overridden
        raise NotImplementedError
