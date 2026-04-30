from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    name: str

    # Window / global
    window_bg: str
    window_fg: str
    panel_bg: str

    # Player
    player_bg: str
    error_fg: str

    # File explorer
    focus_border: str
    unfocus_border: str
    pending_fg: str
    classified_fg: str
    error_item_fg: str

    # Misc
    table_alt_bg: str
    info_fg: str

    def app_stylesheet(self) -> str:
        return (
            f"QMainWindow, QDialog {{ background: {self.window_bg}; color: {self.window_fg}; }}"
            f"QMenuBar {{ background: {self.panel_bg}; color: {self.window_fg}; }}"
            f"QMenuBar::item:selected {{ background: {self.focus_border}; }}"
            f"QMenu {{ background: {self.panel_bg}; color: {self.window_fg}; }}"
            f"QMenu::item:selected {{ background: {self.focus_border}; }}"
            f"QLabel {{ color: {self.window_fg}; }}"
            f"QPushButton {{ background: {self.panel_bg}; color: {self.window_fg}; border: 1px solid {self.unfocus_border}; padding: 4px 8px; }}"
            f"QPushButton:hover {{ background: {self.focus_border}; }}"
            f"QSlider::groove:horizontal {{ background: {self.unfocus_border}; height: 6px; }}"
            f"QSlider::handle:horizontal {{ background: {self.focus_border}; width: 12px; margin: -4px 0; }}"
            f"QListWidget {{ background: {self.panel_bg}; color: {self.window_fg}; }}"
            f"QSplitter::handle {{ background: {self.unfocus_border}; }}"
            f"QFrame {{ background: {self.panel_bg}; }}"
            f"QTextEdit, QTextBrowser {{ background: {self.panel_bg}; color: {self.window_fg}; }}"
            f"QStatusBar {{ background: {self.panel_bg}; color: {self.window_fg}; }}"
        )

    def focused_list_style(self) -> str:
        return f"QListWidget {{ border: 2px solid {self.focus_border}; background: {self.panel_bg}; }}"

    def unfocused_list_style(self) -> str:
        return f"QListWidget {{ border: 2px solid {self.unfocus_border}; background: {self.panel_bg}; }}"

    def player_style(self) -> str:
        return f"background: {self.player_bg};"

    def player_error_style(self) -> str:
        return f"color: {self.error_fg}; font-size: 16px; background: {self.player_bg};"


DARK = Theme(
    name="Dark",
    window_bg="#1e1e1e",
    window_fg="#d4d4d4",
    panel_bg="#252526",
    player_bg="#1a1a1a",
    error_fg="#ff6666",
    focus_border="#42a5f5",
    unfocus_border="#444444",
    pending_fg="#cccccc",
    classified_fg="#66bb6a",
    error_item_fg="#ef5350",
    table_alt_bg="#2a2a2a",
    info_fg="#aaaaaa",
)

LIGHT = Theme(
    name="Light",
    window_bg="#f5f5f5",
    window_fg="#1e1e1e",
    panel_bg="#ffffff",
    player_bg="#e0e0e0",
    error_fg="#c62828",
    focus_border="#1976d2",
    unfocus_border="#bdbdbd",
    pending_fg="#424242",
    classified_fg="#2e7d32",
    error_item_fg="#c62828",
    table_alt_bg="#eeeeee",
    info_fg="#757575",
)

WARM_GRAY = Theme(
    name="Warm Gray",
    window_bg="#f0eee9",
    window_fg="#3b3735",
    panel_bg="#fafaf8",
    player_bg="#e8e5df",
    error_fg="#bf360c",
    focus_border="#8d6e63",
    unfocus_border="#c8c3bc",
    pending_fg="#5d5652",
    classified_fg="#558b2f",
    error_item_fg="#bf360c",
    table_alt_bg="#edeae4",
    info_fg="#8a8279",
)

SOLARIZED_LIGHT = Theme(
    name="Solarized Light",
    window_bg="#fdf6e3",
    window_fg="#657b83",
    panel_bg="#eee8d5",
    player_bg="#e8e1cc",
    error_fg="#dc322f",
    focus_border="#268bd2",
    unfocus_border="#b8b0a0",
    pending_fg="#586e75",
    classified_fg="#859900",
    error_item_fg="#dc322f",
    table_alt_bg="#f5eed8",
    info_fg="#93a1a1",
)

NORD_LIGHT = Theme(
    name="Nord Light",
    window_bg="#eceff4",
    window_fg="#2e3440",
    panel_bg="#e5e9f0",
    player_bg="#d8dee9",
    error_fg="#bf616a",
    focus_border="#5e81ac",
    unfocus_border="#b4bcc8",
    pending_fg="#3b4252",
    classified_fg="#a3be8c",
    error_item_fg="#bf616a",
    table_alt_bg="#e8ecf1",
    info_fg="#7b88a1",
)

THEMES: dict[str, Theme] = {
    "Dark": DARK,
    "Light": LIGHT,
    "Warm Gray": WARM_GRAY,
    "Solarized Light": SOLARIZED_LIGHT,
    "Nord Light": NORD_LIGHT,
}

_current: Theme = DARK


def current_theme() -> Theme:
    return _current


def set_theme(name: str) -> Theme:
    global _current
    _current = THEMES[name]
    return _current
