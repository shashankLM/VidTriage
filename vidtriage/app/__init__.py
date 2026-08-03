"""The application shell: context, window, menus, transport, dialogs."""

from __future__ import annotations

from .commands import RegistryCommands, register_core_commands
from .context import AppContext
from .library import MEDIA_EXTENSIONS, MediaLibrary, discover_media
from .menus import MenuBuilder, ShortcutBinder
from .transport import TransportBar, format_timecode
from .window import MainWindow

__all__ = [
    "MEDIA_EXTENSIONS",
    "AppContext",
    "MainWindow",
    "MediaLibrary",
    "MenuBuilder",
    "RegistryCommands",
    "ShortcutBinder",
    "TransportBar",
    "discover_media",
    "format_timecode",
    "register_core_commands",
]
