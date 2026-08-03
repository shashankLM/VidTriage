"""Entry point: parse arguments, build the context, discover plugins, show the window."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Silence a noisy Wayland warning before Qt is imported.
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.wayland.textinput=false")

from PySide6.QtWidgets import QApplication

from . import __version__
from .app.context import AppContext
from .app.library import discover_media
from .app.startup_report import report_startup
from .app.window import MainWindow
from .core.logging import configure_logging, get_logger
from .persistence.settings import (
    CONFIG_DIR,
    Settings,
    user_plugin_dir,
)
from .plugins.manager import PluginManager

_log = get_logger(__name__)


def _help_formatter() -> type[argparse.HelpFormatter]:
    """``rich_argparse``'s formatter when it is installed, argparse's otherwise.

    Note for anyone editing the help text below: this formatter reads rich
    markup, so a literal ``[`` in a help string is a tag opener and will be
    eaten. Keep square brackets out, or escape them.
    """
    try:
        from rich_argparse import RichHelpFormatter
    except ImportError:
        return argparse.HelpFormatter
    return RichHelpFormatter


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="vidtriage",
        description="Triage videos, annotate frames, and run models on regions you point at.",
        formatter_class=_help_formatter(),
    )
    parser.add_argument(
        "paths", nargs="*", type=Path,
        help="Files or folders to open on launch",
    )
    parser.add_argument(
        "-i", "--input", dest="input_dir", type=Path, default=None,
        help="Input directory for a triage session",
    )
    parser.add_argument(
        "-o", "--output", dest="output_dir", type=Path, default=None,
        help="Output directory for a triage session",
    )
    parser.add_argument(
        "--no-plugins", action="store_true",
        help="Start with only the shell — useful for isolating a misbehaving plugin",
    )
    parser.add_argument(
        "--safe-mode", action="store_true",
        help="Skip user drop-in plugins from ~/.vidtriage/plugins/",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    parser.add_argument("--version", action="version", version=f"VidTriage {__version__}")
    return parser.parse_args(argv)


def _collect_paths(args: argparse.Namespace) -> list[Path]:
    """Files named on the command line, expanding any directories."""
    found: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            found.extend(discover_media(path))
        elif path.is_file():
            found.append(path)
        else:
            _log.warning("Ignoring %s: not a file or directory", path)
    if not found and args.input_dir and args.input_dir.is_dir():
        found.extend(discover_media(args.input_dir))
    return found


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(level=logging.INFO, verbose=args.verbose)
    _log.info("VidTriage %s starting", __version__)

    app = QApplication(sys.argv)
    app.setApplicationName("VidTriage")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("VidTriage")

    settings = Settings()
    manager = PluginManager(state_file=CONFIG_DIR / "plugins.json")
    context = AppContext(settings=settings, plugin_manager=manager)

    window = MainWindow(context)

    if args.no_plugins:
        _log.warning("Started with --no-plugins: no features are loaded")
        context.status("Started with --no-plugins", 8000)
    else:
        manager.discover(user_dir=None if args.safe_mode else user_plugin_dir())
        manager.activate_all(context)
        report_startup(context)

    # Plugins activated before the window's registry hook was live; pick up
    # any panels they contributed.
    window.sync_panels()

    paths = _collect_paths(args)
    if paths:
        context.library.set_items(paths, keep_current=False)

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

