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
        "--log", dest="logs", action="append", type=Path, metavar="PATH",
        help="Triage log to replay; repeat to stack them. Applied in the order "
             "given, last one wins. Omit to replay every log recorded for the "
             "input directory",
    )
    parser.add_argument(
        "--snapshot", dest="snapshot_dir", type=Path, default=None, metavar="DIR",
        help="Copy every classified video into DIR/<class>/ and exit. DIR must "
             "be new or empty",
    )
    parser.add_argument(
        "--link", action="store_true",
        help="Hardlink instead of copying during --snapshot, falling back to a "
             "copy per file when that is not possible",
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

    if args.snapshot_dir is not None:
        # Headless: build the deliverable from the logs and exit, so a rerun can
        # be scripted without a display.
        return snapshot_only(args)

    app = QApplication(sys.argv)
    app.setApplicationName("VidTriage")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("VidTriage")

    settings = Settings()
    manager = PluginManager(state_file=CONFIG_DIR / "plugins.json")
    context = AppContext(
        settings=settings,
        plugin_manager=manager,
        launch_options={"triage_logs": args.logs},
    )

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


def snapshot_only(args: argparse.Namespace) -> int:
    """Build a snapshot from the logs and exit. No Qt, no window.

    Deliberately does not append to a log or create one: a snapshot is a read of
    the decisions, so running it must not become a pass of its own.
    """
    from .core.errors import VidTriageError
    from .plugins.builtin.triage.config import load_last_session
    from .plugins.builtin.triage.session import Session
    from .plugins.builtin.triage.snapshot import write_snapshot

    config = load_last_session()
    input_dir = args.input_dir or config.input_dir
    if input_dir is None or not Path(input_dir).is_dir():
        _log.error("No input directory — pass -i, or run the app once to set one up")
        return 2

    session = Session(
        input_dir, args.output_dir or config.output_dir or input_dir,
        config.classes, logs=args.logs, record=False,
    )
    session.load()

    try:
        result = write_snapshot(session.classified, args.snapshot_dir, link=args.link)
    except VidTriageError as exc:
        _log.error("%s", exc)
        return 1

    for warning in result.warnings:
        _log.warning("%s", warning)
    _log.info("%s", result.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())

