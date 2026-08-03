"""Logging setup.

One named logger tree (``labelkit.*``) with a console handler always attached
and an optional rotating file handler bound to the active session's output
directory.

The previous implementation crashed the app if the output directory happened to
be read-only.  Logging is diagnostics, never a hard dependency, so a handler
that cannot be created is reported once on the console and skipped.

The console handler is a ``rich`` one when ``rich`` is installed.  The colour is
incidental; what earns it a place is ``rich_tracebacks``.  Three subsystems
deliberately swallow exceptions so one fault cannot take the app down — plugin
activation, frame decoding and layer painting, each marked ``# noqa: BLE001`` —
and a contained fault is only useful if the report that replaces the crash is
readable.  The file handler stays plain: a log meant for grep should not carry
box-drawing characters.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from .console import console

__all__ = ["ROOT_LOGGER_NAME", "attach_file_log", "configure_logging", "get_logger"]

ROOT_LOGGER_NAME = "labelkit"

_FILE_HANDLER_TAG = "labelkit.session_file"
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3
#: ``label_kit``. The import package and the logger root differ by an
#: underscore, so ``__name__`` has to be translated rather than passed through.
_PACKAGE = __name__.split(".", 1)[0]


def get_logger(name: str) -> logging.Logger:
    """Logger for a submodule. Pass ``__name__``.

    ``label_kit.media.decoder`` becomes ``labelkit.media.decoder``. Stripping
    the package prefix matters: without it the name would be prefixed rather
    than translated, and every record would read
    ``labelkit.label_kit.media.decoder``.
    """
    if name == _PACKAGE:
        return logging.getLogger(ROOT_LOGGER_NAME)
    if name.startswith(f"{_PACKAGE}."):
        name = name[len(_PACKAGE) + 1:]
    if name == ROOT_LOGGER_NAME or name.startswith(f"{ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def configure_logging(level: int = logging.INFO, verbose: bool = False) -> logging.Logger:
    """Install the console handler. Idempotent."""
    effective_level = logging.DEBUG if verbose else level
    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(effective_level)
    root.propagate = False

    if not any(getattr(h, "_labelkit_console", False) for h in root.handlers):
        handler = _console_handler(verbose=verbose)
        handler.setLevel(effective_level)
        handler._labelkit_console = True  # type: ignore[attr-defined]
        root.addHandler(handler)

    return root


def _console_handler(*, verbose: bool) -> logging.Handler:
    """A ``rich`` handler if we can build one, a plain stderr handler otherwise."""
    terminal = console()
    if terminal is None:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)-7s %(name)s: %(message)s"))
        return handler

    from rich.logging import RichHandler

    return RichHandler(
        console=terminal,
        # Messages interpolate paths, plugin ids and exception text, none of it
        # ours to trust: with markup on, a filename containing square brackets
        # is silently eaten or raises inside the handler.
        markup=False,
        rich_tracebacks=True,
        # Locals are worth the width when someone is already debugging, but the
        # values here are frames and QImages — rendering those on every
        # contained fault would bury the message that matters.
        tracebacks_show_locals=verbose,
        log_time_format="%H:%M:%S",
        # The source link replaces the logger name the plain formatter prints:
        # it says the same thing more precisely, and editors make it clickable.
        show_path=True,
    )


def attach_file_log(directory: Path) -> Path | None:
    """Route the log to ``directory/labelkit_activity.log``.

    Replaces any previously attached session file handler, so switching sessions
    does not leave the old file open.  Returns the log path, or ``None`` if the
    handler could not be created.
    """
    root = logging.getLogger(ROOT_LOGGER_NAME)

    for handler in [h for h in root.handlers if getattr(h, "_labelkit_tag", None) == _FILE_HANDLER_TAG]:
        root.removeHandler(handler)
        handler.close()

    log_path = directory / "labelkit_activity.log"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8",
        )
    except OSError as exc:
        root.warning("File logging disabled — cannot write to %s (%s)", directory, exc)
        return None

    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"),
    )
    handler._labelkit_tag = _FILE_HANDLER_TAG  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.info("Logging to %s", log_path)
    return log_path
