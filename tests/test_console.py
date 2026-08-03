"""Console output: the rich handler, the plain fallback, and the startup report.

Two properties are worth defending here. ``rich`` must stay optional — the app
has to start on an install that lacks it — and the *file* log must stay plain,
because a log that has grown box-drawing characters and ANSI escapes is no
longer greppable.
"""

from __future__ import annotations

import logging

import pytest

from vidtriage.core.console import console, rich_available
from vidtriage.core.logging import ROOT_LOGGER_NAME, attach_file_log, configure_logging


@pytest.fixture
def clean_logger():
    """Give each test the ``vidtriage`` logger tree to itself.

    ``configure_logging`` and ``attach_file_log`` both mutate a process-global
    logger; without this, a handler installed by one test formats another test's
    records — and the file handler keeps a file open.
    """
    root = logging.getLogger(ROOT_LOGGER_NAME)
    saved_handlers, saved_level = root.handlers[:], root.level
    root.handlers = []
    yield root
    for handler in root.handlers:
        handler.close()
    root.handlers, root.level = saved_handlers, saved_level


@pytest.fixture(autouse=True)
def fresh_console_probe():
    """The console is cached process-wide; do not let one test's probe leak."""
    console.cache_clear()
    yield
    console.cache_clear()


class TestConsoleProbe:
    def test_returns_a_console_when_rich_is_installed(self):
        assert console() is not None
        assert rich_available()

    def test_same_instance_every_call(self):
        """One Console, or two cursors fight over one stream."""
        assert console() is console()

    def test_writes_to_stderr(self):
        """Reports and logs share stderr, leaving stdout clean for piping."""
        assert console().stderr

    def test_degrades_to_none_without_rich(self, without_rich):
        assert console() is None
        assert not rich_available()


class TestConsoleHandler:
    def test_installs_a_rich_handler(self, clean_logger):
        from rich.logging import RichHandler

        configure_logging()
        installed = [h for h in clean_logger.handlers if getattr(h, "_vidtriage_console", False)]
        assert len(installed) == 1
        assert isinstance(installed[0], RichHandler)

    def test_falls_back_to_a_plain_handler(self, clean_logger, without_rich):
        configure_logging()
        installed = [h for h in clean_logger.handlers if getattr(h, "_vidtriage_console", False)]
        assert len(installed) == 1
        assert type(installed[0]) is logging.StreamHandler

    def test_fallback_keeps_the_logger_name(self, clean_logger, without_rich, capsys):
        """Without a source-link column, the name is the only subsystem clue."""
        configure_logging()
        logging.getLogger(f"{ROOT_LOGGER_NAME}.media.decoder").warning("no codec")
        assert "vidtriage.media.decoder: no codec" in capsys.readouterr().err

    def test_idempotent(self, clean_logger):
        configure_logging()
        configure_logging()
        installed = [h for h in clean_logger.handlers if getattr(h, "_vidtriage_console", False)]
        assert len(installed) == 1

    def test_markup_is_off(self, clean_logger, capsys):
        """A path with square brackets must survive intact, not be read as a tag."""
        configure_logging()
        logging.getLogger(ROOT_LOGGER_NAME).warning("cannot open /tmp/clip [raw]/a.mp4")
        assert "[raw]" in capsys.readouterr().err

    def test_verbose_enables_locals_in_tracebacks(self, clean_logger):
        configure_logging(verbose=True)
        handler = next(h for h in clean_logger.handlers if getattr(h, "_vidtriage_console", False))
        assert handler.tracebacks_show_locals

    def test_quiet_run_omits_locals(self, clean_logger):
        """Frames and QImages in every contained fault would bury the message."""
        configure_logging()
        handler = next(h for h in clean_logger.handlers if getattr(h, "_vidtriage_console", False))
        assert not handler.tracebacks_show_locals


class TestFileLogStaysPlain:
    def test_no_styling_reaches_the_file(self, clean_logger, tmp_path):
        configure_logging()
        log_path = attach_file_log(tmp_path)
        assert log_path is not None

        logger = logging.getLogger(f"{ROOT_LOGGER_NAME}.media.decoder")
        try:
            raise ValueError("decode failed")
        except ValueError:
            logger.exception("Frame %d unreadable", 12)

        for handler in clean_logger.handlers:
            handler.flush()
        written = log_path.read_text(encoding="utf-8")

        assert "vidtriage.media.decoder: Frame 12 unreadable" in written
        assert "ValueError: decode failed" in written
        assert "\x1b[" not in written, "ANSI escapes in a file meant for grep"
        assert not set(written) & set("╭╮╰╯│─"), "box drawing in a file meant for grep"

    def test_unwritable_directory_does_not_raise(self, clean_logger, tmp_path):
        """Logging is diagnostics; it never gets to be the thing that fails."""
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("", encoding="utf-8")
        configure_logging()
        assert attach_file_log(blocker / "logs") is None
