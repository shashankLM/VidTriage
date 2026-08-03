"""The suite must not write into the developer's real config directory.

This is not hypothetical. ``CONFIG_DIR`` is computed once from ``Path.home()``
at import time, and the session-scoped fixture that used to redirect ``HOME``
ran too late — test modules had already imported the constant during collection.
The suite spent its life writing real sessions and triage logs into
``~/.vidtriage``, and because the session list is most-recently-used first, the
app then restored a pytest temp directory on the next real launch.

If this file fails, every other test is silently touching your home directory.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from conftest import _HOME_PREFIX, _OWNER_FILE, ISOLATED_HOME, _sweep_abandoned_homes

from label_kit.persistence.settings import CONFIG_DIR, default_settings_path, user_plugin_dir
from label_kit.plugins.builtin.triage.config import SESSIONS_FILE
from label_kit.plugins.builtin.triage.ledger import default_log_dir


def test_config_dir_is_redirected():
    assert CONFIG_DIR.is_relative_to(ISOLATED_HOME), (
        f"CONFIG_DIR is {CONFIG_DIR}, which is outside the test home — "
        f"the suite is reading and writing the real ~/.labelkit."
    )


def test_every_writable_location_is_inside_the_test_home(tmp_path):
    """One escapee is enough to corrupt a real session list."""
    for path in (
        default_settings_path(),
        user_plugin_dir(),
        SESSIONS_FILE,
        default_log_dir(tmp_path),
    ):
        assert path.is_relative_to(ISOLATED_HOME), f"{path} escapes the test home"


@pytest.mark.skipif(os.name != "posix", reason="the sweep is a POSIX pid check")
class TestAbandonedHomeSweep:
    """A run killed outright must not leak its temp HOME forever.

    ``atexit`` cleans up a run that ends, including under Ctrl-C. Nothing
    cleans up after SIGKILL, the OOM killer or an IDE's stop button, and the
    leaked directory is invisible to everything afterwards — eighteen of them
    had piled up in ``/tmp``. The next run collects them, which means it has to
    tell a dead run's home from a live one's without ever getting it backwards.
    """

    def test_a_dead_runs_home_is_removed(self, tmp_path):
        home = _fake_home(tmp_path, "dead", pid=_a_reaped_pid())
        _sweep_abandoned_homes(tmp_path)
        assert not home.exists()

    def test_a_live_runs_home_is_left_alone(self, tmp_path):
        """Two suites running at once must not delete each other's HOME."""
        home = _fake_home(tmp_path, "live", pid=os.getpid())
        _sweep_abandoned_homes(tmp_path)
        assert home.exists()

    def test_an_unstamped_home_is_spared_while_it_is_young(self, tmp_path):
        """The stamp lands just after mkdtemp, and that gap must not be fatal."""
        home = _fake_home(tmp_path, "starting-up", pid=None)
        _sweep_abandoned_homes(tmp_path)
        assert home.exists()

    def test_an_unstamped_home_is_removed_once_it_is_old(self, tmp_path):
        """Leaks predating the stamp are still leaks."""
        home = _fake_home(tmp_path, "ancient", pid=None)
        stale = time.time() - 3600
        os.utime(home, (stale, stale))
        _sweep_abandoned_homes(tmp_path)
        assert not home.exists()

    def test_nothing_else_in_the_temp_dir_is_touched(self, tmp_path):
        unrelated = tmp_path / "someone-elses-work"
        unrelated.mkdir()
        _sweep_abandoned_homes(tmp_path)
        assert unrelated.exists()

    def test_the_sweep_spares_the_home_this_suite_is_using(self):
        """The real call, against the real temp dir, as conftest makes it."""
        _sweep_abandoned_homes(Path(tempfile.gettempdir()))
        assert ISOLATED_HOME.exists()
        assert CONFIG_DIR.parent == ISOLATED_HOME


def _fake_home(root: Path, name: str, *, pid: int | None) -> Path:
    home = root / f"{_HOME_PREFIX}{name}"
    home.mkdir()
    if pid is not None:
        (home / _OWNER_FILE).write_text(str(pid), encoding="utf-8")
    return home


def _a_reaped_pid() -> int:
    """A pid that has certainly exited: started, waited on, and collected."""
    process = subprocess.Popen([sys.executable, "-c", ""])
    process.wait()
    return process.pid


class TestLegacyMigration:
    """The project was called VidTriage. Users have its data on disk.

    Two things must survive the rename: the config directory (which holds the
    triage decision logs — the *only* record of what was classified) and the
    annotation sidecars sitting next to real footage.
    """

    def test_config_dir_moves_once(self, tmp_path, monkeypatch):
        from label_kit.persistence import settings as settings_module

        legacy = tmp_path / ".vidtriage"
        (legacy / "logs" / "corpus-abc").mkdir(parents=True)
        (legacy / "logs" / "corpus-abc" / "a.triage.jsonl").write_text('{"file":"x","class":"y"}\n')
        (legacy / "sessions.json").write_text("{}")

        current = tmp_path / ".labelkit"
        monkeypatch.setattr(settings_module, "LEGACY_CONFIG_DIR", legacy)
        monkeypatch.setattr(settings_module, "CONFIG_DIR", current)

        assert settings_module.migrate_legacy_config() == current
        assert (current / "logs" / "corpus-abc" / "a.triage.jsonl").exists()
        assert (current / "sessions.json").exists()
        assert not legacy.exists(), "moved, not copied — weights can be hundreds of MB"

    def test_migration_is_skipped_once_the_new_dir_exists(self, tmp_path, monkeypatch):
        """A second run must never merge two divergent trees."""
        from label_kit.persistence import settings as settings_module

        legacy = tmp_path / ".vidtriage"
        legacy.mkdir()
        (legacy / "stale.json").write_text("{}")
        current = tmp_path / ".labelkit"
        current.mkdir()

        monkeypatch.setattr(settings_module, "LEGACY_CONFIG_DIR", legacy)
        monkeypatch.setattr(settings_module, "CONFIG_DIR", current)

        assert settings_module.migrate_legacy_config() is None
        assert not (current / "stale.json").exists()
        assert legacy.exists(), "the old directory is left alone, not silently dropped"

    def test_no_legacy_dir_is_not_an_error(self, tmp_path, monkeypatch):
        from label_kit.persistence import settings as settings_module

        monkeypatch.setattr(settings_module, "LEGACY_CONFIG_DIR", tmp_path / "absent")
        monkeypatch.setattr(settings_module, "CONFIG_DIR", tmp_path / ".labelkit")
        assert settings_module.migrate_legacy_config() is None

    def test_a_legacy_sidecar_is_still_read(self, tmp_path):
        from label_kit.persistence.sidecar import load_annotations

        media = tmp_path / "clip.mp4"
        media.write_bytes(b"x")
        _write_legacy_sidecar(media, "old")

        loaded = load_annotations(media)
        assert [a.label for a in loaded] == ["old"]

    def test_the_current_name_wins_over_the_legacy_one(self, tmp_path):
        from label_kit.core.annotations import Annotation
        from label_kit.core.frames import FrameRef, source_id_for
        from label_kit.core.geometry import Rect
        from label_kit.persistence.sidecar import load_annotations, save_annotations

        media = tmp_path / "clip.mp4"
        media.write_bytes(b"x")
        _write_legacy_sidecar(media, "old")
        save_annotations(media, [
            Annotation(FrameRef(source_id_for(media), 0), Rect(5, 5, 9, 9), label="new"),
        ])

        assert [a.label for a in load_annotations(media)] == ["new"]

    def test_clearing_annotations_does_not_resurrect_the_legacy_file(self, tmp_path):
        """Deleting only the new sidecar would let the fallback bring them back."""
        from label_kit.persistence.sidecar import load_annotations, save_annotations

        media = tmp_path / "clip.mp4"
        media.write_bytes(b"x")
        legacy = _write_legacy_sidecar(media, "old")

        save_annotations(media, [])

        assert load_annotations(media) == []
        assert not legacy.exists()

    def test_a_legacy_triage_log_still_replays(self, tmp_path):
        """Those logs are the only record of what a previous run classified."""
        from label_kit.plugins.builtin.triage.ledger import replay

        log = tmp_path / "old.triage.jsonl"
        log.write_text(
            '{"v":1,"kind":"vidtriage-triage-log","input_dir":"/x"}\n'
            '{"v":1,"at":"2026-01-01T00:00:00Z","file":"a.mp4","class":"keep"}\n',
        )
        assert replay([log]) == {"a.mp4": "keep"}


def _write_legacy_sidecar(media: Path, label: str) -> Path:
    """A sidecar under the old name, written through the real serialiser.

    Hand-rolled JSON would only prove the fallback reads *something*; going
    through ``save_annotations`` and renaming proves it reads what VidTriage
    actually wrote.
    """
    from label_kit.core.annotations import Annotation
    from label_kit.core.frames import FrameRef, source_id_for
    from label_kit.core.geometry import Rect
    from label_kit.persistence.sidecar import (
        LEGACY_SIDECAR_SUFFIX,
        save_annotations,
        sidecar_path_for,
    )

    save_annotations(media, [
        Annotation(FrameRef(source_id_for(media), 0), Rect(1, 2, 3, 4), label=label),
    ])
    legacy = media.with_name(media.name + LEGACY_SIDECAR_SUFFIX)
    sidecar_path_for(media).rename(legacy)
    return legacy
