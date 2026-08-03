"""Shared fixtures.

Qt runs offscreen and ``HOME`` is redirected, so tests never touch the
developer's real ``~/.labelkit`` settings, sessions, plugin state or triage
logs.

**The redirect happens at import time, not in a fixture, and it has to.**
``label_kit.persistence.settings`` computes ``CONFIG_DIR`` from ``Path.home()``
once, at module scope. Test modules import it while pytest is *collecting*,
which is before any fixture body runs — so a session-scoped fixture that sets
``HOME`` sets it too late, and every test then reads and writes the real config
directory. conftest is imported ahead of the test modules, which makes this the
only place early enough. ``test_isolation.py`` asserts it worked.

Because the directory is created at import time, no fixture teardown owns it and
cleaning it up is this module's job too — on the way out for a run that ends,
and on the way in for one that did not.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_HOME_PREFIX = "labelkit-test-home-"
_OWNER_FILE = "owner.pid"
#: How long an unstamped home is presumed to belong to a run still starting up.
_STARTUP_GRACE_SECONDS = 60


def _sweep_abandoned_homes(root: Path) -> None:
    """Delete the test homes of runs that are no longer alive.

    ``atexit`` covers a clean exit and a Ctrl-C, but not SIGKILL, the OOM killer
    or an IDE's stop button. A home leaked that way is never collected by
    anything afterwards, because nothing else knows the directory exists —
    eighteen had accumulated before anyone looked. Each run stamps its pid, so a
    later run can tell an abandoned home from one currently in use.

    ``os.kill(pid, 0)`` is a POSIX liveness idiom; on Windows that call
    *terminates* the process, so the sweep does not run there.
    """
    if os.name != "posix":
        return
    for home in root.glob(f"{_HOME_PREFIX}*"):
        if home.is_dir() and not _owner_is_alive(home):
            shutil.rmtree(home, ignore_errors=True)


def _owner_is_alive(home: Path) -> bool:
    try:
        pid = int((home / _OWNER_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # The stamp lands just after mkdtemp, so an unstamped home is either a
        # run still starting up or a leak from before stamping existed. Age
        # tells them apart. Keeping one too long costs an empty directory;
        # getting it wrong the other way deletes a live run's HOME mid-test.
        return _age_seconds(home) < _STARTUP_GRACE_SECONDS
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # The pid exists; it just belongs to another user.
    return True


def _age_seconds(home: Path) -> float:
    try:
        return time.time() - home.stat().st_mtime
    except OSError:
        return 0.0  # Vanished under us — treat as busy and leave it alone.


ISOLATED_HOME = Path(tempfile.mkdtemp(prefix=_HOME_PREFIX))
(ISOLATED_HOME / _OWNER_FILE).write_text(str(os.getpid()), encoding="utf-8")
atexit.register(shutil.rmtree, ISOLATED_HOME, ignore_errors=True)
_sweep_abandoned_homes(Path(tempfile.gettempdir()))
os.environ["HOME"] = str(ISOLATED_HOME)
os.environ["USERPROFILE"] = str(ISOLATED_HOME)

import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(scope="session")
def isolated_home() -> Path:
    return ISOLATED_HOME


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def pump(qapp):
    """Run the Qt event loop for ``ms`` milliseconds."""
    from PySide6.QtCore import QEventLoop, QTimer

    def _pump(ms: int = 250) -> None:
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    return _pump


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> Path:
    """A 40-frame 160x120 clip whose red channel encodes the frame index."""
    import cv2

    path = tmp_path_factory.mktemp("media") / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (160, 120),
    )
    assert writer.isOpened(), "OpenCV cannot write mp4 in this environment"
    for index in range(40):
        frame = np.zeros((120, 160, 3), np.uint8)
        frame[:, :, 2] = index * 6  # BGR red ramp
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def fresh_video(sample_video, tmp_path) -> Path:
    """A per-test copy of the sample clip.

    Use this whenever a test may write an annotation sidecar. ``sample_video``
    is session-scoped, so a sidecar written next to it would be loaded by every
    later test that opens it.
    """
    import shutil

    copy = tmp_path / "clip.mp4"
    shutil.copy(sample_video, copy)
    return copy


@pytest.fixture
def rgb_frame():
    from label_kit.core.frames import Frame, FrameRef

    def _make(index: int = 0, width: int = 64, height: int = 48, source: str = "test.mp4"):
        image = np.zeros((height, width, 3), np.uint8)
        image[:, :, 0] = index
        return Frame(ref=FrameRef(source, index), image=image)

    return _make


@pytest.fixture
def without_rich(monkeypatch):
    """Make ``import rich.console`` fail, as it would on a minimal install.

    ``None`` in ``sys.modules`` is the documented way to force an ``ImportError``
    for a module that is in fact installed. Self-contained — it clears the cached
    probe on the way in and out, so it works in any test file.
    """
    import sys

    from label_kit.core.console import console

    monkeypatch.setitem(sys.modules, "rich.console", None)
    console.cache_clear()
    yield
    console.cache_clear()
