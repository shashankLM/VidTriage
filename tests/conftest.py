"""Shared fixtures.

Qt runs offscreen and ``HOME`` is redirected, so tests never touch the
developer's real ``~/.vidtriage`` settings, sessions, plugin state or triage
logs.

**The redirect happens at import time, not in a fixture, and it has to.**
``vidtriage.persistence.settings`` computes ``CONFIG_DIR`` from ``Path.home()``
once, at module scope. Test modules import it while pytest is *collecting*,
which is before any fixture body runs — so a session-scoped fixture that sets
``HOME`` sets it too late, and every test then reads and writes the real config
directory. conftest is imported ahead of the test modules, which makes this the
only place early enough. ``test_isolation.py`` asserts it worked.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ISOLATED_HOME = Path(tempfile.mkdtemp(prefix="vidtriage-test-home-"))
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
    from vidtriage.core.frames import Frame, FrameRef

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

    from vidtriage.core.console import console

    monkeypatch.setitem(sys.modules, "rich.console", None)
    console.cache_clear()
    yield
    console.cache_clear()
