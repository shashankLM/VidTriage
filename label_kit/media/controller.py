"""GUI-side facade over the decode thread.

The rest of the application talks only to this object. It owns the worker
thread, mirrors just enough state on the GUI side that the UI can be queried
synchronously (``current_index``, ``is_playing``) without a cross-thread read,
and drops frames belonging to a source that has since been replaced.

That last part matters: ``open()`` is asynchronous, so frames decoded from the
previous file can still be in the event queue when the new one is requested.
Rendering them would flash the old video into the new file's view.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEventLoop, QObject, QThread, QTimer, Signal

from ..core.frames import Frame, MediaInfo, source_id_for
from ..core.logging import get_logger
from .clock import MAX_SPEED, MIN_SPEED, EndMode
from .decoder import DecodeWorker

__all__ = ["PlaybackController"]

_log = get_logger(__name__)

_SHUTDOWN_TIMEOUT_MS = 3000


class PlaybackController(QObject):
    """Transport controls plus the current frame, for one media file at a time."""

    opened = Signal(object)          # MediaInfo
    open_failed = Signal(str, str)   # path, message
    frame_changed = Signal(object)   # Frame
    reached_end = Signal(str)        # source_id
    playing_changed = Signal(bool)
    error = Signal(str)

    # Internal — connected to worker slots, so emission crosses the thread boundary.
    _do_open = Signal(str)
    _do_close = Signal()
    _do_set_playing = Signal(bool)
    _do_seek = Signal(int)
    _do_step = Signal(int)
    _do_refresh = Signal()
    _do_set_speed = Signal(float)
    _do_set_step_size = Signal(int)
    _do_set_end_mode = Signal(str)
    _do_set_loop_range = Signal(object, object)
    _do_notify_consumed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

        self._info: MediaInfo | None = None
        self._frame: Frame | None = None
        self._expected_source_id: str | None = None
        self._playing = False
        self._speed = 1.0
        self._step_size = 1
        self._end_mode = EndMode.NEXT
        self._shut_down = False

        self._thread = QThread()
        self._thread.setObjectName("label_kit-decode")
        self._worker = DecodeWorker()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.initialize)
        self._thread.finished.connect(self._worker.deleteLater)

        self._do_open.connect(self._worker.open)
        self._do_close.connect(self._worker.close)
        self._do_set_playing.connect(self._worker.set_playing)
        self._do_seek.connect(self._worker.seek)
        self._do_step.connect(self._worker.step)
        self._do_refresh.connect(self._worker.refresh)
        self._do_set_speed.connect(self._worker.set_speed)
        self._do_set_step_size.connect(self._worker.set_step_size)
        self._do_set_end_mode.connect(self._worker.set_end_mode)
        self._do_set_loop_range.connect(self._worker.set_loop_range)
        self._do_notify_consumed.connect(self._worker.notify_consumed)

        self._worker.opened.connect(self._on_opened)
        self._worker.open_failed.connect(self._on_open_failed)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.reached_end.connect(self._on_reached_end)
        self._worker.error.connect(self.error)

        self._thread.start()

    # ── state ───────────────────────────────────────────────────────────

    @property
    def info(self) -> MediaInfo | None:
        return self._info

    @property
    def current_frame(self) -> Frame | None:
        """The frame on screen. This is what inference runs on."""
        return self._frame

    @property
    def current_index(self) -> int:
        return self._frame.index if self._frame else 0

    @property
    def frame_count(self) -> int:
        return self._info.frame_count if self._info else 0

    @property
    def duration(self) -> float:
        return self._info.duration if self._info else 0.0

    @property
    def position(self) -> float:
        return self._info.time_of(self.current_index) if self._info else 0.0

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def is_loaded(self) -> bool:
        return self._info is not None

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def step_size(self) -> int:
        return self._step_size

    @property
    def end_mode(self) -> EndMode:
        return self._end_mode

    # ── commands ────────────────────────────────────────────────────────

    def open(self, path: Path | str) -> None:
        resolved = Path(path)
        self._expected_source_id = source_id_for(resolved)
        self._info = None
        self._frame = None
        self._set_playing_state(False)
        self._do_open.emit(str(resolved))

    def close(self) -> None:
        self._expected_source_id = None
        self._info = None
        self._frame = None
        self._set_playing_state(False)
        self._do_close.emit()

    def close_and_wait(self, timeout_ms: int = 2000) -> bool:
        """Close the source and block until the decoder has released the file.

        ``close()`` is asynchronous, so the ``VideoCapture`` may still hold an
        open handle when it returns. Anything that then moves or deletes that
        file — triage classification, for one — fails outright on Windows and
        races on network filesystems. Callers that are about to touch the file
        on disk must use this instead.

        Returns ``True`` if the decoder confirmed the close within the timeout.
        """
        if self._shut_down or not self._thread.isRunning():
            return True

        loop = QEventLoop()
        confirmed = False

        def on_closed() -> None:
            nonlocal confirmed
            confirmed = True
            loop.quit()

        connection = self._worker.closed.connect(on_closed)
        guard = QTimer()
        guard.setSingleShot(True)
        guard.timeout.connect(loop.quit)
        guard.start(timeout_ms)
        try:
            self.close()
            loop.exec()
        finally:
            guard.stop()
            self._worker.closed.disconnect(connection)

        if not confirmed:
            _log.warning("Decoder did not confirm close within %dms", timeout_ms)
        return confirmed

    def play(self) -> None:
        if self._info is None:
            return
        self._set_playing_state(True)
        self._do_set_playing.emit(True)

    def pause(self) -> None:
        self._set_playing_state(False)
        self._do_set_playing.emit(False)

    def toggle_play(self) -> None:
        self.pause() if self._playing else self.play()

    def seek_index(self, index: int) -> None:
        if self._info is None:
            return
        self._do_seek.emit(int(index))

    def seek_time(self, seconds: float) -> None:
        if self._info is None:
            return
        self._do_seek.emit(self._info.index_of(seconds))

    def seek_fraction(self, fraction: float) -> None:
        """Seek to ``0.0``-``1.0`` through the video — what a scrub bar wants."""
        if self._info is None or self._info.frame_count <= 0:
            return
        clamped = min(max(fraction, 0.0), 1.0)
        self._do_seek.emit(round(clamped * (self._info.frame_count - 1)))

    def step_forward(self) -> None:
        self._set_playing_state(False)
        self._do_step.emit(1)

    def step_backward(self) -> None:
        self._set_playing_state(False)
        self._do_step.emit(-1)

    def refresh(self) -> None:
        self._do_refresh.emit()

    def set_speed(self, speed: float) -> None:
        self._speed = min(max(float(speed), MIN_SPEED), MAX_SPEED)
        self._do_set_speed.emit(self._speed)

    def set_step_size(self, size: int) -> None:
        self._step_size = max(1, int(size))
        self._do_set_step_size.emit(self._step_size)

    def set_end_mode(self, mode: EndMode | str) -> None:
        self._end_mode = EndMode(mode)
        self._do_set_end_mode.emit(self._end_mode.value)

    def set_loop_range(self, start: int | None, end: int | None) -> None:
        self._do_set_loop_range.emit(start, end)

    # ── worker callbacks (GUI thread) ───────────────────────────────────

    def _on_opened(self, info: MediaInfo) -> None:
        if info.source_id != self._expected_source_id:
            return
        self._info = info
        # Re-assert clock settings: the worker resets per-source state on open.
        self._do_set_speed.emit(self._speed)
        self._do_set_step_size.emit(self._step_size)
        self._do_set_end_mode.emit(self._end_mode.value)
        self.opened.emit(info)

    def _on_open_failed(self, path: str, message: str) -> None:
        if source_id_for(Path(path)) != self._expected_source_id:
            return
        self._info = None
        self._frame = None
        self._set_playing_state(False)
        self.open_failed.emit(path, message)

    def _on_frame_ready(self, frame: Frame) -> None:
        # Acknowledge unconditionally, including for stale frames — the worker's
        # in-flight counter must be balanced or playback stalls permanently.
        self._do_notify_consumed.emit()
        if frame.ref.source_id != self._expected_source_id:
            return
        self._frame = frame
        self.frame_changed.emit(frame)

    def _on_reached_end(self, source_id: str) -> None:
        if source_id != self._expected_source_id:
            return
        self._set_playing_state(False)
        self.reached_end.emit(source_id)

    def _set_playing_state(self, playing: bool) -> None:
        if self._playing != playing:
            self._playing = playing
            self.playing_changed.emit(playing)

    # ── teardown ────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Stop the thread. Safe to call more than once; must be called once."""
        if self._shut_down:
            return
        self._shut_down = True

        self._worker.frame_ready.disconnect(self._on_frame_ready)
        self._do_close.emit()
        self._thread.quit()
        if not self._thread.wait(_SHUTDOWN_TIMEOUT_MS):
            _log.warning("Decode thread did not stop in %dms; terminating", _SHUTDOWN_TIMEOUT_MS)
            self._thread.terminate()
            self._thread.wait()
