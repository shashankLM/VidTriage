"""The decode worker — everything that touches OpenCV runs here, off the GUI thread.

Three properties make this robust where the previous ``QTimer``-on-the-GUI-thread
player was not:

**Nothing blocks the UI.** Decoding, seeking and colour conversion all happen on
a dedicated thread. A slow file or a 4K frame can no longer freeze the window.

**Requests coalesce.** Dragging the seek bar produces a burst of requests; only
the most recent one is serviced. The old player decoded every single one.

**Backpressure is explicit.** The worker refuses to run ahead of the renderer by
more than :data:`_MAX_IN_FLIGHT` frames. Without this, a GUI thread busy running
inference would accumulate an unbounded queue of stale frames and the video
would appear to keep playing for seconds after being paused.

The in-flight counter needs no lock: both the increment (in ``_publish``) and the
decrement (in ``notify_consumed``) execute on the worker thread, because Qt
dispatches queued slot invocations on the receiving object's own thread.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot

from ..core.errors import MediaError
from ..core.logging import get_logger
from .clock import EndMode, PlaybackClock
from .source import MediaSource, open_source

__all__ = ["DecodeWorker"]

_log = get_logger(__name__)

_MAX_IN_FLIGHT = 2


class DecodeWorker(QObject):
    """Owns a :class:`MediaSource` and paces frames out of it.

    Every public method is a slot, meant to be invoked across the thread
    boundary by :class:`~label_kit.media.controller.PlaybackController`. Do not
    call them directly from the GUI thread.
    """

    opened = Signal(object)          # MediaInfo
    open_failed = Signal(str, str)   # path, message
    frame_ready = Signal(object)     # Frame
    reached_end = Signal(str)        # source_id
    error = Signal(str)
    closed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._source: MediaSource | None = None
        self._clock = PlaybackClock()
        self._playing = False
        self._in_flight = 0
        self._pending_seek: int | None = None
        self._service_scheduled = False
        self._play_timer: QTimer | None = None

    # ── lifecycle ───────────────────────────────────────────────────────

    @Slot()
    def initialize(self) -> None:
        """Create thread-affine objects. Connected to ``QThread.started``.

        A ``QTimer`` belongs to the thread that constructed it, so it cannot be
        created in ``__init__`` — that runs on the GUI thread, before
        ``moveToThread``.
        """
        self._play_timer = QTimer(self)
        self._play_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._play_timer.timeout.connect(self._on_play_tick)

    @Slot(str)
    def open(self, path: str) -> None:
        self._teardown_source()
        try:
            source = open_source(Path(path))
        except MediaError as exc:
            _log.warning("Open failed: %s", exc)
            self.open_failed.emit(path, str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - a codec fault must not kill the thread
            _log.exception("Unexpected error opening %s", path)
            self.open_failed.emit(path, f"{type(exc).__name__}: {exc}")
            return

        self._source = source
        self._clock.fps = source.info.fps
        self._clock.clear_loop_range()
        self.opened.emit(source.info)
        self._request_service(seek_to=0)

    @Slot()
    def close(self) -> None:
        self._teardown_source()
        self.closed.emit()

    def _teardown_source(self) -> None:
        self._stop_timer()
        self._playing = False
        self._pending_seek = None
        self._in_flight = 0
        if self._source is not None:
            self._source.close()
            self._source = None

    # ── transport ───────────────────────────────────────────────────────

    @Slot(bool)
    def set_playing(self, playing: bool) -> None:
        if self._source is None or self._clock.is_static:
            self._playing = False
            self._stop_timer()
            return
        self._playing = bool(playing)
        if self._playing:
            self._start_timer()
        else:
            self._stop_timer()

    @Slot(int)
    def seek(self, index: int) -> None:
        """Queue a seek. Repeated calls collapse into a single decode."""
        self._request_service(seek_to=index)

    @Slot(int)
    def step(self, direction: int) -> None:
        if self._source is None:
            return
        self._playing = False
        self._stop_timer()
        target = self._clock.step_index(
            self._current_index(), self._playable_count(), direction,
        )
        self._request_service(seek_to=target)

    @Slot()
    def refresh(self) -> None:
        """Re-decode and re-emit the current frame."""
        self._request_service(seek_to=self._current_index())

    # ── clock configuration ─────────────────────────────────────────────

    @Slot(float)
    def set_speed(self, speed: float) -> None:
        self._clock.set_speed(speed)
        if self._playing:
            self._start_timer()

    @Slot(int)
    def set_step_size(self, size: int) -> None:
        self._clock.step_size = max(1, int(size))

    @Slot(str)
    def set_end_mode(self, mode: str) -> None:
        try:
            self._clock.end_mode = EndMode(mode)
        except ValueError:
            _log.warning("Unknown end mode %r, ignoring", mode)

    @Slot(object, object)
    def set_loop_range(self, start: object, end: object) -> None:
        self._clock.set_loop_range(
            None if start is None else int(start),
            None if end is None else int(end),
        )

    # ── renderer acknowledgement ────────────────────────────────────────

    @Slot()
    def notify_consumed(self) -> None:
        """Called by the controller once a frame has been handed to the view."""
        if self._in_flight > 0:
            self._in_flight -= 1

    # ── internals ───────────────────────────────────────────────────────

    def _playable_count(self) -> int:
        return self._source.info.frame_count if self._source else 0

    @property
    def _length_unknown(self) -> bool:
        """True when the container did not report a usable frame count.

        Such a source is played sequentially until ``read()`` runs dry; asking
        the clock for bounds would clamp everything to frame 0.
        """
        return self._playable_count() <= 0

    def _current_index(self) -> int:
        """Index of the frame currently on screen (position points at the next)."""
        if self._source is None:
            return 0
        return max(0, self._source.position - 1)

    def _start_timer(self) -> None:
        if self._play_timer is not None:
            self._play_timer.start(self._clock.interval_ms)

    def _stop_timer(self) -> None:
        if self._play_timer is not None:
            self._play_timer.stop()

    def _request_service(self, seek_to: int | None = None) -> None:
        """Coalesce a request; at most one ``_service`` runs per event-loop turn."""
        if seek_to is not None:
            self._pending_seek = seek_to
        if self._service_scheduled:
            return
        self._service_scheduled = True
        QTimer.singleShot(0, self._service)

    @Slot()
    def _service(self) -> None:
        """Handle a user-initiated seek. Always produces a frame if one exists."""
        self._service_scheduled = False
        source = self._source
        if source is None:
            return

        target = self._pending_seek
        self._pending_seek = None
        if target is not None:
            source.seek(target)

        frame = self._decode()
        if frame is not None:
            self._publish(frame)
        elif target is not None and target > 0:
            # Seeking past a container-over-reported end: fall back to frame 0
            # rather than leaving the view blank.
            source.seek(0)
            recovered = self._decode()
            if recovered is not None:
                self._publish(recovered)

    @Slot()
    def _on_play_tick(self) -> None:
        source = self._source
        if not self._playing or source is None:
            return
        if self._pending_seek is not None:
            return  # a seek is queued; let _service win
        if self._in_flight >= _MAX_IN_FLIGHT:
            return  # renderer is behind — drop this tick rather than queue up

        if self._length_unknown:
            frame = self._decode()
            if frame is not None:
                self._publish(frame)
            else:
                self._finish_playback()
            return

        nxt = self._clock.next_index(self._current_index(), self._playable_count())
        if nxt is None:
            self._finish_playback()
            return
        if nxt != source.position:
            source.seek(nxt)

        frame = self._decode()
        if frame is not None:
            self._publish(frame)
        else:
            self._finish_playback()

    def _decode(self):
        source = self._source
        if source is None:
            return None
        try:
            return source.read()
        except Exception as exc:  # noqa: BLE001 - a bad packet must not kill the thread
            _log.exception("Decode failed for %s", source.path.name)
            self.error.emit(f"Decode error in {source.path.name}: {exc}")
            self._playing = False
            self._stop_timer()
            return None

    def _publish(self, frame) -> None:
        self._in_flight += 1
        self.frame_ready.emit(frame)

    def _finish_playback(self) -> None:
        """End of stream reached during playback."""
        source = self._source
        if source is None:
            return

        if self._clock.end_mode is EndMode.LOOP:
            first, _ = self._clock.effective_range(self._playable_count())
            source.seek(first)
            frame = self._decode()
            if frame is not None:
                self._publish(frame)
                return

        self._playing = False
        self._stop_timer()
        self.reached_end.emit(source.source_id)
