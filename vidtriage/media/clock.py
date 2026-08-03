"""Playback policy as a pure state machine.

Speed, frame step, end-of-stream behaviour and the loop region are decisions,
not I/O.  Keeping them here — free of Qt, OpenCV and threads — means the awkward
cases (what happens when you step past the last frame with looping on?) are
covered by fast unit tests instead of by clicking around the app.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = ["MAX_SPEED", "MIN_SPEED", "EndMode", "PlaybackClock"]

MIN_SPEED = 0.1
MAX_SPEED = 8.0
_MIN_INTERVAL_MS = 1
_MAX_INTERVAL_MS = 1000
_FALLBACK_FPS = 30.0


class EndMode(StrEnum):
    """What to do when playback reaches the end of the source."""

    NEXT = "next"
    LOOP = "loop"
    STOP = "stop"


@dataclass
class PlaybackClock:
    """Timing and end-of-stream policy for one source.

    Args:
        fps: Source frame rate. Zero or negative means "still image" — the clock
            reports :attr:`is_static` and playback is meaningless.
        speed: Playback rate multiplier, clamped to ``[MIN_SPEED, MAX_SPEED]``.
        step_size: Frames advanced per manual step.
        end_mode: Behaviour at the final frame.
        loop_start / loop_end: Inclusive frame range to loop over when
            ``end_mode`` is :attr:`EndMode.LOOP`. ``None`` loops the whole source.
    """

    fps: float = _FALLBACK_FPS
    speed: float = 1.0
    step_size: int = 1
    end_mode: EndMode = EndMode.NEXT
    loop_start: int | None = None
    loop_end: int | None = None

    def __post_init__(self) -> None:
        self.speed = self.clamp_speed(self.speed)
        self.step_size = max(1, int(self.step_size))
        if not isinstance(self.end_mode, EndMode):
            self.end_mode = EndMode(self.end_mode)

    @staticmethod
    def clamp_speed(speed: float) -> float:
        return min(max(float(speed), MIN_SPEED), MAX_SPEED)

    @property
    def is_static(self) -> bool:
        return self.fps <= 0

    @property
    def interval_ms(self) -> int:
        """Milliseconds between frames at the current speed.

        Clamped at both ends: a 0 ms timer starves the event loop, and an
        interval over a second makes the UI feel hung.
        """
        if self.is_static:
            return _MAX_INTERVAL_MS
        raw = 1000.0 / (self.fps * self.speed)
        return int(min(max(round(raw), _MIN_INTERVAL_MS), _MAX_INTERVAL_MS))

    def set_speed(self, speed: float) -> float:
        self.speed = self.clamp_speed(speed)
        return self.speed

    def set_loop_range(self, start: int | None, end: int | None) -> None:
        if start is not None and end is not None and start > end:
            start, end = end, start
        self.loop_start = start
        self.loop_end = end

    def clear_loop_range(self) -> None:
        self.loop_start = self.loop_end = None

    @property
    def has_loop_range(self) -> bool:
        return self.loop_start is not None or self.loop_end is not None

    def effective_range(self, frame_count: int) -> tuple[int, int]:
        """Inclusive ``(first, last)`` playable index, honouring the loop region."""
        last_available = max(frame_count - 1, 0)
        first = 0 if self.loop_start is None else max(0, min(self.loop_start, last_available))
        last = last_available if self.loop_end is None else max(0, min(self.loop_end, last_available))
        if first > last:
            first, last = last, first
        return first, last

    def next_index(self, current: int, frame_count: int) -> int | None:
        """Index after ``current`` during playback.

        ``None`` means "stop here" — the caller decides whether that ends the
        video, advances to the next file, or just pauses.
        """
        first, last = self.effective_range(frame_count)
        nxt = current + 1
        if nxt <= last:
            return nxt
        if self.end_mode is EndMode.LOOP:
            return first
        return None

    def step_index(self, current: int, frame_count: int, direction: int = 1) -> int:
        """Index after a manual step. Always lands on a valid frame."""
        first, last = self.effective_range(frame_count)
        target = current + self.step_size * (1 if direction >= 0 else -1)
        if self.end_mode is EndMode.LOOP and last > first:
            span = last - first + 1
            return first + (target - first) % span
        return max(first, min(target, last))
