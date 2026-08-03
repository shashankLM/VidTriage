"""The transport bar: scrub, play/pause, file navigation, position readout.

The slider is indexed in **frames**, not per-mille of the duration. The old
0–1000 mapping quantised every seek on a video longer than ~33 seconds, so
stepping one frame and then scrubbing could land you somewhere else entirely.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..core.frames import Frame
from ..media.controller import PlaybackController
from ..view.theme import Theme, ThemedMixin
from .library import MediaLibrary

__all__ = ["TransportBar", "format_timecode"]

_PLAY = "▶"
_PAUSE = "⏸"


def format_timecode(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


class TransportBar(QFrame, ThemedMixin):
    """Playback and file-navigation controls, bound to a controller and library."""

    previous_requested = Signal()
    next_requested = Signal()

    def __init__(
        self,
        playback: PlaybackController,
        library: MediaLibrary,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._playback = playback
        self._library = library
        self._dragging = False

        self.setFrameShape(QFrame.Shape.StyledPanel)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(2)

        row = QHBoxLayout()
        self._btn_prev = QPushButton("◀◀")
        self._btn_prev.setToolTip("Previous file (Up)")
        self._btn_prev.setFixedWidth(48)
        self._btn_prev.clicked.connect(self.previous_requested)
        row.addWidget(self._btn_prev)

        self._btn_play = QPushButton(_PLAY)
        self._btn_play.setToolTip("Play / Pause (Space)")
        self._btn_play.setFixedWidth(40)
        self._btn_play.clicked.connect(playback.toggle_play)
        row.addWidget(self._btn_play)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.setTracking(True)
        self._slider.sliderPressed.connect(self._on_press)
        self._slider.sliderReleased.connect(self._on_release)
        self._slider.valueChanged.connect(self._on_value_changed)
        row.addWidget(self._slider, stretch=1)

        self._time_label = QLabel("00:00 / 00:00")
        self._time_label.setMinimumWidth(110)
        self._time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self._time_label)

        self._btn_next = QPushButton("▶▶")
        self._btn_next.setToolTip("Next file (Down)")
        self._btn_next.setFixedWidth(48)
        self._btn_next.clicked.connect(self.next_requested)
        row.addWidget(self._btn_next)
        root.addLayout(row)

        self._info_label = QLabel("No media loaded")
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._info_label)

        playback.frame_changed.connect(self._on_frame)
        playback.opened.connect(self._on_opened)
        playback.playing_changed.connect(self._on_playing_changed)
        library.items_changed.connect(self._refresh_nav_buttons)
        library.current_changed.connect(lambda _p: self._refresh_nav_buttons())

        self._refresh_nav_buttons()
        self.init_theme()

    def apply_theme(self, theme: Theme) -> None:
        self._info_label.setStyleSheet(f"color: {theme.info_fg};")

    # ── playback callbacks ──────────────────────────────────────────────

    def _on_opened(self, info) -> None:
        self._slider.blockSignals(True)
        self._slider.setRange(0, max(info.frame_count - 1, 0))
        self._slider.setEnabled(info.frame_count > 1)
        self._slider.blockSignals(False)
        self._update_labels()

    def _on_frame(self, frame: Frame) -> None:
        if not self._dragging:
            self._slider.blockSignals(True)
            self._slider.setValue(frame.index)
            self._slider.blockSignals(False)
        self._update_labels()

    def _on_playing_changed(self, playing: bool) -> None:
        self._btn_play.setText(_PAUSE if playing else _PLAY)

    # ── slider ──────────────────────────────────────────────────────────

    def _on_press(self) -> None:
        self._dragging = True

    def _on_release(self) -> None:
        self._dragging = False
        self._playback.seek_index(self._slider.value())

    def _on_value_changed(self, value: int) -> None:
        # Live scrubbing while dragging; the decoder coalesces the burst.
        if self._dragging:
            self._playback.seek_index(value)

    # ── labels ──────────────────────────────────────────────────────────

    def _update_labels(self) -> None:
        info = self._playback.info
        if info is None:
            self._time_label.setText("00:00 / 00:00")
            self._info_label.setText("No media loaded")
            return

        index = self._playback.current_index
        # A still has one frame and no frame rate; reporting "frame 0/0 @ 0fps"
        # for it is noise dressed up as information.
        is_still = info.frame_count <= 1 and info.fps <= 0
        self._time_label.setText(
            "still" if is_still else
            f"{format_timecode(info.time_of(index))} / {format_timecode(info.duration)}",
        )

        position = ""
        if len(self._library) > 1:
            position = f"[{self._library.index + 1}/{len(self._library)}]  "
        width, height = info.size.as_int()
        detail = (
            f"{width}×{height}" if is_still else
            f"frame {index}/{max(info.frame_count - 1, 0)}"
            f"  ·  {width}×{height} @ {info.fps:.3g}fps"
        )
        self._info_label.setText(f"{position}{info.path.name}  ·  {detail}")

    def _refresh_nav_buttons(self) -> None:
        self._btn_prev.setEnabled(self._library.has_previous)
        self._btn_next.setEnabled(self._library.has_next)
        self._update_labels()
