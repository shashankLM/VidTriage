from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel


class CvPlayerWidget(QWidget):
    position_changed = Signal(float)
    duration_changed = Signal(float)
    file_loaded = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._display = QLabel()
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._display.setStyleSheet("background: #1a1a1a;")
        self._display.setScaledContents(False)
        layout.addWidget(self._display)

        self._error_label = QLabel()
        self._error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._error_label.setStyleSheet("color: #ff6666; font-size: 16px; background: #1a1a1a;")
        self._error_label.hide()
        layout.addWidget(self._error_label)

        self._cap: cv2.VideoCapture | None = None
        self._fps: float = 30.0
        self._frame_count: int = 0
        self._duration: float = 0.0
        self._current_frame: int = 0
        self._loaded = False
        self._paused = False
        self._looping = True

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def load(self, path: Path) -> None:
        self.stop()
        self._error_label.hide()
        self._display.show()

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            self.show_error(f"Cannot open: {path.name}")
            return

        self._cap = cap
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration = self._frame_count / self._fps if self._fps > 0 else 0.0
        self._current_frame = 0
        self._loaded = True
        self._paused = False

        self.duration_changed.emit(self._duration)
        self.file_loaded.emit()

        self._read_and_show()
        self._timer.start(int(1000 / self._fps))

    def _read_and_show(self) -> bool:
        if not self._cap:
            return False
        ret, frame = self._cap.read()
        if not ret:
            if self._looping:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._current_frame = 0
                ret, frame = self._cap.read()
                if not ret:
                    return False
            else:
                return False

        self._current_frame = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        self._show_frame(frame)
        pos = self._current_frame / self._fps if self._fps > 0 else 0.0
        self.position_changed.emit(pos)
        return True

    def _show_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(img)

        label_size = self._display.size()
        scaled = pixmap.scaled(
            label_size, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._display.setPixmap(scaled)

    def _tick(self) -> None:
        if self._paused or not self._loaded:
            return
        self._read_and_show()

    def show_error(self, msg: str) -> None:
        self._display.hide()
        self._error_label.setText(f"Error: {msg}")
        self._error_label.show()
        self._loaded = False

    def play(self) -> None:
        if self._loaded:
            self._paused = False
            if not self._timer.isActive():
                self._timer.start(int(1000 / self._fps))

    def pause(self) -> None:
        self._paused = True

    def toggle_pause(self) -> None:
        if not self._loaded:
            return
        if self._paused:
            self.play()
        else:
            self.pause()

    def is_paused(self) -> bool:
        return self._paused

    def seek(self, seconds: float) -> None:
        if not self._loaded or not self._cap:
            return
        target_frame = int(seconds * self._fps)
        target_frame = max(0, min(target_frame, self._frame_count - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        self._current_frame = target_frame
        self._read_and_show()

    def frame_step(self) -> None:
        if not self._loaded or not self._cap:
            return
        self._paused = True
        self._read_and_show()

    def frame_back_step(self) -> None:
        if not self._loaded or not self._cap:
            return
        self._paused = True
        target = max(0, self._current_frame - 2)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        self._current_frame = target
        self._read_and_show()

    def get_duration(self) -> float:
        return self._duration

    def get_position(self) -> float:
        if self._fps > 0:
            return self._current_frame / self._fps
        return 0.0

    def stop(self) -> None:
        self._timer.stop()
        if self._cap:
            self._cap.release()
            self._cap = None
        self._loaded = False

    def cleanup(self) -> None:
        self.stop()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._loaded and self._paused and self._cap:
            current = self._current_frame
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current - 1))
            self._read_and_show()
