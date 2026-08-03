"""Media: the playback clock, the source, and the threaded controller."""

from __future__ import annotations

import pytest

from vidtriage.core.errors import SourceOpenError
from vidtriage.media.clock import MAX_SPEED, MIN_SPEED, EndMode, PlaybackClock
from vidtriage.media.source import VideoFileSource


class TestPlaybackClock:
    def test_interval_follows_fps_and_speed(self):
        assert PlaybackClock(fps=25.0).interval_ms == 40
        assert PlaybackClock(fps=25.0, speed=2.0).interval_ms == 20

    def test_interval_is_clamped_at_both_ends(self):
        """A 0 ms timer starves the event loop; a huge one looks like a hang."""
        assert PlaybackClock(fps=1000.0, speed=MAX_SPEED).interval_ms >= 1
        assert PlaybackClock(fps=0.2, speed=MIN_SPEED).interval_ms <= 1000

    def test_speed_is_clamped(self):
        assert PlaybackClock(speed=1e6).speed == MAX_SPEED
        assert PlaybackClock(speed=0.0).speed == MIN_SPEED

    def test_still_image_reports_static(self):
        assert PlaybackClock(fps=0.0).is_static

    def test_next_index_stops_at_the_end(self):
        clock = PlaybackClock(end_mode=EndMode.STOP)
        assert clock.next_index(8, 10) == 9
        assert clock.next_index(9, 10) is None

    def test_next_index_wraps_when_looping(self):
        clock = PlaybackClock(end_mode=EndMode.LOOP)
        assert clock.next_index(9, 10) == 0

    def test_loop_range_bounds_playback(self):
        clock = PlaybackClock(end_mode=EndMode.LOOP)
        clock.set_loop_range(4, 6)
        assert clock.next_index(5, 100) == 6
        assert clock.next_index(6, 100) == 4

    def test_loop_range_is_order_insensitive(self):
        clock = PlaybackClock()
        clock.set_loop_range(20, 5)
        assert clock.effective_range(100) == (5, 20)

    def test_loop_range_is_clamped_to_the_video(self):
        clock = PlaybackClock()
        clock.set_loop_range(0, 9999)
        assert clock.effective_range(50) == (0, 49)

    def test_step_clamps_at_the_boundaries(self):
        clock = PlaybackClock(step_size=5, end_mode=EndMode.STOP)
        assert clock.step_index(0, 10, direction=-1) == 0
        assert clock.step_index(8, 10, direction=1) == 9

    def test_step_wraps_when_looping(self):
        clock = PlaybackClock(step_size=5, end_mode=EndMode.LOOP)
        assert clock.step_index(8, 10, direction=1) == 3
        assert clock.step_index(1, 10, direction=-1) == 6


class TestVideoFileSource:
    def test_reports_metadata(self, sample_video):
        with VideoFileSource(sample_video) as source:
            assert source.info.size.as_int() == (160, 120)
            assert source.info.fps == pytest.approx(20.0, abs=0.5)
            assert source.info.frame_count == 40

    def test_sequential_read_indexes_from_zero(self, sample_video):
        """The frame just read is at ``position``, not ``POS_FRAMES``."""
        with VideoFileSource(sample_video) as source:
            assert source.read().ref.index == 0
            assert source.read().ref.index == 1
            assert source.read().ref.index == 2

    @pytest.mark.parametrize("target", [0, 1, 5, 19, 39])
    def test_seek_lands_on_the_requested_frame(self, sample_video, target):
        with VideoFileSource(sample_video) as source:
            assert source.read_at(target).ref.index == target

    def test_backwards_seek(self, sample_video):
        with VideoFileSource(sample_video) as source:
            source.read_at(30)
            assert source.read_at(3).ref.index == 3

    def test_frames_are_rgb_not_bgr(self, sample_video):
        """The clip is written with a BGR red ramp; channel 0 must dominate."""
        with VideoFileSource(sample_video) as source:
            frame = source.read_at(30)
            assert frame.image[:, :, 0].mean() > frame.image[:, :, 2].mean() + 50
            assert frame.image.dtype.name == "uint8"

    def test_read_past_the_end_returns_none(self, sample_video):
        with VideoFileSource(sample_video) as source:
            source.seek(39)
            assert source.read() is not None
            assert source.read() is None

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(SourceOpenError):
            VideoFileSource(tmp_path / "nope.mp4")


class TestPlaybackController:
    @pytest.fixture
    def controller(self, qapp, pump):
        from vidtriage.media.controller import PlaybackController

        controller = PlaybackController()
        yield controller
        controller.shutdown()

    def test_open_emits_the_first_frame(self, controller, sample_video, pump):
        seen = []
        controller.frame_changed.connect(lambda f: seen.append(f.index))
        controller.open(sample_video)
        pump(600)
        assert controller.is_loaded
        assert seen and seen[-1] == 0

    def test_rapid_seeks_collapse_into_one_decode(self, controller, sample_video, pump):
        """Dragging the scrub bar must not decode every intermediate position."""
        controller.open(sample_video)
        pump(600)
        seen = []
        controller.frame_changed.connect(lambda f: seen.append(f.index))
        for i in range(50):
            controller.seek_index(i % 35)
        pump(500)
        assert 0 < len(seen) <= 3, f"expected coalescing, got {len(seen)} decodes"

    def test_stepping(self, controller, sample_video, pump):
        controller.open(sample_video)
        pump(600)
        controller.seek_index(10)
        pump(200)
        seen = []
        controller.frame_changed.connect(lambda f: seen.append(f.index))
        controller.step_forward()
        pump(200)
        assert seen[-1] == 11
        controller.step_backward()
        pump(200)
        assert seen[-1] == 10

    def test_step_size_is_honoured(self, controller, sample_video, pump):
        controller.open(sample_video)
        pump(600)
        controller.set_step_size(5)
        controller.seek_index(10)
        pump(250)
        seen = []
        controller.frame_changed.connect(lambda f: seen.append(f.index))
        controller.step_forward()
        pump(250)
        assert seen[-1] == 15

    def test_playback_reaches_the_end_and_stops(self, controller, sample_video, pump):
        controller.open(sample_video)
        pump(600)
        controller.set_end_mode(EndMode.STOP)
        controller.seek_index(36)
        pump(250)
        ended = []
        controller.reached_end.connect(ended.append)
        controller.play()
        pump(1500)
        assert ended
        assert not controller.is_playing

    def test_loop_mode_wraps_instead_of_ending(self, controller, sample_video, pump):
        controller.open(sample_video)
        pump(600)
        controller.set_end_mode(EndMode.LOOP)
        controller.seek_index(37)
        pump(250)
        ended, seen = [], []
        controller.reached_end.connect(ended.append)
        controller.frame_changed.connect(lambda f: seen.append(f.index))
        controller.play()
        pump(1200)
        controller.pause()
        assert not ended
        assert min(seen) < 5, f"never wrapped: {sorted(set(seen))[:8]}"

    def test_open_failure_is_reported(self, controller, tmp_path, pump):
        failures = []
        controller.open_failed.connect(lambda p, m: failures.append(m))
        controller.open(tmp_path / "missing.mp4")
        pump(400)
        assert failures
        assert not controller.is_loaded

    def test_close_and_wait_releases_the_file(self, controller, sample_video, pump):
        """Triage moves files; the decoder must have let go first."""
        controller.open(sample_video)
        pump(600)
        assert controller.close_and_wait(2000)
        assert not controller.is_loaded
