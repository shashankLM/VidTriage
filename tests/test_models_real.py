"""Tests against the actual model backends.

Split out because they need real weights and, the first time, network access.
Run them with ``pytest -m models``; the default suite skips them.

They exist because an inference adapter that merely *looks* right is worthless:
the failure modes are channel order, coordinate frames and patch offsets, and
none of those show up until a real forward pass runs.
"""

from __future__ import annotations

import importlib.util

import cv2
import numpy as np
import pytest

from vidtriage.core.frames import Frame, FrameRef
from vidtriage.core.geometry import Point, Rect
from vidtriage.plugins.models import (
    BoxPrompt,
    Capability,
    InferenceRequest,
    PointPrompt,
    WholeFramePrompt,
)

pytestmark = pytest.mark.models

_HAS_ULTRALYTICS = importlib.util.find_spec("ultralytics") is not None
_HAS_SAM = importlib.util.find_spec("segment_anything") is not None


@pytest.fixture
def scene() -> Frame:
    """A synthetic frame with one salient red disc, delivered as RGB."""
    bgr = np.full((480, 640, 3), 200, np.uint8)
    cv2.circle(bgr, (450, 150), 40, (20, 20, 200), -1)  # red, in BGR
    return Frame(FrameRef("synthetic.png", 0), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


@pytest.fixture(scope="module")
def yolo_model():
    """Loaded once per module — weights loading dominates the runtime."""
    from vidtriage.plugins.builtin.yolo import YoloDetectModel

    detector = YoloDetectModel()
    if not detector.availability().ok:
        pytest.skip("ultralytics unavailable")
    return detector


@pytest.fixture(scope="module")
def sam_model():
    from vidtriage.plugins.builtin.sam import SamModel

    segmenter = SamModel()
    availability = segmenter.availability()
    if not availability.ok:
        pytest.skip(f"SAM unavailable: {availability.reason}")
    return segmenter


@pytest.mark.skipif(not _HAS_ULTRALYTICS, reason="ultralytics not installed")
class TestYolo:
    @pytest.fixture
    def model(self, yolo_model):
        return yolo_model

    def test_whole_frame_finds_the_disc(self, model, scene):
        params = model.resolved_params({"confidence": 0.05})
        result = model.run(InferenceRequest(scene, WholeFramePrompt(), params))
        assert result.count >= 1
        box = result.annotations[0].geometry
        assert box.center.distance_to(Point(450, 150)) < 30

    def test_weights_do_not_land_in_the_working_directory(self, model, scene):
        """Ultralytics defaults to downloading into the CWD; we redirect it."""
        from pathlib import Path

        from vidtriage.plugins.builtin.yolo import WEIGHTS_DIR

        model.run(InferenceRequest(scene, WholeFramePrompt(), model.resolved_params()))
        assert (WEIGHTS_DIR / "yolo11n.pt").exists()
        assert not (Path.cwd() / "yolo11n.pt").exists()

    def test_box_prompt_maps_results_back_to_full_frame_coordinates(self, model, scene):
        """The patch offset must be added back, or every box lands near 0,0."""
        params = model.resolved_params({"confidence": 0.05})
        region = Rect(400, 100, 500, 200)
        result = model.run(InferenceRequest(scene, BoxPrompt(region), params))
        assert result.count >= 1
        for annotation in result.annotations:
            assert region.expanded(80).contains(annotation.geometry.center)

    def test_no_detection_escapes_the_frame(self, model, scene):
        params = model.resolved_params({"confidence": 0.01})
        result = model.run(InferenceRequest(scene, WholeFramePrompt(), params))
        for annotation in result.annotations:
            box = annotation.geometry
            assert box.x1 >= -1 and box.x2 <= 641
            assert box.y1 >= -1 and box.y2 <= 481

    def test_channel_order_is_bgr_for_ultralytics(self, model, scene, monkeypatch):
        """A Frame is RGB; ultralytics reads numpy input as BGR.

        Without the swap a red object is presented to the model as blue, which
        silently costs accuracy on exactly the traffic footage this is built for.
        """
        captured = {}
        real_predict = model._predict

        def spy(image, params):
            captured["image"] = image.copy()
            return real_predict(image, params)

        monkeypatch.setattr(model, "_predict", spy)
        model.run(InferenceRequest(scene, WholeFramePrompt(), model.resolved_params()))

        # The adapter is handed RGB; what reaches ultralytics must be BGR, so at
        # the disc the blue channel dominates.
        delivered = captured["image"]
        assert delivered[150, 450, 0] > delivered[150, 450, 2] + 100


@pytest.mark.skipif(not _HAS_SAM, reason="segment_anything not installed")
class TestSam:
    @pytest.fixture
    def model(self, sam_model):
        return sam_model

    def test_point_prompt_returns_a_mask(self, model, scene):
        prompt = PointPrompt(((Point(450, 150), True),))
        result = model.run(InferenceRequest(scene, prompt, model.resolved_params()))
        assert result.count >= 1
        mask = result.annotations[0].geometry
        assert not mask.is_empty
        assert mask.bounds.contains(Point(450, 150))

    def test_box_prompt_returns_a_mask(self, model, scene):
        prompt = BoxPrompt(Rect(400, 100, 500, 200))
        result = model.run(InferenceRequest(scene, prompt, model.resolved_params()))
        assert result.count >= 1
        assert not result.annotations[0].geometry.is_empty

    def test_the_embedding_is_reused_across_prompts_on_one_frame(self, model, scene):
        """The encoder dominates SAM's cost; re-running it per click is unusable."""
        import time

        first = time.perf_counter()
        model.run(InferenceRequest(
            scene, PointPrompt(((Point(450, 150), True),)), model.resolved_params(),
        ))
        first_elapsed = time.perf_counter() - first

        second = time.perf_counter()
        model.run(InferenceRequest(
            scene, PointPrompt(((Point(450, 160), True),)), model.resolved_params(),
        ))
        second_elapsed = time.perf_counter() - second

        assert second_elapsed < first_elapsed * 0.6, (
            f"second prompt took {second_elapsed:.2f}s vs {first_elapsed:.2f}s — "
            "the image embedding does not appear to be cached"
        )


class TestSamAvailabilityWithoutWeights:
    """Runs even without weights — the graceful-degradation path is the point."""

    @pytest.mark.skipif(not _HAS_SAM, reason="segment_anything not installed")
    def test_missing_checkpoint_names_the_fix(self, monkeypatch):
        from vidtriage.plugins.builtin import sam

        monkeypatch.setattr(sam, "find_checkpoint", lambda explicit=None: None)
        availability = sam.SamModel().availability()
        assert not availability.ok
        assert "checkpoint" in availability.reason.lower()
        assert "curl" in availability.remedy or "download" in availability.remedy.lower()

    @pytest.mark.skipif(not _HAS_SAM, reason="segment_anything not installed")
    def test_capabilities_are_prompt_only(self):
        from vidtriage.plugins.builtin.sam import SamModel

        model = SamModel()
        assert model.capabilities & Capability.POINT_PROMPT
        assert model.capabilities & Capability.BOX_PROMPT
        assert not (model.capabilities & Capability.WHOLE_FRAME)
