"""The plugin contract, discovery, and the inference runner."""

from __future__ import annotations

import time
from collections.abc import Sequence

import pytest

from vidtriage.core.annotations import Annotation
from vidtriage.core.geometry import Rect
from vidtriage.plugins.models import (
    Availability,
    BoxPrompt,
    Capability,
    InferenceModel,
    InferenceRequest,
    ParamSpec,
    PointPrompt,
    WholeFramePrompt,
)
from vidtriage.plugins.runner import InferenceRunner


class EchoModel(InferenceModel):
    """A model that returns one box, without importing anything heavy."""

    id = "test.echo"
    display_name = "Echo"
    capabilities = Capability.WHOLE_FRAME | Capability.BOX_PROMPT | Capability.POINT_PROMPT
    parameters = (
        ParamSpec("confidence", "Confidence", "float", 0.5, minimum=0.0, maximum=1.0),
        ParamSpec("limit", "Limit", "int", 3, minimum=1, maximum=10),
    )

    def __init__(self, delay: float = 0.0, fail: bool = False) -> None:
        super().__init__()
        self.delay = delay
        self.fail = fail
        self.load_count = 0
        self.calls: list[str] = []

    def load(self) -> None:
        self.load_count += 1

    def infer(self, request: InferenceRequest) -> Sequence[Annotation]:
        self.calls.append(request.id)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("model exploded")
        region = request.prompt.region or request.frame.rect
        return [request.annotation(region, label="echo", score=0.9, source=self.id)]


class TestCapabilities:
    def test_prompt_matching(self):
        model = EchoModel()
        assert model.supports(WholeFramePrompt())
        assert model.supports(BoxPrompt(Rect(0, 0, 5, 5)))

    def test_unsupported_prompt_is_rejected(self):
        class BoxOnly(EchoModel):
            capabilities = Capability.BOX_PROMPT

        assert not BoxOnly().supports(WholeFramePrompt())


class TestAvailability:
    def test_missing_package_carries_a_remedy(self):
        availability = Availability.missing_package("ultralytics")
        assert not availability
        assert "pip install ultralytics" in availability.remedy

    def test_available_is_truthy(self):
        assert Availability.available()


class TestParamSpec:
    def test_clamps_out_of_range(self):
        spec = ParamSpec("c", "C", "float", 0.5, minimum=0.0, maximum=1.0)
        assert spec.coerce(5.0) == 1.0
        assert spec.coerce(-1.0) == 0.0

    def test_falls_back_to_default_on_junk(self):
        spec = ParamSpec("c", "C", "float", 0.5)
        assert spec.coerce("banana") == 0.5

    def test_choice_rejects_unknown_values(self):
        spec = ParamSpec("w", "W", "choice", "a", choices=("a", "b"))
        assert spec.coerce("b") == "b"
        assert spec.coerce("zzz") == "a"

    def test_resolved_params_merges_defaults(self):
        resolved = EchoModel().resolved_params({"confidence": 99})
        assert resolved == {"confidence": 1.0, "limit": 3}


class TestInferenceRequest:
    def test_patch_returns_the_prompt_region(self, rgb_frame):
        frame = rgb_frame(width=100, height=80)
        request = InferenceRequest(frame, BoxPrompt(Rect(10, 20, 40, 60)))
        patch, region = request.patch()
        assert patch.shape[:2] == (40, 30)
        assert region.as_xyxy() == (10, 20, 40, 60)

    def test_patch_clips_to_the_frame(self, rgb_frame):
        frame = rgb_frame(width=100, height=80)
        patch, region = InferenceRequest(frame, BoxPrompt(Rect(-50, -50, 20, 20))).patch()
        assert region.as_xyxy() == (0, 0, 20, 20)
        assert patch.shape[:2] == (20, 20)

    def test_whole_frame_prompt_gives_the_whole_frame(self, rgb_frame):
        frame = rgb_frame(width=100, height=80)
        patch, _region = InferenceRequest(frame).patch()
        assert patch.shape[:2] == (80, 100)

    def test_annotation_helper_pins_the_frame(self, rgb_frame):
        frame = rgb_frame(index=12)
        annotation = InferenceRequest(frame).annotation(Rect(0, 0, 1, 1), label="x")
        assert annotation.frame == frame.ref
        assert annotation.is_prediction


class TestPointPrompt:
    def test_splits_positive_and_negative(self):
        from vidtriage.core.geometry import Point

        prompt = PointPrompt(((Point(1, 1), True), (Point(2, 2), False)))
        assert prompt.positive_points == (Point(1, 1),)
        assert prompt.negative_points == (Point(2, 2),)


class TestRunner:
    @pytest.fixture
    def runner(self, qapp):
        runner = InferenceRunner()
        yield runner
        runner.shutdown()

    def test_delivers_the_result_on_the_gui_thread(self, runner, rgb_frame, pump):
        model = EchoModel()
        results = []
        runner.finished.connect(results.append)
        runner.submit(model, InferenceRequest(rgb_frame()))
        pump(800)
        assert len(results) == 1
        assert results[0].count == 1
        assert results[0].annotations[0].label == "echo"

    def test_loads_lazily_and_only_once(self, runner, rgb_frame, pump):
        model = EchoModel()
        assert model.load_count == 0
        for _ in range(3):
            runner.submit(model, InferenceRequest(rgb_frame()))
            pump(400)
        assert model.load_count == 1

    def test_failures_are_reported_not_raised(self, runner, rgb_frame, pump):
        failures = []
        runner.failed.connect(lambda m, r, msg: failures.append(msg))
        runner.submit(EchoModel(fail=True), InferenceRequest(rgb_frame()))
        pump(800)
        assert failures and "model exploded" in failures[0]

    def test_a_failure_leaves_the_lane_usable(self, runner, rgb_frame, pump):
        model = EchoModel(fail=True)
        runner.submit(model, InferenceRequest(rgb_frame()))
        pump(600)
        model.fail = False
        results = []
        runner.finished.connect(results.append)
        runner.submit(model, InferenceRequest(rgb_frame()))
        pump(600)
        assert len(results) == 1

    def test_newer_requests_supersede_older_ones(self, runner, rgb_frame, pump):
        """Three fast clicks must yield the third answer, not three answers."""
        model = EchoModel(delay=0.25)
        results, superseded = [], []
        runner.finished.connect(results.append)
        runner.superseded.connect(lambda m, r: superseded.append(r))

        ids = [runner.submit(model, InferenceRequest(rgb_frame())) for _ in range(4)]
        pump(2000)

        assert len(results) == 1, f"expected one surviving result, got {len(results)}"
        assert results[0].request_id == ids[-1]
        assert len(superseded) == 3

    def test_busy_state_toggles(self, runner, rgb_frame, pump):
        states = []
        runner.busy_changed.connect(states.append)
        runner.submit(EchoModel(delay=0.1), InferenceRequest(rgb_frame()))
        pump(900)
        assert states[:2] == [True, False]

    def test_cancel_discards_the_result(self, runner, rgb_frame, pump):
        model = EchoModel(delay=0.3)
        results = []
        runner.finished.connect(results.append)
        runner.submit(model, InferenceRequest(rgb_frame()))
        runner.cancel(model.id)
        pump(1200)
        assert results == []

    def test_different_models_run_independently(self, runner, rgb_frame, pump):
        class Other(EchoModel):
            id = "test.other"

        results = []
        runner.finished.connect(results.append)
        runner.submit(EchoModel(), InferenceRequest(rgb_frame()))
        runner.submit(Other(), InferenceRequest(rgb_frame()))
        pump(1000)
        assert {r.model_id for r in results} == {"test.echo", "test.other"}


class TestDiscovery:
    def test_builtins_are_found(self):
        from vidtriage.plugins.manager import PluginManager

        manager = PluginManager()
        manager.discover(user_dir=None)
        assert {"triage", "annotate", "yolo", "sam", "guides"} <= set(manager.states)

    def test_drop_in_plugin_is_discovered(self, tmp_path):
        from vidtriage.plugins.manager import PluginManager

        (tmp_path / "hello.py").write_text(
            "from vidtriage.plugins.api import Plugin\n"
            "class HelloPlugin(Plugin):\n"
            "    id = 'hello'\n"
            "    name = 'Hello'\n"
            "    def activate(self, ctx):\n"
            "        pass\n"
            "PLUGIN = HelloPlugin\n",
        )
        manager = PluginManager()
        manager.discover(user_dir=tmp_path)
        assert "hello" in manager.states
        assert manager.states["hello"].origin == "user:hello.py"

    def test_a_broken_plugin_is_recorded_not_fatal(self, tmp_path):
        """One bad drop-in must never stop the app from starting."""
        from vidtriage.plugins.manager import PluginManager

        (tmp_path / "broken.py").write_text("raise ValueError('bad plugin')\n")
        manager = PluginManager()
        manager.discover(user_dir=tmp_path)
        assert manager.states["broken"].error
        assert manager.states["broken"].status == "error"
        assert {"triage", "annotate"} <= set(manager.states)

    def test_dependencies_activate_first(self):
        from vidtriage.plugins.api import Plugin
        from vidtriage.plugins.manager import PluginManager, PluginState

        class Base(Plugin):
            id = "base"

            def activate(self, ctx):
                pass

        class Dependent(Plugin):
            id = "dependent"
            requires = ("base",)

            def activate(self, ctx):
                pass

        manager = PluginManager()
        for plugin in (Dependent(), Base()):
            manager.states[plugin.id] = PluginState(plugin=plugin, origin="test")
        assert manager.activation_order().index("base") < manager.activation_order().index("dependent")

    def test_a_dependency_cycle_is_reported_not_hung(self):
        from vidtriage.plugins.api import Plugin
        from vidtriage.plugins.manager import PluginManager, PluginState

        class A(Plugin):
            id = "a"
            requires = ("b",)

            def activate(self, ctx):
                pass

        class B(Plugin):
            id = "b"
            requires = ("a",)

            def activate(self, ctx):
                pass

        manager = PluginManager()
        for plugin in (A(), B()):
            manager.states[plugin.id] = PluginState(plugin=plugin, origin="test")
        assert sorted(manager.activation_order()) == ["a", "b"]
