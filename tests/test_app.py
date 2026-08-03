"""End-to-end: the shell, the triage workflow, and model-prompted annotation.

These exercise the paths a user actually takes, through real files on disk.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence

import numpy as np
import pytest

from vidtriage.app.context import AppContext
from vidtriage.app.window import MainWindow
from vidtriage.core.annotations import MANUAL_SOURCE, Annotation
from vidtriage.core.geometry import Point, Rect
from vidtriage.persistence.settings import Settings
from vidtriage.persistence.sidecar import load_annotations, sidecar_path_for
from vidtriage.plugins.manager import PluginManager
from vidtriage.plugins.models import Capability, InferenceModel, InferenceRequest


@pytest.fixture
def app_ctx(qapp, tmp_path, pump):
    context = AppContext(
        settings=Settings(tmp_path / "settings.json"),
        plugin_manager=PluginManager(state_file=tmp_path / "plugins.json"),
    )
    yield context
    context.shutdown()
    pump(50)


@pytest.fixture
def full_app(app_ctx, pump):
    """The shell with every built-in plugin active — what a user actually gets."""
    window = MainWindow(app_ctx)
    app_ctx.plugins.discover(user_dir=None)
    app_ctx.plugins.activate_all(app_ctx)
    window.sync_panels()
    pump(100)
    yield app_ctx, window
    window.close()
    pump(50)


class TestShell:
    def test_every_builtin_plugin_activates(self, full_app):
        context, _window = full_app
        for plugin_id in ("triage", "annotate", "guides", "yolo", "sam"):
            state = context.plugins.states[plugin_id]
            assert state.active, f"{plugin_id}: {state.error or state.availability.reason}"

    def test_registries_are_populated(self, full_app):
        context, _window = full_app
        assert {"select", "pan", "box", "point", "polygon"} <= set(context.canvas.tools.keys())
        assert {"coco", "yolo", "csv"} <= set(context.exporters.keys())
        assert {"yolo.detect", "yolo.segment", "sam.predict"} <= set(context.models.keys())
        assert {"annotate.panel", "triage.explorer"} <= set(context.panels.keys())

    def test_menus_and_shortcuts_are_generated(self, full_app):
        context, window = full_app
        titles = {a.text().replace("&", "") for a in window.menuBar().actions()}
        assert {"File", "Edit", "View", "Playback", "Annotate", "Models", "Tools", "Help"} <= titles
        assert window._shortcuts.count == len(context.commands.shortcut_map())

    def test_no_shortcut_is_claimed_twice(self, full_app):
        context, _window = full_app
        claims: dict[str, list[str]] = {}
        for command in context.commands.values():
            if command.shortcut:
                claims.setdefault(command.shortcut, []).append(command.id)
        conflicts = {k: v for k, v in claims.items() if len(v) > 1}
        assert not conflicts, f"shortcut conflicts: {conflicts}"

    def test_disabling_a_plugin_removes_its_contributions(self, full_app):
        """Off means gone — no orphan menu entries or dead shortcuts."""
        context, _window = full_app
        assert "guides.lines" in context.canvas.layers
        before = len(context.commands)

        context.plugins.set_enabled("guides", False, context)
        assert "guides.lines" not in context.canvas.layers
        assert "overlay.guides.lines" not in context.commands
        assert len(context.commands) < before

        context.plugins.set_enabled("guides", True, context)
        assert "guides.lines" in context.canvas.layers

    def test_a_new_model_appears_in_the_menu_by_itself(self, full_app):
        """Registering a model is the entire cost of adding one."""
        context, _window = full_app

        class Toy(InferenceModel):
            id = "test.toy"
            display_name = "Toy"
            capabilities = Capability.WHOLE_FRAME

            def infer(self, request):
                return []

        assert "model.run.test.toy" not in context.commands
        context.models.register("test.toy", Toy())
        assert "model.run.test.toy" in context.commands
        context.models.unregister("test.toy")
        assert "model.run.test.toy" not in context.commands

    def test_prompt_only_models_get_no_run_command(self, full_app):
        """SAM cannot run unprompted, so a permanently-disabled entry is noise."""
        context, _window = full_app
        assert "model.run.sam.predict" not in context.commands
        assert "model.run.yolo.detect" in context.commands


class TestMediaAndAnnotations:
    def test_opening_a_video_shows_a_frame(self, full_app, fresh_video, pump):
        context, _window = full_app
        context.open_media(fresh_video)
        pump(700)
        assert context.current_frame is not None
        assert context.canvas.has_frame
        assert context.media_info.frame_count == 40

    def test_annotations_autosave_to_a_sidecar(self, full_app, sample_video, tmp_path, pump):
        context, _window = full_app
        clip = tmp_path / "auto.mp4"
        shutil.copy(sample_video, clip)

        context.open_media(clip)
        pump(700)
        context.annotations.add(
            Annotation(context.current_frame.ref, Rect(1, 2, 30, 40), label="thing"),
        )
        context.flush_annotations()

        assert sidecar_path_for(clip).exists()
        assert load_annotations(clip)[0].label == "thing"

    def test_annotations_reload_when_the_file_is_reopened(
        self, full_app, sample_video, tmp_path, pump,
    ):
        context, _window = full_app
        clip = tmp_path / "reload.mp4"
        other = tmp_path / "other.mp4"
        shutil.copy(sample_video, clip)
        shutil.copy(sample_video, other)

        context.library.set_items([clip, other], keep_current=False)
        pump(700)
        context.annotations.add(
            Annotation(context.current_frame.ref, Rect(5, 5, 15, 15), label="kept"),
        )

        context.library.next()      # leaving clip flushes it
        pump(700)
        assert len(context.annotations) == 0

        context.library.previous()  # returning reloads it
        pump(700)
        assert len(context.annotations) == 1
        assert context.annotations.all()[0].label == "kept"

    def test_a_corrupt_sidecar_does_not_stop_the_video_opening(
        self, full_app, sample_video, tmp_path, pump,
    ):
        context, _window = full_app
        clip = tmp_path / "bad_sidecar.mp4"
        shutil.copy(sample_video, clip)
        sidecar_path_for(clip).write_text("{{{ not json")

        context.open_media(clip)
        pump(700)
        assert context.canvas.has_frame
        assert len(context.annotations) == 0


class TestModelPrompting:
    """The headline workflow: point or box at something, a model runs there."""

    class PatchModel(InferenceModel):
        id = "test.patch"
        display_name = "Patch"
        capabilities = Capability.BOX_PROMPT | Capability.POINT_PROMPT

        def __init__(self):
            super().__init__()
            self.seen: list[InferenceRequest] = []

        def infer(self, request: InferenceRequest) -> Sequence[Annotation]:
            self.seen.append(request)
            region = request.prompt.region or Rect(0, 0, 4, 4)
            return [request.annotation(region, label="found", score=0.77, source=self.id)]

    @pytest.fixture
    def armed(self, full_app, fresh_video, pump):
        context, _window = full_app
        context.open_media(fresh_video)
        pump(700)
        model = self.PatchModel()
        context.models.register(model.id, model)
        annotate = context.plugins.plugins.require("annotate")
        annotate.set_prompt_model(model.id)
        annotate.set_current_label("light")
        return context, annotate, model

    def test_a_box_drag_runs_the_model_on_that_region(self, armed, pump):
        from vidtriage.view.tools import ToolResult

        context, _annotate, model = armed
        context.canvas.tool_result.emit(ToolResult("box", Rect(10, 20, 60, 90)))
        pump(900)

        assert len(model.seen) == 1
        assert model.seen[0].prompt.region.as_xyxy() == (10, 20, 60, 90)
        assert len(context.annotations) == 1
        assert context.annotations.all()[0].source == "test.patch"

    def test_the_model_receives_the_true_frame_pixels(self, armed, pump):
        """Overlays must not be baked into what the model sees."""
        from vidtriage.view.tools import ToolResult

        context, _annotate, model = armed
        context.canvas.frame_info_layer.visible = True
        context.canvas.crosshair_layer.visible = True
        pump(50)
        expected = context.current_frame.image.copy()

        context.canvas.tool_result.emit(ToolResult("box", Rect(0, 0, 50, 50)))
        pump(900)
        assert np.array_equal(model.seen[0].frame.image, expected)

    def test_a_point_click_prompts_the_model(self, armed, pump):
        from vidtriage.view.tools import ToolResult

        context, _annotate, model = armed
        context.canvas.tool_result.emit(ToolResult("point", Point(40, 50), positive=True))
        pump(900)
        prompt = model.seen[0].prompt
        assert prompt.positive_points == (Point(40, 50),)

    def test_shift_accumulates_points_into_one_prompt(self, armed, pump):
        from vidtriage.view.tools import ToolResult

        context, _annotate, model = armed
        context.canvas.tool_result.emit(ToolResult("point", Point(10, 10), positive=True))
        pump(700)
        context.canvas.tool_result.emit(
            ToolResult("point", Point(20, 20), positive=False, additive=True),
        )
        pump(900)

        latest = model.seen[-1].prompt
        assert latest.positive_points == (Point(10, 10),)
        assert latest.negative_points == (Point(20, 20),)

    def test_without_shift_a_click_starts_a_fresh_prompt(self, armed, pump):
        from vidtriage.view.tools import ToolResult

        context, _annotate, model = armed
        context.canvas.tool_result.emit(ToolResult("point", Point(10, 10)))
        pump(700)
        context.canvas.tool_result.emit(ToolResult("point", Point(90, 90)))
        pump(900)
        assert model.seen[-1].prompt.points == ((Point(90, 90), True),)

    def test_unlabelled_predictions_inherit_the_working_label(self, armed, pump):
        from vidtriage.view.tools import ToolResult

        class Unlabelled(self.PatchModel):
            id = "test.unlabelled"

            def infer(self, request):
                return [request.annotation(Rect(0, 0, 9, 9), source=self.id)]

        context, annotate, _model = armed
        model = Unlabelled()
        context.models.register(model.id, model)
        annotate.set_prompt_model(model.id)
        context.canvas.tool_result.emit(ToolResult("box", Rect(1, 1, 8, 8)))
        pump(900)
        assert context.annotations.all()[0].label == "light"

    def test_manual_mode_saves_the_shape_as_drawn(self, armed, pump):
        from vidtriage.view.tools import ToolResult

        context, annotate, model = armed
        annotate.set_prompt_model(None)
        context.canvas.tool_result.emit(ToolResult("box", Rect(3, 4, 33, 44)))
        pump(300)

        assert model.seen == []
        annotation = context.annotations.all()[0]
        assert annotation.source == MANUAL_SOURCE
        assert annotation.label == "light"
        assert annotation.geometry.as_xyxy() == (3, 4, 33, 44)

    def test_accepting_predictions_keeps_their_provenance(self, armed, pump):
        from vidtriage.view.tools import ToolResult

        context, annotate, _model = armed
        context.canvas.tool_result.emit(ToolResult("box", Rect(1, 1, 20, 20)))
        pump(900)
        assert context.annotations.all()[0].is_prediction

        annotate.promote_predictions()
        promoted = context.annotations.all()[0]
        assert promoted.source == MANUAL_SOURCE
        assert promoted.attributes["predicted_by"] == "test.patch"

    def test_prompts_do_not_leak_across_frames(self, armed, pump):
        """A point picked on frame 3 must not segment frame 40."""
        from vidtriage.view.tools import ToolResult

        context, _annotate, model = armed
        context.canvas.tool_result.emit(ToolResult("point", Point(10, 10)))
        pump(700)
        context.playback.seek_index(20)
        pump(500)
        context.canvas.tool_result.emit(
            ToolResult("point", Point(30, 30), additive=True),
        )
        pump(900)
        assert model.seen[-1].prompt.points == ((Point(30, 30), True),)

    def test_an_unavailable_model_reports_instead_of_crashing(self, full_app, fresh_video, pump):
        from vidtriage.plugins.models import Availability

        class Unavailable(InferenceModel):
            id = "test.unavailable"
            display_name = "Nope"
            capabilities = Capability.WHOLE_FRAME

            def availability(self):
                return Availability.missing_package("nonexistent_pkg")

            def infer(self, request):
                raise AssertionError("must never run")

        context, _window = full_app
        context.open_media(fresh_video)
        pump(700)
        messages = []
        context.status_message.connect(lambda text, _t: messages.append(text))
        assert context.run_model(Unavailable()) is None
        assert any("pip install nonexistent_pkg" in m for m in messages)


@pytest.fixture
def session_dirs(tmp_path, sample_video):
    source = tmp_path / "in"
    output = tmp_path / "out"
    source.mkdir()
    output.mkdir()
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        shutil.copy(sample_video, source / name)
    return source, output


@pytest.fixture
def triage(full_app, session_dirs, pump):
    from vidtriage.plugins.builtin.triage.models import ClassEntry
    from vidtriage.plugins.builtin.triage.session import Session

    context, _window = full_app
    source, output = session_dirs
    plugin = context.plugins.plugins.require("triage")
    plugin._start_session(
        Session(source, output, [ClassEntry("1", "cat"), ClassEntry("2", "dog")]),
    )
    pump(700)
    return context, plugin, source, output


class TestTriageWorkflow:
    def test_session_loads_the_pending_videos(self, triage):
        _context, plugin, _source, _output = triage
        assert len(plugin.session.pending) == 3
        assert plugin.current_item is not None

    def test_class_keys_become_commands(self, triage):
        context, _plugin, _source, _output = triage
        shortcuts = context.commands.shortcut_map()
        assert shortcuts["1"].title.endswith("cat")
        assert shortcuts["2"].title.endswith("dog")
        assert "New class" in shortcuts["3"].title

    def test_classifying_touches_no_file_and_advances(self, triage, pump):
        """The whole point: a decision is a log line, not a move."""
        from vidtriage.plugins.builtin.triage.models import ClassEntry

        _context, plugin, source, output = triage
        first = plugin.current_item
        before = sorted(p.name for p in source.iterdir())

        plugin._classify(ClassEntry("1", "cat"))
        pump(700)

        assert sorted(p.name for p in source.iterdir()) == before
        assert not (output / "cat").exists()
        assert first.class_name == "cat"
        assert len(plugin.session.pending) == 2
        assert plugin.current_item is not first

    def test_the_decision_lands_in_the_log(self, triage, pump):
        from vidtriage.plugins.builtin.triage.ledger import read_log
        from vidtriage.plugins.builtin.triage.models import ClassEntry

        _context, plugin, _source, _output = triage
        first = plugin.current_item
        plugin._classify(ClassEntry("1", "cat"))
        pump(600)

        decisions = read_log(plugin.session.log_path)
        assert [(d.file, d.class_name) for d in decisions] == [(first.name, "cat")]
        assert decisions[0].at, "a decision with no timestamp is unreadable later"

    def test_undo_appends_a_correction_rather_than_rewriting(self, triage, pump):
        """Append-only is what lets a concurrent reader trust the file."""
        from vidtriage.plugins.builtin.triage.ledger import read_log
        from vidtriage.plugins.builtin.triage.models import ClassEntry

        _context, plugin, source, _output = triage
        first = plugin.current_item
        plugin._classify(ClassEntry("1", "cat"))
        pump(600)
        plugin._undo()
        pump(600)

        assert (source / first.name).exists()
        assert len(plugin.session.pending) == 3
        assert [d.class_name for d in read_log(plugin.session.log_path)] == ["cat", None]

    def test_reclassifying_is_just_a_later_record(self, triage, pump):
        from vidtriage.plugins.builtin.triage.ledger import read_log, replay
        from vidtriage.plugins.builtin.triage.models import ClassEntry

        context, plugin, _source, _output = triage
        item = plugin.current_item
        plugin._classify(ClassEntry("1", "cat"))
        pump(600)

        context.library.set_index(plugin._order.index(item))
        pump(600)
        plugin._classify(ClassEntry("2", "dog"))
        pump(600)

        assert item.class_name == "dog"
        assert [d.class_name for d in read_log(plugin.session.log_path)] == ["cat", "dog"]
        assert replay([plugin.session.log_path])[item.name] == "dog"

    def test_mark_error(self, triage, pump):
        _context, plugin, _source, output = triage
        item = plugin.current_item
        plugin._mark_error()
        pump(700)

        assert item.class_name == "_errors"
        assert not (output / "_errors").exists()

    def test_a_snapshot_copies_without_disturbing_the_source(self, triage, pump, tmp_path):
        from vidtriage.plugins.builtin.triage.models import ClassEntry
        from vidtriage.plugins.builtin.triage.snapshot import write_snapshot

        _context, plugin, source, _output = triage
        item = plugin.current_item
        plugin._classify(ClassEntry("1", "cat"))
        pump(600)

        target = tmp_path / "snap"
        result = write_snapshot(plugin.session.classified, target)

        assert result.written == 1
        assert (target / "cat" / item.name).exists()
        assert (source / item.name).exists(), "the source must survive a snapshot"

    def test_annotations_travel_into_the_snapshot(self, triage, pump, tmp_path):
        """A sidecar left behind would orphan every label on the clip."""
        from vidtriage.plugins.builtin.triage.models import ClassEntry
        from vidtriage.plugins.builtin.triage.snapshot import write_snapshot

        context, plugin, source, _output = triage
        item = plugin.current_item
        context.annotations.add(
            Annotation(context.current_frame.ref, Rect(1, 1, 9, 9), label="sticky"),
        )
        plugin._classify(ClassEntry("2", "dog"))
        pump(700)
        context.flush_annotations()

        target = tmp_path / "snap"
        write_snapshot(plugin.session.classified, target)

        copied = target / "dog" / item.name
        assert sidecar_path_for(source / item.name).exists(), "the original keeps its labels"
        assert load_annotations(copied)[0].label == "sticky"

    def test_a_reopened_session_recovers_prior_classifications(self, triage, pump):
        from vidtriage.plugins.builtin.triage.models import ClassEntry
        from vidtriage.plugins.builtin.triage.session import Session

        _context, plugin, source, output = triage
        plugin._classify(ClassEntry("1", "cat"))
        pump(600)

        reopened = Session(source, output, [])
        reopened.load()
        assert len(reopened.classified) == 1
        assert reopened.classified[0].class_name == "cat"
        # The class list is rebuilt from what the replayed logs mention.
        assert any(c.name == "cat" for c in reopened.classes)
        assert reopened.log_path != plugin.session.log_path, "a rerun is its own pass"

    def test_duplicate_filenames_are_detected(self, tmp_path, sample_video):
        from vidtriage.plugins.builtin.triage.session import Session

        source = tmp_path / "dupes"
        nested = source / "sub"
        nested.mkdir(parents=True)
        shutil.copy(sample_video, source / "same.mp4")

        session = Session(source, tmp_path / "out2", [])
        session.load()
        session._videos["fake"] = type(session.all_videos[0])(
            original_path=nested / "same.mp4",
        )
        assert "same.mp4" in session.find_duplicate_names()

    def test_disabling_triage_leaves_a_working_annotation_tool(self, triage, pump):
        """Triage is a plugin, so turning it off must not break the app."""
        context, _plugin, _source, _output = triage
        context.plugins.set_enabled("triage", False, context)
        pump(100)

        assert "triage.explorer" not in context.panels
        assert "1" not in context.commands.shortcut_map()
        assert "annotate.panel" in context.panels
        assert context.canvas.has_frame or context.current_media is not None


class TestStartupReport:
    """The plugin table printed on the way up.

    It is the only place a user sees *why* a plugin is missing, so an
    unavailable plugin has to carry its remedy and a broken one has to carry the
    exception rather than a traceback nobody reads at startup.
    """

    @staticmethod
    def _states():
        from vidtriage.plugins.api import Plugin
        from vidtriage.plugins.manager import PluginState
        from vidtriage.plugins.models import Availability

        class Stub(Plugin):
            def __init__(self, plugin_id: str) -> None:
                super().__init__()
                self.id = plugin_id

            def activate(self, ctx) -> None:
                pass

        return [
            PluginState(plugin=Stub("annotate"), origin="builtin", active=True),
            PluginState(
                plugin=Stub("sam"),
                origin="builtin",
                availability=Availability(
                    False,
                    reason="Model weights not found at ~/.vidtriage/weights",
                    remedy="curl -LO https://example.invalid/sam_vit_b.pth",
                ),
            ),
            PluginState(
                plugin=Stub("broken"),
                origin="user:broken.py",
                enabled=False,
                error='Traceback (most recent call last):\n  File "x.py", line 1\n'
                      "RuntimeError: no module named frobnicate\n",
            ),
        ]

    @staticmethod
    def _render(renderable) -> None:
        from vidtriage.core.console import console

        console().print(renderable)

    def test_table_reports_every_plugin_and_its_status(self, capsys):
        from vidtriage.app.startup_report import _plugin_table

        self._render(_plugin_table(self._states()))
        printed = capsys.readouterr().err

        assert "annotate" in printed and "active" in printed
        assert "sam" in printed and "unavailable" in printed
        assert "broken" in printed and "error" in printed

    def test_unavailable_plugin_carries_its_remedy(self, capsys):
        from vidtriage.app.startup_report import _plugin_table

        self._render(_plugin_table(self._states()))
        printed = capsys.readouterr().err.replace("\n", "")

        assert "Model weights not found" in printed
        assert "curl -LO" in printed

    def test_broken_plugin_shows_the_exception_not_the_traceback(self, capsys):
        from vidtriage.app.startup_report import _plugin_table

        self._render(_plugin_table(self._states()))
        printed = capsys.readouterr().err

        assert "RuntimeError: no module named frobnicate" in printed
        assert "Traceback" not in printed, "startup is not the place for a full traceback"

    def test_model_table_reports_capabilities_and_remedies(self, app_ctx, capsys):
        """Plugin availability and model availability are different questions."""
        from vidtriage.app.startup_report import _model_table
        from vidtriage.plugins.models import Availability

        class Ready(InferenceModel):
            id = "test.ready"
            display_name = "Ready"
            capabilities = Capability.WHOLE_FRAME | Capability.BOX_PROMPT

            def infer(self, request):
                return []

        class Blocked(Ready):
            id = "test.blocked"

            def availability(self) -> Availability:
                return Availability(
                    False,
                    reason="No SAM checkpoint found",
                    remedy="curl -LO https://example.invalid/sam.pth",
                )

        app_ctx.models.register(Ready.id, Ready())
        app_ctx.models.register(Blocked.id, Blocked())

        self._render(_model_table(app_ctx))
        printed = capsys.readouterr().err.replace("\n", "")

        assert "test.ready" in printed and "ready" in printed
        assert "whole-frame" in printed and "box" in printed
        assert "No SAM checkpoint found" in printed
        assert "curl -LO" in printed

    def test_a_model_whose_probe_raises_does_not_stop_startup(self, app_ctx, capsys):
        from vidtriage.app.startup_report import _model_table

        class Exploding(InferenceModel):
            id = "test.exploding"
            display_name = "Exploding"
            capabilities = Capability.WHOLE_FRAME

            def availability(self):
                raise RuntimeError("driver gone")

            def infer(self, request):
                return []

        app_ctx.models.register(Exploding.id, Exploding())

        self._render(_model_table(app_ctx))
        printed = capsys.readouterr().err.replace("\n", "")

        assert "unavailable" in printed
        assert "driver gone" in printed

    def test_falls_back_to_one_log_line_without_rich(self, app_ctx, without_rich, caplog):
        """No rich means no table, but the report itself must not vanish."""
        from vidtriage.app.startup_report import report_startup

        app_ctx.plugins.states = {state.id: state for state in self._states()}
        with caplog.at_level("INFO", logger="vidtriage"):
            report_startup(app_ctx)

        assert "annotate=active" in caplog.text
        assert "sam=unavailable" in caplog.text
        assert "broken=error" in caplog.text

    def test_a_broken_plugin_reaches_the_status_bar(self, app_ctx):
        from vidtriage.app.startup_report import report_startup

        seen: list[str] = []
        app_ctx.status_message.connect(lambda message, _timeout: seen.append(message))
        app_ctx.plugins.states = {state.id: state for state in self._states()}
        report_startup(app_ctx)

        assert seen and "1 plugin(s) failed to load" in seen[0]


class TestHelpFormatter:
    def test_uses_rich_argparse_when_available(self):
        from rich_argparse import RichHelpFormatter

        from vidtriage.__main__ import _help_formatter

        assert _help_formatter() is RichHelpFormatter

    def test_falls_back_to_argparse(self, monkeypatch):
        import argparse
        import sys

        from vidtriage.__main__ import _help_formatter

        monkeypatch.setitem(sys.modules, "rich_argparse", None)
        assert _help_formatter() is argparse.HelpFormatter

    def test_help_text_survives_the_formatter(self, capsys):
        """Rich markup is live in help strings; nothing may be silently eaten."""
        from vidtriage.__main__ import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--help"])
        printed = capsys.readouterr().out.replace("\n", " ")

        assert "~/.vidtriage/plugins/" in printed
        assert "Triage videos, annotate frames" in printed
        for flag in ("--no-plugins", "--safe-mode", "--version"):
            assert flag in printed

    def test_flags_still_parse(self):
        from vidtriage.__main__ import parse_args

        args = parse_args(["--safe-mode", "-v"])
        assert args.safe_mode and args.verbose


class TestLaunchOptions:
    """``-i`` names a triage session, so it has to reach the triage plugin.

    It did not: the flag only filled the media library, the plugin restored
    whatever session ran last, and the file panel then described a different set
    of videos than the player was showing.
    """

    @pytest.fixture
    def corpus(self, tmp_path, sample_video):
        source = tmp_path / "clips"
        source.mkdir()
        for name in ("x.mp4", "y.mp4"):
            shutil.copy(sample_video, source / name)
        return source

    @staticmethod
    def _launch(_qapp, tmp_path, pump, **options):
        """``_qapp`` is required for its side effect: a live QApplication."""
        context = AppContext(
            settings=Settings(tmp_path / "settings.json"),
            plugin_manager=PluginManager(state_file=tmp_path / "plugins.json"),
            launch_options=options,
        )
        window = MainWindow(context)
        context.plugins.discover(user_dir=None)
        context.plugins.activate_all(context)
        window.sync_panels()
        pump(400)
        return context, context.plugins.plugins.require("triage")

    def test_input_dir_starts_a_session_for_that_directory(
        self, qapp, corpus, tmp_path, pump,
    ):
        context, plugin = self._launch(qapp, tmp_path, pump, triage_input=corpus)
        try:
            assert plugin.session is not None
            assert plugin.session.input_dir == corpus.resolve()
            assert sorted(i.name for i in plugin.session.pending) == ["x.mp4", "y.mp4"]
        finally:
            context.shutdown()
            pump(50)

    def test_the_file_panel_lists_those_videos(self, qapp, corpus, tmp_path, pump):
        context, plugin = self._launch(qapp, tmp_path, pump, triage_input=corpus)
        try:
            explorer = context.panels.require("triage.explorer").factory()
            plugin._explorer = explorer
            plugin._refresh_explorer()
            assert [i.name for i in explorer.items("pending")] == ["x.mp4", "y.mp4"]
        finally:
            context.shutdown()
            pump(50)

    def test_output_dir_defaults_beside_the_input(self, qapp, corpus, tmp_path, pump):
        context, plugin = self._launch(qapp, tmp_path, pump, triage_input=corpus)
        try:
            assert plugin.session.output_dir == corpus.parent / "clips_triage"
        finally:
            context.shutdown()
            pump(50)

    def test_an_explicit_output_dir_wins(self, qapp, corpus, tmp_path, pump):
        target = tmp_path / "elsewhere"
        context, plugin = self._launch(
            qapp, tmp_path, pump, triage_input=corpus, triage_output=target,
        )
        try:
            assert plugin.session.output_dir == target.resolve()
        finally:
            context.shutdown()
            pump(50)

    def test_a_missing_input_dir_falls_back_to_the_saved_session(
        self, qapp, tmp_path, pump,
    ):
        context, plugin = self._launch(
            qapp, tmp_path, pump, triage_input=tmp_path / "does-not-exist",
        )
        try:
            assert plugin.session is None or plugin.session.input_dir.exists()
        finally:
            context.shutdown()
            pump(50)


class TestCurrentItemResolution:
    """``current_item`` must describe the video actually on screen.

    It used to index ``_order`` with the library's index. The library is shared,
    so anything that replaced the playlist made the two disagree — and then a
    number key filed a decision against a completely different file.
    """

    def test_a_replaced_playlist_cannot_misattribute_a_decision(
        self, triage, pump, tmp_path, sample_video,
    ):
        context, plugin, _source, _output = triage
        intruder = tmp_path / "intruder.mp4"
        shutil.copy(sample_video, intruder)

        context.library.set_items([intruder], keep_current=False)
        pump(300)

        assert context.library.current == intruder
        assert plugin.current_item is None, (
            "the shown video is not in the session, so no decision may be attributed"
        )

    def test_it_tracks_the_library_not_the_list_order(self, triage, pump):
        context, plugin, _source, _output = triage
        for index in range(len(plugin._order)):
            context.library.set_index(index)
            pump(120)
            assert plugin.current_item is not None
            assert plugin.current_item.original_path == context.library.current


class TestFileSearch:
    """The filter box in the file panel.

    The interesting risk is not the filtering — it is that the panel used to
    identify a clicked video by its row number. Filtering makes row N stop
    meaning video N, so these check that clicking a filtered row still selects
    the video that was clicked.
    """

    @pytest.fixture
    def explorer(self, triage, pump):
        context, plugin, _source, _output = triage
        widget = context.panels.require("triage.explorer").factory()
        plugin._explorer = widget
        plugin._refresh_explorer()
        pump(50)
        return context, plugin, widget

    def test_all_files_show_when_the_filter_is_empty(self, explorer):
        _context, _plugin, widget = explorer
        assert [i.name for i in widget.visible_items("pending")] == ["a.mp4", "b.mp4", "c.mp4"]

    def test_filtering_narrows_the_list(self, explorer):
        _context, _plugin, widget = explorer
        widget._search.setText("b")
        assert [i.name for i in widget.visible_items("pending")] == ["b.mp4"]

    def test_the_filter_is_case_insensitive(self, explorer):
        _context, _plugin, widget = explorer
        widget._search.setText("A.MP4")
        assert [i.name for i in widget.visible_items("pending")] == ["a.mp4"]

    def test_terms_are_anded(self, explorer):
        """Long shared prefixes are the case this exists for."""
        _context, _plugin, widget = explorer
        widget._search.setText("mp4 c")
        assert [i.name for i in widget.visible_items("pending")] == ["c.mp4"]
        widget._search.setText("mp4 zzz")
        assert widget.visible_items("pending") == []

    def test_the_header_says_what_is_hidden(self, explorer):
        _context, _plugin, widget = explorer
        assert widget._pending_header.text() == "Pending (3)"
        widget._search.setText("b")
        assert widget._pending_header.text() == "Pending (1 of 3)"

    def test_the_class_name_is_searchable(self, explorer, pump):
        from vidtriage.plugins.builtin.triage.models import ClassEntry

        _context, plugin, widget = explorer
        plugin._classify(ClassEntry("1", "cat"))
        pump(300)

        widget._search.setText("cat")
        assert [i.name for i in widget.visible_items("classified")] == ["a.mp4"]
        assert widget.visible_items("pending") == []

    def test_clicking_a_filtered_row_selects_the_video_that_was_clicked(
        self, explorer, pump,
    ):
        """Row 0 of a filtered list is not video 0 of the session."""
        _context, _plugin, widget = explorer
        seen = []
        widget.file_selected.connect(seen.append)

        widget._search.setText("c")
        widget._pending_list.setCurrentRow(0)
        pump(50)

        assert [i.name for i in seen] == ["c.mp4"]

    def test_the_library_follows_a_filtered_click(self, explorer, pump):
        context, _plugin, widget = explorer
        widget._search.setText("c")
        widget._pending_list.setCurrentRow(0)
        pump(300)
        assert context.library.current.name == "c.mp4"

    def test_clearing_the_filter_restores_everything(self, explorer):
        _context, _plugin, widget = explorer
        widget._search.setText("b")
        widget.clear_filter()
        assert len(widget.visible_items("pending")) == 3

    def test_a_hidden_selection_does_not_jump_the_cursor(self, explorer, pump):
        """Filtering out the current video must not silently select another."""
        _context, _plugin, widget = explorer
        seen = []
        widget.file_selected.connect(seen.append)

        widget._search.setText("zzz-matches-nothing")
        pump(50)
        assert widget.selected_item() is None
        assert seen == [], "hiding a row is not a selection change"

    def test_the_command_shows_the_panel_and_focuses_the_box(self, explorer, pump):
        """Offscreen docks never report visible, so check what was asked for.

        ``set_panel_visible`` persists the flag, and ``focusWidget`` records the
        last child ``setFocus`` was called on — both independent of whether the
        window is actually mapped.
        """
        context, plugin, widget = explorer
        context.window.set_panel_visible("triage.explorer", False)
        assert context.settings.get("panels.triage.explorer.visible") is False

        plugin._focus_search()
        pump(100)

        assert context.settings.get("panels.triage.explorer.visible") is True
        assert widget.focusWidget() is widget._search

    def test_the_shortcut_reaches_the_help_dialog(self, triage):
        """Help is generated from the registry, so registering is all it takes."""
        context, _plugin, _source, _output = triage
        command = context.commands.shortcut_map().get("Ctrl+F")
        assert command is not None
        assert command.id == "triage.search"
        assert command.menu_path[0] == "View"
