"""The image canvas: coordinates, layers, items, tools."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt

from vidtriage.core.annotations import Annotation, AnnotationStore
from vidtriage.core.frames import Frame, FrameRef
from vidtriage.core.geometry import Mask, Point, Polygon, Rect, Size
from vidtriage.view.canvas import ImageCanvas
from vidtriage.view.theme import set_theme
from vidtriage.view.tools import ToolEvent

SOURCE = "clip.mp4"


@pytest.fixture
def canvas(qapp, pump):
    widget = ImageCanvas()
    widget.resize(800, 600)
    widget.show()
    pump(50)
    yield widget
    widget.close()


@pytest.fixture
def loaded(canvas, pump):
    image = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    frame = Frame(FrameRef(SOURCE, 7), image)
    canvas.set_frame(frame)
    canvas.zoom_to(1.0)
    pump(50)
    return canvas, frame


class TestCoordinates:
    @pytest.mark.parametrize("zoom", [0.05, 0.25, 1.0, 3.7, 12.0, 45.0])
    def test_roundtrip_is_exact(self, loaded, pump, zoom):
        """Sub-pixel accuracy is what lets a click become a model prompt.

        Qt's own mapToScene takes an integer QPoint; at 0.25x that quantises to
        a four-image-pixel error, which is enough to prompt the wrong object.
        """
        canvas, _frame = loaded
        canvas.zoom_to(zoom)
        pump(20)
        for point in (Point(0, 0), Point(320, 240), Point(639, 479), Point(123.5, 77.25)):
            widget = canvas.image_to_widget(point)
            back = canvas.widget_to_image(QPointF(widget.x, widget.y))
            assert back.x == pytest.approx(point.x, abs=1e-6)
            assert back.y == pytest.approx(point.y, abs=1e-6)

    def test_view_transform_matches_qt(self, loaded, pump):
        canvas, _frame = loaded
        canvas.zoom_to(2.5)
        pump(20)
        transform = canvas.view_transform
        for point in (Point(10, 10), Point(500, 400)):
            a = canvas.image_to_widget(point)
            b = transform.image_to_widget(point)
            assert (a.x, a.y) == pytest.approx((b.x, b.y))

    def test_roundtrip_survives_panning(self, loaded, pump):
        canvas, _frame = loaded
        canvas.zoom_to(4.0)
        canvas.center_on_image_point(Point(600, 450))
        pump(20)
        point = Point(600.5, 450.25)
        widget = canvas.image_to_widget(point)
        back = canvas.widget_to_image(QPointF(widget.x, widget.y))
        assert back.x == pytest.approx(point.x, abs=1e-6)

    def test_is_inside_image(self, loaded):
        canvas, _frame = loaded
        assert canvas.is_inside_image(Point(0, 0))
        assert canvas.is_inside_image(Point(639, 479))
        assert not canvas.is_inside_image(Point(640, 480))
        assert not canvas.is_inside_image(Point(-1, 5))


class TestFrame:
    def test_reports_size_and_fits(self, loaded):
        canvas, _frame = loaded
        assert canvas.has_frame
        assert canvas.image_size == Size(640, 480)

    def test_clear(self, loaded, pump):
        canvas, _frame = loaded
        canvas.clear()
        pump(20)
        assert not canvas.has_frame

    def test_error_state_replaces_the_frame(self, loaded, pump):
        canvas, _frame = loaded
        canvas.show_error("Cannot open: broken.mp4")
        pump(20)
        assert not canvas.has_frame

    def test_a_new_video_size_refits(self, canvas, pump):
        canvas.set_frame(Frame(FrameRef(SOURCE, 0), np.zeros((100, 200, 3), np.uint8)))
        pump(20)
        small = canvas.zoom
        canvas.set_frame(Frame(FrameRef("other.mp4", 0), np.zeros((1000, 2000, 3), np.uint8)))
        pump(20)
        assert canvas.zoom < small


class TestOverlays:
    def test_layers_never_touch_the_pixels(self, loaded, pump):
        """The whole point of the layer system: the frame stays pristine.

        A model prompted on this frame must see the real image, not one with a
        frame counter baked into the top-left corner.
        """
        canvas, frame = loaded
        before = frame.image.copy()
        canvas.frame_info_layer.set_media(100, 30.0)
        canvas.frame_info_layer.visible = True
        canvas.crosshair_layer.visible = True
        canvas.busy_layer.set_running("sam", "SAM")
        pump(80)
        canvas.snapshot()
        assert np.array_equal(frame.image, before)

    def test_snapshot_matches_the_image_size(self, loaded, pump):
        canvas, _frame = loaded
        canvas.frame_info_layer.visible = True
        pump(50)
        image = canvas.snapshot()
        assert (image.width(), image.height()) == (640, 480)

    def test_a_layer_that_raises_is_hidden_not_fatal(self, loaded, pump):
        from vidtriage.view.layers import OverlayLayer

        class Exploding(OverlayLayer):
            id = "test.explode"
            title = "Explode"

            def paint(self, painter, ctx):
                raise RuntimeError("bad layer")

        canvas, _frame = loaded
        layer = Exploding()
        canvas.add_layer(layer)
        canvas.snapshot()
        pump(50)
        assert not layer.visible, "a raising layer should disable itself"

    def test_add_and_remove(self, loaded):
        from vidtriage.view.layers import OverlayLayer

        class Noop(OverlayLayer):
            id = "test.noop"

            def paint(self, painter, ctx):
                pass

        canvas, _frame = loaded
        canvas.add_layer(Noop())
        assert canvas.layer("test.noop") is not None
        canvas.remove_layer("test.noop")
        assert canvas.layer("test.noop") is None


class TestAnnotationItems:
    @pytest.fixture
    def store(self, loaded, pump):
        canvas, frame = loaded
        store = AnnotationStore(SOURCE)
        canvas.bind_store(store)
        full = np.zeros((480, 640), bool)
        full[200:260, 100:190] = True
        store.add([
            Annotation(frame.ref, Rect(50, 50, 200, 180), label="car", score=0.9, source="yolo"),
            Annotation(frame.ref, Point(300, 300), label="tip"),
            Annotation(
                frame.ref, Polygon.from_iterable([(400, 100), (500, 120), (460, 220)]),
                label="sign",
            ),
            Annotation(frame.ref, Mask.from_full_frame(full), label="road", source="sam"),
            Annotation(FrameRef(SOURCE, 9), Rect(0, 0, 10, 10), label="elsewhere"),
        ])
        pump(50)
        return canvas, store

    def test_only_the_current_frame_is_rendered(self, store):
        canvas, _store = store
        assert len(canvas._items) == 4

    def test_items_follow_the_frame(self, store, pump):
        canvas, _store = store
        canvas.set_frame(Frame(FrameRef(SOURCE, 9), np.zeros((480, 640, 3), np.uint8)))
        pump(30)
        assert len(canvas._items) == 1
        canvas.set_frame(Frame(FrameRef(SOURCE, 7), np.zeros((480, 640, 3), np.uint8)))
        pump(30)
        assert len(canvas._items) == 4

    def test_selection_is_reported(self, store, pump):
        canvas, annotation_store = store
        target = annotation_store.for_frame(7)[0]
        canvas.select_annotations([target.id])
        pump(30)
        assert canvas.selected_ids() == (target.id,)
        canvas.clear_selection()
        pump(30)
        assert canvas.selected_ids() == ()

    def test_removing_an_annotation_removes_its_item(self, store, pump):
        canvas, annotation_store = store
        annotation_store.remove(annotation_store.for_frame(7)[0])
        pump(30)
        assert len(canvas._items) == 3

    def test_box_resize_uses_the_dragged_handle(self, store):
        from vidtriage.view.items import BoxItem, Handle

        canvas, _annotation_store = store
        item = next(i for i in canvas._items.values() if isinstance(i, BoxItem))
        original = item.annotation.geometry
        resized = item.geometry_after_resize(Handle.BOTTOM_RIGHT, 20, 30)
        assert resized.x1 == original.x1
        assert resized.x2 == original.x2 + 20
        assert resized.y2 == original.y2 + 30

    def test_box_cannot_be_resized_to_nothing(self, store):
        from vidtriage.view.items import BoxItem, Handle

        canvas, _store = store
        item = next(i for i in canvas._items.values() if isinstance(i, BoxItem))
        collapsed = item.geometry_after_resize(Handle.LEFT, 1e6, 0)
        assert collapsed.width > 0 and collapsed.height > 0


class TestTools:
    def test_box_drag_publishes_a_rect(self, loaded):
        canvas, _frame = loaded
        results = []
        canvas.tool_result.connect(results.append)
        canvas.set_tool("box")
        tool = canvas.tools.require("box")
        tool.mouse_press(ToolEvent(Point(10, 10), Point(0, 0), Qt.MouseButton.LeftButton, view_scale=1.0))
        tool.mouse_move(ToolEvent(Point(90, 70), Point(0, 0), view_scale=1.0))
        tool.mouse_release(ToolEvent(Point(90, 70), Point(0, 0), Qt.MouseButton.LeftButton, view_scale=1.0))
        assert results[-1].tool_id == "box"
        assert results[-1].geometry.as_xyxy() == (10, 10, 90, 70)

    def test_a_stray_click_does_not_create_a_box(self, loaded):
        canvas, _frame = loaded
        results = []
        canvas.tool_result.connect(results.append)
        tool = canvas.tools.require("box")
        tool.mouse_press(ToolEvent(Point(10, 10), Point(0, 0), Qt.MouseButton.LeftButton, view_scale=1.0))
        tool.mouse_release(ToolEvent(Point(10.2, 10.1), Point(0, 0), Qt.MouseButton.LeftButton, view_scale=1.0))
        assert results == []

    def test_the_drag_threshold_is_in_screen_pixels(self, loaded):
        """At 30x, a 0.2 image-pixel drag is 6 screen pixels — a real drag."""
        canvas, _frame = loaded
        results = []
        canvas.tool_result.connect(results.append)
        tool = canvas.tools.require("box")
        tool.mouse_press(ToolEvent(Point(10, 10), Point(0, 0), Qt.MouseButton.LeftButton, view_scale=30.0))
        tool.mouse_release(ToolEvent(Point(10.2, 10.1), Point(0, 0), Qt.MouseButton.LeftButton, view_scale=30.0))
        assert len(results) == 1

    def test_right_click_is_a_negative_point(self, loaded):
        canvas, _frame = loaded
        results = []
        canvas.tool_result.connect(results.append)
        tool = canvas.tools.require("point")
        tool.mouse_press(ToolEvent(Point(5, 6), Point(0, 0), Qt.MouseButton.LeftButton, view_scale=1.0))
        assert results[-1].positive is True
        tool.mouse_press(ToolEvent(Point(5, 6), Point(0, 0), Qt.MouseButton.RightButton, view_scale=1.0))
        assert results[-1].positive is False

    def test_shift_marks_the_prompt_additive(self, loaded):
        canvas, _frame = loaded
        results = []
        canvas.tool_result.connect(results.append)
        tool = canvas.tools.require("point")
        tool.mouse_press(ToolEvent(
            Point(5, 6), Point(0, 0), Qt.MouseButton.LeftButton,
            modifiers=Qt.KeyboardModifier.ShiftModifier, view_scale=1.0,
        ))
        assert results[-1].additive is True

    def test_polygon_closes_on_enter(self, loaded):
        canvas, _frame = loaded
        results = []
        canvas.tool_result.connect(results.append)
        tool = canvas.tools.require("polygon")
        for x, y in ((10, 10), (80, 20), (60, 90)):
            tool.mouse_press(ToolEvent(Point(x, y), Point(0, 0), Qt.MouseButton.LeftButton, view_scale=1.0))
        tool.key_press(Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
        assert isinstance(results[-1].geometry, Polygon)
        assert len(results[-1].geometry) == 3

    def test_polygon_needs_three_points(self, loaded):
        canvas, _frame = loaded
        results = []
        canvas.tool_result.connect(results.append)
        tool = canvas.tools.require("polygon")
        for x, y in ((10, 10), (80, 20)):
            tool.mouse_press(ToolEvent(Point(x, y), Point(0, 0), Qt.MouseButton.LeftButton, view_scale=1.0))
        tool.key_press(Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
        assert results == []

    def test_escape_abandons_a_partial_gesture(self, loaded):
        canvas, _frame = loaded
        results = []
        canvas.tool_result.connect(results.append)
        tool = canvas.tools.require("polygon")
        tool.mouse_press(ToolEvent(Point(1, 1), Point(0, 0), Qt.MouseButton.LeftButton, view_scale=1.0))
        tool.key_press(Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
        tool.key_press(Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
        assert results == []

    def test_switching_tools_clears_partial_state(self, loaded):
        canvas, _frame = loaded
        results = []
        canvas.tool_result.connect(results.append)
        canvas.set_tool("polygon")
        tool = canvas.tools.require("polygon")
        for x, y in ((1, 1), (2, 2), (3, 3)):
            tool.mouse_press(ToolEvent(Point(x, y), Point(0, 0), Qt.MouseButton.LeftButton, view_scale=1.0))
        canvas.set_tool("select")
        tool.key_press(Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
        assert results == []

    def test_registering_a_tool_needs_no_canvas_change(self, loaded):
        """The extension point: a plugin's tool publishes on the same channel."""
        from vidtriage.view.tools import Tool

        class Custom(Tool):
            id = "test.custom"
            title = "Custom"

            def mouse_press(self, event):
                self._emit(Rect(0, 0, 5, 5))
                return True

        canvas, _frame = loaded
        results = []
        canvas.tool_result.connect(results.append)
        canvas.register_tool(Custom())
        canvas.set_tool("test.custom")
        canvas.active_tool.mouse_press(
            ToolEvent(Point(0, 0), Point(0, 0), Qt.MouseButton.LeftButton, view_scale=1.0),
        )
        assert results[-1].tool_id == "test.custom"


class TestZoom:
    def test_zoom_is_clamped(self, loaded):
        canvas, _frame = loaded
        canvas.zoom_to(1e9)
        assert canvas.zoom <= 100
        canvas.zoom_to(1e-9)
        assert canvas.zoom > 0

    def test_zoom_reset_is_one_to_one(self, loaded):
        canvas, _frame = loaded
        canvas.zoom_to(7.0)
        canvas.zoom_reset()
        assert canvas.zoom == pytest.approx(1.0)

    def test_visible_rect_tracks_the_viewport(self, loaded, pump):
        canvas, _frame = loaded
        canvas.fit_to_window()
        pump(20)
        assert canvas.visible_image_rect().width == pytest.approx(640, abs=2)


class TestTheme:
    def test_theme_change_reaches_the_canvas_without_wiring(self, loaded, pump):
        """Widgets subscribe themselves; nothing maintains a list of them."""
        canvas, _frame = loaded
        set_theme("Nord Light")
        pump(30)
        assert canvas.style.theme.name == "Nord Light"
        set_theme("Dark")
        pump(30)
        assert canvas.style.theme.name == "Dark"
