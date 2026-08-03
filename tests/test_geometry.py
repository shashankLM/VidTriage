"""Geometry: the coordinate space everything else is expressed in."""

from __future__ import annotations

import numpy as np
import pytest

from label_kit.core.geometry import Mask, Point, Polygon, Rect, Size, ViewTransform


class TestRect:
    def test_inverted_corners_are_normalised(self):
        """A rubber-band drag up-and-left must not collapse the rect."""
        rect = Rect(100, 50, 20, 10)
        assert rect.as_xyxy() == (20, 10, 100, 50)
        assert rect.width == 80 and rect.height == 40

    def test_partially_inverted(self):
        assert Rect(100, 10, 20, 50).as_xyxy() == (20, 10, 100, 50)

    def test_center_and_area(self):
        rect = Rect(0, 0, 10, 20)
        assert rect.center == Point(5, 10)
        assert rect.area == 200

    def test_iou(self):
        a, b = Rect(0, 0, 10, 10), Rect(5, 0, 15, 10)
        assert a.iou(b) == pytest.approx(50 / 150)
        assert a.iou(a) == pytest.approx(1.0)
        assert a.iou(Rect(100, 100, 110, 110)) == 0.0

    def test_clamped_to_image(self):
        assert Rect(-10, -10, 50, 50).clamped_to(Size(30, 30)).as_xyxy() == (0, 0, 30, 30)
        assert Rect(100, 100, 110, 110).clamped_to(Size(30, 30)).is_empty

    @pytest.mark.parametrize(
        "rect",
        [Rect(5.2, 5.2, 5.3, 5.3), Rect(0, 0, 0.4, 0.4), Rect(9.9, 9.9, 10.5, 10.5)],
    )
    def test_pixel_slice_is_never_degenerate(self, rect):
        """A sub-pixel region must still yield a croppable patch, not an empty array."""
        rows, cols = rect.to_pixel_slice(Size(10, 10))
        assert rows.stop > rows.start and cols.stop > cols.start
        assert rows.start >= 0 and rows.stop <= 10
        assert cols.start >= 0 and cols.stop <= 10
        assert np.zeros((10, 10, 3), np.uint8)[rows, cols].size > 0

    def test_normalised_is_yolo_convention(self):
        cx, cy, w, h = Rect(0, 0, 50, 100).normalised(Size(100, 200))
        assert (cx, cy, w, h) == (0.25, 0.25, 0.5, 0.5)


class TestPolygon:
    def test_shoelace_area(self):
        assert Polygon.from_iterable([(0, 0), (10, 0), (10, 10), (0, 10)]).area == 100

    def test_area_is_winding_independent(self):
        clockwise = Polygon.from_iterable([(0, 0), (0, 10), (10, 10), (10, 0)])
        assert clockwise.area == 100

    def test_needs_three_points(self):
        assert not Polygon.from_iterable([(0, 0), (1, 1)]).is_valid
        assert Polygon.from_iterable([(0, 0), (1, 1), (2, 0)]).is_valid

    def test_flat_list_roundtrip(self):
        polygon = Polygon.from_iterable([(1, 2), (3, 4), (5, 6)])
        assert polygon.as_flat_list() == [1, 2, 3, 4, 5, 6]


class TestMask:
    def test_cropped_to_tight_bounds(self):
        full = np.zeros((10, 10), bool)
        full[3:6, 2:8] = True
        mask = Mask.from_full_frame(full)
        assert mask.bounds.as_xyxy() == (2, 3, 8, 6)
        assert mask.data.shape == (3, 6)
        assert mask.pixel_count == 18

    def test_roundtrips_through_full_frame(self):
        full = np.zeros((10, 10), bool)
        full[3:6, 2:8] = True
        restored = Mask.from_full_frame(full).to_full_frame(Size(10, 10))
        assert np.array_equal(restored, full)

    def test_empty_mask(self):
        mask = Mask.from_full_frame(np.zeros((5, 5), bool))
        assert mask.is_empty
        assert not mask.to_full_frame(Size(5, 5)).any()


class TestViewTransform:
    def test_fit_letterboxes_and_centres(self):
        transform = ViewTransform.fit(Size(1000, 500), Size(400, 400))
        assert transform.scale == pytest.approx(0.4)
        assert transform.offset_y == pytest.approx(100)
        assert transform.offset_x == pytest.approx(0)

    @pytest.mark.parametrize("scale", [0.01, 0.5, 1.0, 7.3, 100.0])
    def test_roundtrip_is_exact(self, scale):
        transform = ViewTransform(scale=scale, offset_x=13.5, offset_y=-7.25)
        for point in (Point(0, 0), Point(0.125, 0.875), Point(1919, 1079)):
            back = transform.widget_to_image(transform.image_to_widget(point))
            assert back.x == pytest.approx(point.x, abs=1e-9)
            assert back.y == pytest.approx(point.y, abs=1e-9)

    def test_zoom_keeps_the_anchor_pixel_fixed(self):
        """Wheel-zoom must not slide the image out from under the cursor."""
        transform = ViewTransform(scale=1.0, offset_x=10, offset_y=20)
        anchor = Point(300, 200)
        before = transform.widget_to_image(anchor)
        zoomed = transform.zoomed(2.5, anchor)
        after = zoomed.widget_to_image(anchor)
        assert after.x == pytest.approx(before.x)
        assert after.y == pytest.approx(before.y)

    def test_widget_length_to_image_scales_inversely(self):
        assert ViewTransform(scale=4.0).widget_length_to_image(8.0) == 2.0

    def test_rejects_non_positive_scale(self):
        with pytest.raises(ValueError):
            ViewTransform(scale=0.0)
