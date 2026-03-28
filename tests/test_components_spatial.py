"""Tests for spatial methods on BoundingBox, Polygon, KeyPoint.

All spatial methods return new instances (immutable).
"""
import numpy as np
import pytest

from daugx.core.data.components.bounding_box import BoundingBox
from daugx.core.data.components.polygon import Polygon
from daugx.core.data.components.keypoint import KeyPoint


# -------------------------------------------------------------------
# BoundingBox spatial methods
# -------------------------------------------------------------------

class TestBoundingBoxShift:
    def test_shift_positive(self):
        bb = BoundingBox(np.array([[10, 20], [50, 60]]))
        shifted = bb.shift(5, 10)
        assert shifted.x_min == 15
        assert shifted.y_min == 30
        assert shifted.x_max == 55
        assert shifted.y_max == 70

    def test_shift_negative(self):
        bb = BoundingBox(np.array([[10, 20], [50, 60]]))
        shifted = bb.shift(-5, -10)
        assert shifted.x_min == 5
        assert shifted.y_min == 10

    def test_shift_returns_new_instance(self):
        bb = BoundingBox(np.array([[10, 20], [50, 60]]))
        shifted = bb.shift(5, 5)
        assert shifted is not bb
        assert bb.x_min == 10  # original unchanged


class TestBoundingBoxScale:
    def test_scale_up(self):
        bb = BoundingBox(np.array([[10, 20], [30, 40]]))
        scaled = bb.scale(2.0, 2.0)
        assert scaled.x_min == pytest.approx(20.0)
        assert scaled.y_min == pytest.approx(40.0)
        assert scaled.x_max == pytest.approx(60.0)
        assert scaled.y_max == pytest.approx(80.0)

    def test_scale_down(self):
        bb = BoundingBox(np.array([[10, 20], [30, 40]]))
        scaled = bb.scale(0.5, 0.5)
        assert scaled.x_min == pytest.approx(5.0)
        assert scaled.y_min == pytest.approx(10.0)

    def test_scale_returns_new_instance(self):
        bb = BoundingBox(np.array([[10, 20], [30, 40]]))
        scaled = bb.scale(2.0, 2.0)
        assert scaled is not bb
        assert bb.x_min == 10


class TestBoundingBoxRotate:
    def test_rotate_360_returns_same(self):
        bb = BoundingBox(np.array([[10, 10], [20, 20]]))
        center = np.array([15.0, 15.0])
        rotated = bb.rotate(360.0, center)
        assert rotated.x_min == pytest.approx(10.0, abs=1e-6)
        assert rotated.y_min == pytest.approx(10.0, abs=1e-6)
        assert rotated.x_max == pytest.approx(20.0, abs=1e-6)
        assert rotated.y_max == pytest.approx(20.0, abs=1e-6)

    def test_rotate_90_around_center(self):
        bb = BoundingBox(np.array([[0, 0], [10, 20]]))
        center = np.array([5.0, 10.0])
        rotated = bb.rotate(90.0, center)
        assert rotated.width == pytest.approx(20.0, abs=1e-6)
        assert rotated.height == pytest.approx(10.0, abs=1e-6)

    def test_rotate_returns_new_instance(self):
        bb = BoundingBox(np.array([[0, 0], [10, 10]]))
        rotated = bb.rotate(45.0, np.array([5.0, 5.0]))
        assert rotated is not bb
        assert bb.x_min == 0


class TestBoundingBoxClip:
    def test_clip_within_bounds(self):
        bb = BoundingBox(np.array([[-5, -5], [15, 15]]))
        clipped = bb.clip(0, 0, 10, 10)
        assert clipped.x_min == 0
        assert clipped.y_min == 0
        assert clipped.x_max == 10
        assert clipped.y_max == 10

    def test_clip_no_change_when_inside(self):
        bb = BoundingBox(np.array([[2, 2], [8, 8]]))
        clipped = bb.clip(0, 0, 10, 10)
        assert clipped.x_min == 2
        assert clipped.x_max == 8

    def test_clip_returns_new_instance(self):
        bb = BoundingBox(np.array([[0, 0], [10, 10]]))
        clipped = bb.clip(0, 0, 10, 10)
        assert clipped is not bb


class TestBoundingBoxIsValid:
    def test_valid_box(self):
        bb = BoundingBox(np.array([[0, 0], [10, 10]]))
        assert bb.is_valid() is True

    def test_degenerate_box(self):
        bb = BoundingBox(np.array([[5, 5], [5, 10]]))
        assert bb.is_valid() is False

    def test_inverted_box(self):
        bb = BoundingBox(np.array([[10, 10], [5, 5]]))
        assert bb.is_valid() is False

    def test_min_area(self):
        bb = BoundingBox(np.array([[0, 0], [1, 1]]))
        assert bb.is_valid(min_area=0) is True
        assert bb.is_valid(min_area=2) is False


# -------------------------------------------------------------------
# Polygon spatial methods
# -------------------------------------------------------------------

class TestPolygonShift:
    def test_shift(self):
        poly = Polygon(
            np.array([[0, 0], [4, 0], [4, 3], [0, 3]]),
        )
        shifted = poly.shift(10, 20)
        expected = np.array(
            [[10, 20], [14, 20], [14, 23], [10, 23]],
        )
        np.testing.assert_allclose(shifted.points, expected)

    def test_shift_returns_new_instance(self):
        poly = Polygon(
            np.array([[0, 0], [4, 0], [4, 3], [0, 3]]),
        )
        shifted = poly.shift(1, 1)
        assert shifted is not poly
        assert poly.points[0, 0] == 0


class TestPolygonScale:
    def test_scale(self):
        poly = Polygon(
            np.array([[0, 0], [4, 0], [4, 3], [0, 3]]),
        )
        scaled = poly.scale(2.0, 3.0)
        expected = np.array(
            [[0, 0], [8, 0], [8, 9], [0, 9]],
        )
        np.testing.assert_allclose(scaled.points, expected)


class TestPolygonRotate:
    def test_rotate_360(self):
        poly = Polygon(
            np.array([[0, 0], [4, 0], [4, 3], [0, 3]]),
        )
        center = np.array([2.0, 1.5])
        rotated = poly.rotate(360.0, center)
        np.testing.assert_allclose(
            rotated.points, poly.points, atol=1e-6,
        )

    def test_rotate_returns_new_instance(self):
        poly = Polygon(
            np.array([[0, 0], [4, 0], [4, 3], [0, 3]]),
        )
        rotated = poly.rotate(45.0, np.array([2.0, 1.5]))
        assert rotated is not poly


class TestPolygonClip:
    def test_clip(self):
        poly = Polygon(
            np.array([[-1, -1], [5, -1], [5, 5], [-1, 5]]),
        )
        clipped = poly.clip(0, 0, 4, 4)
        assert clipped.points[:, 0].min() >= 0
        assert clipped.points[:, 1].min() >= 0
        assert clipped.points[:, 0].max() <= 4
        assert clipped.points[:, 1].max() <= 4


class TestPolygonIsValid:
    def test_valid_polygon(self):
        poly = Polygon(
            np.array([[0, 0], [4, 0], [4, 3], [0, 3]]),
        )
        assert poly.is_valid() is True

    def test_degenerate_polygon(self):
        poly = Polygon(
            np.array([[0, 0], [0, 0], [0, 0]]),
        )
        assert poly.is_valid() is False

    def test_min_area(self):
        poly = Polygon(
            np.array([[0, 0], [1, 0], [0, 1]]),
        )
        assert poly.is_valid(min_area=0) is True
        assert poly.is_valid(min_area=1) is False


# -------------------------------------------------------------------
# KeyPoint spatial methods
# -------------------------------------------------------------------

class TestKeyPointShift:
    def test_shift(self):
        kp = KeyPoint(10.0, 20.0, visibility=2)
        shifted = kp.shift(5.0, -3.0)
        assert shifted.x == 15.0
        assert shifted.y == 17.0
        assert shifted.visibility == 2

    def test_shift_returns_new_instance(self):
        kp = KeyPoint(10.0, 20.0)
        shifted = kp.shift(1, 1)
        assert shifted is not kp
        assert kp.x == 10.0


class TestKeyPointScale:
    def test_scale(self):
        kp = KeyPoint(10.0, 20.0)
        scaled = kp.scale(2.0, 0.5)
        assert scaled.x == 20.0
        assert scaled.y == 10.0


class TestKeyPointRotate:
    def test_rotate_360(self):
        kp = KeyPoint(10.0, 20.0)
        center = np.array([10.0, 20.0])
        rotated = kp.rotate(360.0, center)
        assert rotated.x == pytest.approx(10.0, abs=1e-6)
        assert rotated.y == pytest.approx(20.0, abs=1e-6)

    def test_rotate_preserves_visibility(self):
        kp = KeyPoint(10.0, 20.0, visibility=1)
        rotated = kp.rotate(45.0, np.array([0.0, 0.0]))
        assert rotated.visibility == 1


class TestKeyPointClip:
    def test_clip(self):
        kp = KeyPoint(-5.0, 15.0)
        clipped = kp.clip(0, 0, 10, 10)
        assert clipped.x == 0.0
        assert clipped.y == 10.0


class TestKeyPointIsValid:
    def test_valid_within_bounds(self):
        kp = KeyPoint(5.0, 5.0)
        assert kp.is_valid(0, 0, 10, 10) is True

    def test_out_of_bounds(self):
        kp = KeyPoint(-1.0, 5.0)
        assert kp.is_valid(0, 0, 10, 10) is False
