"""Tests for shared annotation component behaviour."""
import numpy as np
import pytest

from daugx.core.data.component import Component, ComponentState
from daugx.core.data.components.bounding_box import (
    ImageBoundingBox,
)
from daugx.core.data.components.keypoint import ImageKeyPoint
from daugx.core.data.components.label import ImageCategory
from daugx.core.data.components.polygon import ImagePolygon


class TestAnnotationBase:
    def test_subclasses_are_components(self):
        bbox = ImageBoundingBox(
            np.array([[0, 0], [10, 10]]),
        )
        assert isinstance(bbox, Component)
        assert hasattr(bbox, 'applies_to')

    def test_image_category_is_annotation(self):
        cat = ImageCategory(class_id=0)
        assert hasattr(cat, 'applies_to')
        assert isinstance(cat, Component)

    def test_image_polygon_is_annotation(self):
        poly = ImagePolygon(
            np.array([[0, 0], [5, 0], [5, 5]]),
        )
        assert hasattr(poly, 'applies_to')
        assert isinstance(poly, Component)

    def test_image_keypoint_is_annotation(self):
        kp = ImageKeyPoint(x=5.0, y=10.0)
        assert hasattr(kp, 'applies_to')
        assert isinstance(kp, Component)


class TestAnnotationTarget:
    def test_target_default_none(self):
        bbox = ImageBoundingBox(
            np.array([[0, 0], [10, 10]]),
        )
        assert bbox.target is None

    def test_target_set(self):
        bbox = ImageBoundingBox(
            np.array([[0, 0], [10, 10]]),
            target="cam_left",
        )
        assert bbox.target == "cam_left"

    def test_target_on_category(self):
        cat = ImageCategory(class_id=1, target="main")
        assert cat.target == "main"

    def test_target_on_polygon(self):
        poly = ImagePolygon(
            np.array([[0, 0], [5, 0], [5, 5]]),
            target="drone_cam",
        )
        assert poly.target == "drone_cam"

    def test_target_on_keypoint(self):
        kp = ImageKeyPoint(x=1.0, y=2.0, target="front")
        assert kp.target == "front"


class TestAnnotationLifecycle:
    def test_always_materialized(self):
        bbox = ImageBoundingBox(
            np.array([[0, 0], [10, 10]]),
        )
        assert bbox.is_materialized is True
        assert bbox.state == ComponentState.MATERIALIZED

    def test_materialize_noop(self):
        bbox = ImageBoundingBox(
            np.array([[0, 0], [10, 10]]),
        )
        bbox.materialize()
        assert bbox.is_materialized is True

    def test_category_always_materialized(self):
        cat = ImageCategory(class_id=0)
        assert cat.is_materialized is True
        assert cat.state == ComponentState.MATERIALIZED


class TestAnnotationName:
    def test_name_default_none(self):
        bbox = ImageBoundingBox(
            np.array([[0, 0], [10, 10]]),
        )
        assert bbox.name is None

    def test_name_set(self):
        bbox = ImageBoundingBox(
            np.array([[0, 0], [10, 10]]),
            name="obj_0",
        )
        assert bbox.name == "obj_0"

    def test_name_and_target_independent(self):
        bbox = ImageBoundingBox(
            np.array([[0, 0], [10, 10]]),
            target="cam",
            name="box_1",
        )
        assert bbox.target == "cam"
        assert bbox.name == "box_1"


class TestBoundingBoxType:
    def test_default_xyxy(self):
        bbox = ImageBoundingBox(
            np.array([[10, 20], [30, 40]]),
        )
        np.testing.assert_array_equal(
            bbox.points, [[10, 20], [30, 40]],
        )

    def test_explicit_xyxy(self):
        bbox = ImageBoundingBox(
            np.array([[10, 20], [30, 40]]),
            box_type="XYXY",
        )
        np.testing.assert_array_equal(
            bbox.points, [[10, 20], [30, 40]],
        )

    def test_xywh(self):
        bbox = ImageBoundingBox(
            np.array([[10, 20], [30, 40]]),
            box_type="XYWH",
        )
        np.testing.assert_array_equal(
            bbox.points, [[10, 20], [40, 60]],
        )

    def test_cxcywh(self):
        bbox = ImageBoundingBox(
            np.array([[50, 50], [20, 30]]),
            box_type="CXCYWH",
        )
        np.testing.assert_array_equal(
            bbox.points, [[40, 35], [60, 65]],
        )

    def test_yxyx(self):
        bbox = ImageBoundingBox(
            np.array([[10, 20], [30, 40]]),
            box_type="YXYX",
        )
        np.testing.assert_array_equal(
            bbox.points, [[20, 10], [40, 30]],
        )

    def test_invalid_box_type(self):
        with pytest.raises(ValueError, match="box_type"):
            ImageBoundingBox(
                np.array([[0, 0], [10, 10]]),
                box_type="INVALID",
            )

    def test_case_insensitive(self):
        bbox = ImageBoundingBox(
            np.array([[10, 20], [30, 40]]),
            box_type="xywh",
        )
        np.testing.assert_array_equal(
            bbox.points, [[10, 20], [40, 60]],
        )

    def test_case_insensitive_mixed(self):
        bbox = ImageBoundingBox(
            np.array([[50, 50], [20, 30]]),
            box_type="CxCyWh",
        )
        np.testing.assert_array_equal(
            bbox.points, [[40, 35], [60, 65]],
        )


class TestBoundingBoxFlatInput:
    """Flat [x, y, x, y] input is reshaped to (2, 2) internally."""

    def test_flat_xyxy(self):
        bbox = ImageBoundingBox(
            np.array([10, 20, 30, 40]),
            box_type="XYXY",
        )
        np.testing.assert_array_equal(
            bbox.points, [[10, 20], [30, 40]],
        )

    def test_flat_xywh(self):
        bbox = ImageBoundingBox(
            np.array([10, 20, 30, 40]),
            box_type="XYWH",
        )
        np.testing.assert_array_equal(
            bbox.points, [[10, 20], [40, 60]],
        )

    def test_flat_cxcywh(self):
        bbox = ImageBoundingBox(
            np.array([50, 50, 20, 30]),
            box_type="CXCYWH",
        )
        np.testing.assert_array_equal(
            bbox.points, [[40, 35], [60, 65]],
        )

    def test_flat_yxyx(self):
        bbox = ImageBoundingBox(
            np.array([10, 20, 30, 40]),
            box_type="YXYX",
        )
        np.testing.assert_array_equal(
            bbox.points, [[20, 10], [40, 30]],
        )

    def test_flat_list_input(self):
        bbox = ImageBoundingBox([5, 10, 25, 35])
        np.testing.assert_array_equal(
            bbox.points, [[5, 10], [25, 35]],
        )

    def test_flat_wrong_length_raises(self):
        with pytest.raises(ValueError):
            ImageBoundingBox(np.array([10, 20, 30]))

    def test_flat_preserves_metadata(self):
        bbox = ImageBoundingBox(
            [0, 0, 10, 10],
            class_id=3,
            class_name="dog",
            target="cam",
            name="b0",
        )
        assert bbox.class_id == 3
        assert bbox.class_name == "dog"
        assert bbox.target == "cam"
        assert bbox.name == "b0"


class TestPolygonFlatInput:
    """Flat [x0, y0, x1, y1, ...] input is reshaped to (n, 2)."""

    def test_flat_numpy_input(self):
        poly = ImagePolygon(np.array([0, 0, 5, 0, 5, 5]))
        np.testing.assert_array_equal(
            poly.points, [[0, 0], [5, 0], [5, 5]],
        )

    def test_flat_list_input(self):
        poly = ImagePolygon([0, 0, 10, 0, 10, 10, 0, 10])
        np.testing.assert_array_equal(
            poly.points, [[0, 0], [10, 0], [10, 10], [0, 10]],
        )

    def test_flat_odd_length_raises(self):
        with pytest.raises(ValueError):
            ImagePolygon(np.array([0, 0, 5, 0, 5]))

    def test_flat_too_few_points_raises(self):
        with pytest.raises(ValueError):
            ImagePolygon(np.array([0, 0, 5, 0]))

    def test_flat_preserves_metadata(self):
        poly = ImagePolygon(
            [0, 0, 10, 0, 10, 10],
            class_id=2,
            class_name="car",
            target="img",
            name="p0",
        )
        assert poly.class_id == 2
        assert poly.class_name == "car"
        assert poly.target == "img"
        assert poly.name == "p0"
