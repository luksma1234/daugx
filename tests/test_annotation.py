"""Tests for the Annotation base class and annotation components."""
import numpy as np
import pytest

from daugx.core.data.annotation import Annotation
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
        assert isinstance(bbox, Annotation)

    def test_image_category_is_annotation(self):
        cat = ImageCategory(class_id=0)
        assert isinstance(cat, Annotation)
        assert isinstance(cat, Component)

    def test_image_polygon_is_annotation(self):
        poly = ImagePolygon(
            np.array([[0, 0], [5, 0], [5, 5]]),
        )
        assert isinstance(poly, Annotation)
        assert isinstance(poly, Component)

    def test_image_keypoint_is_annotation(self):
        kp = ImageKeyPoint(x=5.0, y=10.0)
        assert isinstance(kp, Annotation)
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
