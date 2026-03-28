"""Tests for the data loading architecture.

Covers: Component, Image, Label, BoundingBox, Polygon, KeyPoint,
Schema, Sample, DataPackage, and reworked Dataset.
"""
import os
import tempfile

import cv2
import numpy as np
import pytest

from daugx.core.data.component import Component, ComponentState
from daugx.core.data.components.label import Label
from daugx.core.data.components.bounding_box import BoundingBox
from daugx.core.data.components.polygon import Polygon
from daugx.core.data.components.keypoint import KeyPoint
from daugx.core.data.components.image import Image
from daugx.core.data.schema import Schema
from daugx.core.data.sample import Sample
from daugx.core.data.data_package import DataPackage
from daugx.core.dataset import Dataset
from daugx.errors import SchemaValidationError


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def tmp_image_path():
    """Create a temporary 4x4 RGB image on disk."""
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[1, 1] = [255, 0, 0]
    with tempfile.NamedTemporaryFile(
        suffix=".png", delete=False,
    ) as f:
        cv2.imwrite(f.name, img)
        yield f.name
    os.unlink(f.name)


# -------------------------------------------------------------------
# Component base
# -------------------------------------------------------------------

class TestComponentABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Component()


# -------------------------------------------------------------------
# Label
# -------------------------------------------------------------------

class TestLabel:
    def test_construction(self):
        lbl = Label(class_id=3, name="dog")
        assert lbl.class_id == 3
        assert lbl.name == "dog"

    def test_name_optional(self):
        lbl = Label(class_id=0)
        assert lbl.name is None

    def test_always_materialized(self):
        lbl = Label(class_id=1)
        assert lbl.state == ComponentState.MATERIALIZED
        assert lbl.is_materialized is True

    def test_materialize_noop(self):
        lbl = Label(class_id=1)
        lbl.materialize()
        assert lbl.is_materialized is True


# -------------------------------------------------------------------
# BoundingBox
# -------------------------------------------------------------------

class TestBoundingBox:
    def test_construction_2x2(self):
        pts = np.array([[10, 20], [100, 200]])
        bb = BoundingBox(points=pts)
        assert bb.x_min == 10
        assert bb.y_min == 20
        assert bb.x_max == 100
        assert bb.y_max == 200

    def test_properties(self):
        bb = BoundingBox(
            points=np.array([[0, 0], [10, 20]]),
        )
        assert bb.width == 10
        assert bb.height == 20
        assert bb.area == 200
        np.testing.assert_array_equal(
            bb.center, [5.0, 10.0],
        )

    def test_always_materialized(self):
        bb = BoundingBox(
            points=np.array([[0, 0], [1, 1]]),
        )
        assert bb.state == ComponentState.MATERIALIZED
        assert bb.is_materialized is True

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            BoundingBox(points=np.array([1, 2, 3]))

    def test_points_shape(self):
        bb = BoundingBox(
            points=np.array([[5, 10], [50, 100]]),
        )
        assert bb.points.shape == (2, 2)


# -------------------------------------------------------------------
# Polygon
# -------------------------------------------------------------------

class TestPolygon:
    def test_construction(self):
        pts = np.array([[0, 0], [10, 0], [10, 10]])
        poly = Polygon(points=pts)
        assert poly.points.shape == (3, 2)

    def test_area_triangle(self):
        pts = np.array(
            [[0, 0], [10, 0], [0, 10]], dtype=float,
        )
        poly = Polygon(points=pts)
        assert poly.area == pytest.approx(50.0)

    def test_center(self):
        pts = np.array(
            [[0, 0], [6, 0], [6, 6], [0, 6]], dtype=float,
        )
        poly = Polygon(points=pts)
        np.testing.assert_array_almost_equal(
            poly.center, [3.0, 3.0],
        )

    def test_always_materialized(self):
        pts = np.array([[0, 0], [1, 0], [1, 1]])
        poly = Polygon(points=pts)
        assert poly.is_materialized is True

    def test_rejects_fewer_than_3_points(self):
        with pytest.raises(ValueError):
            Polygon(points=np.array([[0, 0], [1, 1]]))


# -------------------------------------------------------------------
# KeyPoint
# -------------------------------------------------------------------

class TestKeyPoint:
    def test_construction(self):
        kp = KeyPoint(x=5.0, y=10.0)
        assert kp.x == 5.0
        assert kp.y == 10.0
        assert kp.visibility is None

    def test_visibility(self):
        kp = KeyPoint(x=1.0, y=2.0, visibility=2)
        assert kp.visibility == 2

    def test_point_array(self):
        kp = KeyPoint(x=3.0, y=7.0)
        np.testing.assert_array_equal(
            kp.point, [3.0, 7.0],
        )

    def test_always_materialized(self):
        kp = KeyPoint(x=0.0, y=0.0)
        assert kp.is_materialized is True


# -------------------------------------------------------------------
# Image
# -------------------------------------------------------------------

class TestImage:
    def test_preloaded_state(self, tmp_image_path):
        img = Image(path=tmp_image_path)
        assert img.state == ComponentState.PRELOADED
        assert img.is_materialized is False
        assert img.path == tmp_image_path

    def test_data_raises_before_materialize(
        self, tmp_image_path,
    ):
        img = Image(path=tmp_image_path)
        with pytest.raises(RuntimeError):
            _ = img.data

    def test_materialize_loads_pixels(
        self, tmp_image_path,
    ):
        img = Image(path=tmp_image_path)
        img.materialize()
        assert img.is_materialized is True
        assert img.state == ComponentState.MATERIALIZED
        assert isinstance(img.data, np.ndarray)
        assert img.data.shape == (4, 4, 3)

    def test_materialize_idempotent(
        self, tmp_image_path,
    ):
        img = Image(path=tmp_image_path)
        img.materialize()
        data1 = img.data
        img.materialize()
        assert img.data is data1

    def test_format_hint(self, tmp_image_path):
        img = Image(
            path=tmp_image_path, format_hint="png",
        )
        assert img.format_hint == "png"

    def test_invalid_path_raises_on_materialize(self):
        img = Image(path="/nonexistent/image.png")
        with pytest.raises(FileNotFoundError):
            img.materialize()


# -------------------------------------------------------------------
# Schema
# -------------------------------------------------------------------

class TestSchema:
    def test_singular_key(self):
        s = Schema({"image": Image})
        assert "image" in s.singular_keys
        assert "image" not in s.plural_keys

    def test_plural_key(self):
        s = Schema({"objects": [BoundingBox, Label]})
        assert "objects" in s.plural_keys
        assert "objects" not in s.singular_keys

    def test_keys(self):
        s = Schema({
            "image": Image,
            "objects": [BoundingBox, Label],
        })
        assert set(s.keys) == {"image", "objects"}

    def test_definition_property(self):
        defn = {"image": Image}
        s = Schema(defn)
        assert s.definition == defn

    def test_validate_valid_sample(self, tmp_image_path):
        s = Schema({
            "image": Image,
            "objects": [BoundingBox, Label],
        })
        sample = Sample(
            image=Image(path=tmp_image_path),
            objects=[
                (
                    BoundingBox(
                        np.array([[0, 0], [10, 10]]),
                    ),
                    Label(0, "cat"),
                ),
            ],
        )
        s.validate(sample)  # should not raise

    def test_validate_missing_key(self, tmp_image_path):
        s = Schema({
            "image": Image,
            "objects": [BoundingBox, Label],
        })
        sample = Sample(
            image=Image(path=tmp_image_path),
        )
        with pytest.raises(SchemaValidationError):
            s.validate(sample)

    def test_validate_wrong_singular_type(
        self, tmp_image_path,
    ):
        s = Schema({"image": Image})
        sample = Sample(image=Label(0))
        with pytest.raises(SchemaValidationError):
            s.validate(sample)

    def test_validate_wrong_tuple_type(
        self, tmp_image_path,
    ):
        s = Schema({"objects": [BoundingBox, Label]})
        sample = Sample(
            objects=[
                (Label(0), Label(1)),
            ],
        )
        with pytest.raises(SchemaValidationError):
            s.validate(sample)

    def test_validate_empty_plural_list_ok(self):
        s = Schema({"objects": [BoundingBox, Label]})
        sample = Sample(objects=[])
        s.validate(sample)  # empty list is valid

    def test_validate_extra_keys_ok(
        self, tmp_image_path,
    ):
        s = Schema({"image": Image})
        sample = Sample(
            image=Image(path=tmp_image_path),
            extra=Label(0),
        )
        s.validate(sample)  # extra keys allowed

    def test_validate_plural_not_list(self):
        s = Schema({"objects": [BoundingBox]})
        sample = Sample(
            objects=BoundingBox(
                np.array([[0, 0], [1, 1]]),
            ),
        )
        with pytest.raises(SchemaValidationError):
            s.validate(sample)


# -------------------------------------------------------------------
# Sample
# -------------------------------------------------------------------

class TestSample:
    def test_getitem(self, tmp_image_path):
        img = Image(path=tmp_image_path)
        sample = Sample(image=img)
        assert sample["image"] is img

    def test_contains(self, tmp_image_path):
        sample = Sample(
            image=Image(path=tmp_image_path),
        )
        assert "image" in sample
        assert "missing" not in sample

    def test_keys(self, tmp_image_path):
        sample = Sample(
            image=Image(path=tmp_image_path),
            label=Label(0),
        )
        assert set(sample.keys) == {"image", "label"}

    def test_is_materialized_false_when_preloaded(
        self, tmp_image_path,
    ):
        sample = Sample(
            image=Image(path=tmp_image_path),
        )
        assert sample.is_materialized is False

    def test_is_materialized_true_lightweight_only(self):
        sample = Sample(label=Label(0))
        assert sample.is_materialized is True

    def test_materialize_returns_data_package(
        self, tmp_image_path,
    ):
        sample = Sample(
            image=Image(path=tmp_image_path),
            label=Label(0, "cat"),
        )
        pkg = sample.materialize()
        assert isinstance(pkg, DataPackage)

    def test_materialize_loads_image(
        self, tmp_image_path,
    ):
        sample = Sample(
            image=Image(path=tmp_image_path),
        )
        pkg = sample.materialize()
        assert isinstance(pkg["image"].data, np.ndarray)

    def test_materialize_with_plural(
        self, tmp_image_path,
    ):
        sample = Sample(
            image=Image(path=tmp_image_path),
            objects=[
                (
                    BoundingBox(
                        np.array([[0, 0], [5, 5]]),
                    ),
                    Label(1, "dog"),
                ),
            ],
        )
        pkg = sample.materialize()
        assert len(pkg["objects"]) == 1
        assert isinstance(pkg["objects"][0][0], BoundingBox)


# -------------------------------------------------------------------
# DataPackage
# -------------------------------------------------------------------

class TestDataPackage:
    def test_getitem(self):
        pkg = DataPackage({"label": Label(0)})
        assert isinstance(pkg["label"], Label)

    def test_contains(self):
        pkg = DataPackage({"label": Label(0)})
        assert "label" in pkg
        assert "missing" not in pkg

    def test_keys(self):
        pkg = DataPackage({
            "a": Label(0), "b": Label(1),
        })
        assert set(pkg.keys) == {"a", "b"}

    def test_get_default(self):
        pkg = DataPackage({})
        assert pkg.get("missing", 42) == 42


# -------------------------------------------------------------------
# Dataset (reworked)
# -------------------------------------------------------------------

class TestDataset:
    def test_construction(self, tmp_image_path):
        schema = Schema({"image": Image})
        samples = [
            Sample(image=Image(path=tmp_image_path)),
        ]
        ds = Dataset(
            schema=schema, samples=samples, name="Test",
        )
        assert ds.name == "Test"
        assert len(ds) == 1

    def test_getitem(self, tmp_image_path):
        schema = Schema({"image": Image})
        s = Sample(image=Image(path=tmp_image_path))
        ds = Dataset(schema=schema, samples=[s])
        assert ds[0] is s

    def test_schema_property(self):
        schema = Schema({"label": Label})
        ds = Dataset(
            schema=schema,
            samples=[Sample(label=Label(0))],
        )
        assert ds.schema is schema

    def test_default_name(self):
        schema = Schema({"label": Label})
        ds = Dataset(
            schema=schema,
            samples=[Sample(label=Label(0))],
        )
        assert ds.name == "Dataset"

    def test_validation_on_construction(
        self, tmp_image_path,
    ):
        schema = Schema({
            "image": Image,
            "label": Label,
        })
        bad_sample = Sample(
            image=Image(path=tmp_image_path),
        )
        with pytest.raises(SchemaValidationError):
            Dataset(
                schema=schema, samples=[bad_sample],
            )

    def test_empty_samples(self):
        schema = Schema({"image": Image})
        ds = Dataset(schema=schema, samples=[])
        assert len(ds) == 0
