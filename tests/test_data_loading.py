"""Tests for the data loading architecture.

Covers: Component, Image, ImageCategory, ImageBoundingBox,
ImagePolygon, ImageKeyPoint, Text, Constant,
Sample, and Dataset.
"""
import os
import tempfile

import cv2
import numpy as np
import pytest

from daugx.core.data.component import Component, ComponentState
from daugx.core.data.components.bounding_box import (
    ImageBoundingBox,
)
from daugx.core.data.components.constant import Constant
from daugx.core.data.components.image import Image
from daugx.core.data.components.keypoint import ImageKeyPoint
from daugx.core.data.components.label import ImageCategory
from daugx.core.data.components.polygon import ImagePolygon
from daugx.core.data.components.text import Text
from daugx.core.data.sample import Sample
from daugx.core.dataset import Dataset


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
# ImageCategory (was Label)
# -------------------------------------------------------------------

class TestImageCategory:
    def test_construction(self):
        cat = ImageCategory(class_id=3, class_name="dog")
        assert cat.class_id == 3
        assert cat.class_name == "dog"

    def test_class_name_optional(self):
        cat = ImageCategory(class_id=0)
        assert cat.class_name is None

    def test_component_name_default_none(self):
        cat = ImageCategory(class_id=0)
        assert cat.name is None

    def test_component_name_set(self):
        cat = ImageCategory(class_id=0, name="primary")
        assert cat.name == "primary"

    def test_target_default_none(self):
        cat = ImageCategory(class_id=0)
        assert cat.target is None

    def test_target_set(self):
        cat = ImageCategory(class_id=0, target="cam")
        assert cat.target == "cam"

    def test_always_materialized(self):
        cat = ImageCategory(class_id=1)
        assert cat.state == ComponentState.MATERIALIZED
        assert cat.is_materialized is True

    def test_materialize_noop(self):
        cat = ImageCategory(class_id=1)
        cat.materialize()
        assert cat.is_materialized is True

    def test_is_annotation(self):
        cat = ImageCategory(class_id=0)
        assert hasattr(cat, 'applies_to')


# -------------------------------------------------------------------
# ImageBoundingBox (was BoundingBox)
# -------------------------------------------------------------------

class TestImageBoundingBox:
    def test_construction_2x2(self):
        pts = np.array([[10, 20], [100, 200]])
        bb = ImageBoundingBox(points=pts)
        assert bb.x_min == 10
        assert bb.y_min == 20
        assert bb.x_max == 100
        assert bb.y_max == 200

    def test_properties(self):
        bb = ImageBoundingBox(
            points=np.array([[0, 0], [10, 20]]),
        )
        assert bb.width == 10
        assert bb.height == 20
        assert bb.area == 200
        np.testing.assert_array_equal(
            bb.center, [5.0, 10.0],
        )

    def test_class_id_and_name(self):
        bb = ImageBoundingBox(
            np.array([[0, 0], [10, 10]]),
            class_id=2,
            class_name="car",
        )
        assert bb.class_id == 2
        assert bb.class_name == "car"

    def test_class_id_optional(self):
        bb = ImageBoundingBox(np.array([[0, 0], [1, 1]]))
        assert bb.class_id is None
        assert bb.class_name is None

    def test_always_materialized(self):
        bb = ImageBoundingBox(
            points=np.array([[0, 0], [1, 1]]),
        )
        assert bb.state == ComponentState.MATERIALIZED
        assert bb.is_materialized is True

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError):
            ImageBoundingBox(points=np.array([1, 2, 3]))

    def test_points_shape(self):
        bb = ImageBoundingBox(
            points=np.array([[5, 10], [50, 100]]),
        )
        assert bb.points.shape == (2, 2)

    def test_component_name(self):
        bb = ImageBoundingBox(
            np.array([[0, 0], [1, 1]]), name="main",
        )
        assert bb.name == "main"

    def test_component_name_default_none(self):
        bb = ImageBoundingBox(np.array([[0, 0], [1, 1]]))
        assert bb.name is None

    def test_target(self):
        bb = ImageBoundingBox(
            np.array([[0, 0], [1, 1]]),
            target="left_cam",
        )
        assert bb.target == "left_cam"

    def test_is_annotation(self):
        bb = ImageBoundingBox(np.array([[0, 0], [1, 1]]))
        assert hasattr(bb, 'applies_to')


# -------------------------------------------------------------------
# ImagePolygon (was Polygon)
# -------------------------------------------------------------------

class TestImagePolygon:
    def test_construction(self):
        pts = np.array([[0, 0], [10, 0], [10, 10]])
        poly = ImagePolygon(points=pts)
        assert poly.points.shape == (3, 2)

    def test_area_triangle(self):
        pts = np.array(
            [[0, 0], [10, 0], [0, 10]], dtype=float,
        )
        poly = ImagePolygon(points=pts)
        assert poly.area == pytest.approx(50.0)

    def test_center(self):
        pts = np.array(
            [[0, 0], [6, 0], [6, 6], [0, 6]], dtype=float,
        )
        poly = ImagePolygon(points=pts)
        np.testing.assert_array_almost_equal(
            poly.center, [3.0, 3.0],
        )

    def test_class_id_and_name(self):
        pts = np.array([[0, 0], [5, 0], [5, 5]])
        poly = ImagePolygon(pts, class_id=1, class_name="face")
        assert poly.class_id == 1
        assert poly.class_name == "face"

    def test_always_materialized(self):
        pts = np.array([[0, 0], [1, 0], [1, 1]])
        poly = ImagePolygon(points=pts)
        assert poly.is_materialized is True

    def test_rejects_fewer_than_3_points(self):
        with pytest.raises(ValueError):
            ImagePolygon(
                points=np.array([[0, 0], [1, 1]]),
            )

    def test_component_name(self):
        pts = np.array([[0, 0], [1, 0], [1, 1]])
        poly = ImagePolygon(pts, name="outline")
        assert poly.name == "outline"

    def test_target(self):
        pts = np.array([[0, 0], [1, 0], [1, 1]])
        poly = ImagePolygon(pts, target="cam")
        assert poly.target == "cam"

    def test_is_annotation(self):
        pts = np.array([[0, 0], [1, 0], [1, 1]])
        assert hasattr(ImagePolygon(pts), 'applies_to')


# -------------------------------------------------------------------
# ImageKeyPoint (was KeyPoint)
# -------------------------------------------------------------------

class TestImageKeyPoint:
    def test_construction(self):
        kp = ImageKeyPoint(x=5.0, y=10.0)
        assert kp.x == 5.0
        assert kp.y == 10.0
        assert kp.visibility is None

    def test_visibility(self):
        kp = ImageKeyPoint(x=1.0, y=2.0, visibility=2)
        assert kp.visibility == 2

    def test_point_array(self):
        kp = ImageKeyPoint(x=3.0, y=7.0)
        np.testing.assert_array_equal(
            kp.point, [3.0, 7.0],
        )

    def test_class_id_and_name(self):
        kp = ImageKeyPoint(
            x=0, y=0, class_id=5, class_name="nose",
        )
        assert kp.class_id == 5
        assert kp.class_name == "nose"

    def test_always_materialized(self):
        kp = ImageKeyPoint(x=0.0, y=0.0)
        assert kp.is_materialized is True

    def test_component_name(self):
        kp = ImageKeyPoint(x=0, y=0, name="nose")
        assert kp.name == "nose"

    def test_target(self):
        kp = ImageKeyPoint(x=0, y=0, target="body")
        assert kp.target == "body"

    def test_is_annotation(self):
        assert hasattr(ImageKeyPoint(x=0, y=0), 'applies_to')


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

    def test_component_name(self, tmp_image_path):
        img = Image(path=tmp_image_path, name="left")
        assert img.name == "left"

    def test_component_name_default_none(
        self, tmp_image_path,
    ):
        img = Image(path=tmp_image_path)
        assert img.name is None


# -------------------------------------------------------------------
# Constant
# -------------------------------------------------------------------

class TestConstant:
    def test_construction_int(self):
        c = Constant(value=42)
        assert c.value == 42

    def test_construction_str(self):
        c = Constant(value="hello")
        assert c.value == "hello"

    def test_construction_dict(self):
        c = Constant(value={"key": "val"})
        assert c.value == {"key": "val"}

    def test_construction_list(self):
        c = Constant(value=[1, 2, 3])
        assert c.value == [1, 2, 3]

    def test_always_materialized(self):
        c = Constant(value=0)
        assert c.state == ComponentState.MATERIALIZED
        assert c.is_materialized is True

    def test_materialize_noop(self):
        c = Constant(value=99)
        c.materialize()
        assert c.value == 99

    def test_component_name(self):
        c = Constant(value=1, name="image_id")
        assert c.name == "image_id"

    def test_component_name_default_none(self):
        c = Constant(value=1)
        assert c.name is None

    def test_in_sample(self):
        c = Constant(value="train", name="split")
        s = Sample(c)
        assert s.get(Constant) is c

    def test_in_materialized_sample(self):
        c = Constant(value=123, name="img_id")
        s = Sample(c)
        assert s.get(Constant) is c
        assert s.get(Constant).value == 123


# -------------------------------------------------------------------
# Sample
# -------------------------------------------------------------------

class TestSample:
    def test_construction_bare_component(
        self, tmp_image_path,
    ):
        s = Sample(Image(path=tmp_image_path))
        assert len(s.components) == 1

    def test_construction_with_flat_annotations(
        self, tmp_image_path,
    ):
        s = Sample(
            Image(path=tmp_image_path),
            ImageBoundingBox(
                np.array([[0, 0], [10, 10]]),
                class_id=0,
                class_name="cat",
            ),
        )
        assert len(s.components) == 2

    def test_construction_mixed(self, tmp_image_path):
        s = Sample(
            Image(path=tmp_image_path),
            Text("hello"),
            ImageCategory(0),
        )
        assert len(s.components) == 3

    def test_get_by_type(self, tmp_image_path):
        img = Image(path=tmp_image_path)
        s = Sample(img, Text("hi"))
        assert s.get(Image) is img

    def test_get_annotation_component(self):
        cat = ImageCategory(0)
        s = Sample(cat)
        assert s.get(ImageCategory) is cat

    def test_get_annotation_base(self):
        cat = ImageCategory(0)
        s = Sample(cat)
        assert s.get_annotations() == [cat]

    def test_get_all_annotations(self):
        bb = ImageBoundingBox(np.array([[0, 0], [5, 5]]))
        cat = ImageCategory(0)
        s = Sample(bb, cat)
        assert s.get_annotations() == [bb, cat]

    def test_get_with_name(self, tmp_image_path):
        left = Image(path=tmp_image_path, name="left")
        right = Image(path=tmp_image_path, name="right")
        s = Sample(left, right)
        assert s.get(Image, name="right") is right

    def test_get_returns_none(self, tmp_image_path):
        s = Sample(Image(path=tmp_image_path))
        assert s.get(Text) is None

    def test_is_materialized_false(self, tmp_image_path):
        s = Sample(Image(path=tmp_image_path))
        assert s.is_materialized is False

    def test_is_materialized_true(self):
        s = Sample(ImageCategory(0))
        assert s.is_materialized is True

    def test_materialize_returns_self(
        self, tmp_image_path,
    ):
        s = Sample(
            Image(path=tmp_image_path),
            ImageCategory(0, "cat"),
        )
        result = s.materialize()
        assert result is s

    def test_materialize_loads_image(
        self, tmp_image_path,
    ):
        s = Sample(Image(path=tmp_image_path))
        s.materialize()
        assert isinstance(
            s.get(Image).data, np.ndarray,
        )

    def test_components_property(self, tmp_image_path):
        img = Image(path=tmp_image_path)
        cat = ImageCategory(0)
        s = Sample(img, cat)
        assert s.components == (img, cat)

    def test_get_annotations_all(self):
        bb = ImageBoundingBox(
            np.array([[0, 0], [5, 5]]),
            target="cam",
        )
        cat = ImageCategory(0, target="cam")
        img = Image.from_array(
            np.zeros((10, 10, 3), dtype=np.uint8),
            name="cam",
        )
        s = Sample(img, bb, cat)
        annots = s.get_annotations()
        assert bb in annots
        assert cat in annots
        assert img not in annots

    def test_get_annotations_by_target(self):
        bb_a = ImageBoundingBox(
            np.array([[0, 0], [5, 5]]),
            target="cam_a",
        )
        bb_b = ImageBoundingBox(
            np.array([[0, 0], [5, 5]]),
            target="cam_b",
        )
        img_a = Image.from_array(
            np.zeros((10, 10, 3), dtype=np.uint8),
            name="cam_a",
        )
        img_b = Image.from_array(
            np.zeros((10, 10, 3), dtype=np.uint8),
            name="cam_b",
        )
        s = Sample(img_a, img_b, bb_a, bb_b)
        assert s.get_annotations(target="cam_a") == [bb_a]
        assert s.get_annotations(target="cam_b") == [bb_b]


# -------------------------------------------------------------------
# Sample.replacing
# -------------------------------------------------------------------

class TestSampleReplacing:
    def test_replacing_returns_new_instance(self):
        img = Image.from_array(
            np.zeros((5, 5, 3), dtype=np.uint8),
        )
        new_img = Image.from_array(
            np.ones((5, 5, 3), dtype=np.uint8),
        )
        cat = ImageCategory(0)
        s = Sample(img, cat)
        s2 = s.replacing(img, new_img)
        assert s2 is not s
        assert s2.get(Image) is new_img
        assert s2.get(ImageCategory) is cat

    def test_replacing_does_not_mutate_original(self):
        img = Image.from_array(
            np.zeros((5, 5, 3), dtype=np.uint8),
        )
        new_img = Image.from_array(
            np.ones((5, 5, 3), dtype=np.uint8),
        )
        s = Sample(img)
        s.replacing(img, new_img)
        assert s.get(Image) is img


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------

class TestDataset:
    def test_construction(self, tmp_image_path):
        samples = [
            Sample(Image(path=tmp_image_path)),
        ]
        ds = Dataset(samples=samples, name="Test")
        assert ds.name == "Test"
        assert len(ds) == 1

    def test_getitem(self, tmp_image_path):
        s = Sample(Image(path=tmp_image_path))
        ds = Dataset(samples=[s])
        assert ds[0] is s

    def test_default_name(self):
        ds = Dataset(samples=[Sample(ImageCategory(0))])
        assert ds.name == "Dataset"

    def test_empty_samples(self):
        ds = Dataset(samples=[])
        assert len(ds) == 0


# -------------------------------------------------------------------
# Image.from_array
# -------------------------------------------------------------------

class TestImageFromArray:
    def test_creates_materialized_image(self):
        pixels = np.zeros((10, 10, 3), dtype=np.uint8)
        img = Image.from_array(pixels)
        assert img.is_materialized
        assert img.state == ComponentState.MATERIALIZED

    def test_data_matches_input(self):
        pixels = np.ones((5, 8, 3), dtype=np.uint8) * 42
        img = Image.from_array(pixels)
        assert np.array_equal(img.data, pixels)

    def test_path_is_empty_string(self):
        img = Image.from_array(np.zeros((2, 2, 3)))
        assert img.path == ""

    def test_format_hint(self):
        img = Image.from_array(
            np.zeros((2, 2, 3)), format_hint="png",
        )
        assert img.format_hint == "png"

    def test_materialize_is_noop(self):
        pixels = np.zeros((3, 3, 3), dtype=np.uint8)
        img = Image.from_array(pixels)
        img.materialize()
        assert np.array_equal(img.data, pixels)

    def test_name(self):
        img = Image.from_array(
            np.zeros((2, 2, 3)), name="test",
        )
        assert img.name == "test"


# -------------------------------------------------------------------
# Text component
# -------------------------------------------------------------------

class TestTextComponent:
    def test_construction(self):
        t = Text("hello world")
        assert t.text == "hello world"

    def test_default_language(self):
        t = Text("hello")
        assert t.language == "en"

    def test_custom_language(self):
        t = Text("bonjour", language="fr")
        assert t.language == "fr"

    def test_default_metadata(self):
        t = Text("hello")
        assert t.metadata == {}

    def test_custom_metadata(self):
        t = Text("hello", metadata={"src": "wiki"})
        assert t.metadata == {"src": "wiki"}

    def test_metadata_returns_copy(self):
        meta = {"key": "val"}
        t = Text("hello", metadata=meta)
        t.metadata["key"] = "changed"
        assert t.metadata["key"] == "val"

    def test_words(self):
        t = Text("the quick brown fox")
        assert t.words == ["the", "quick", "brown", "fox"]

    def test_words_empty(self):
        t = Text("")
        assert t.words == []

    def test_always_materialized(self):
        t = Text("hello")
        assert t.is_materialized is True
        assert t.state == ComponentState.MATERIALIZED

    def test_materialize_is_noop(self):
        t = Text("hello")
        t.materialize()
        assert t.text == "hello"

    def test_component_name(self):
        t = Text("hello", name="source")
        assert t.name == "source"


# -------------------------------------------------------------------
# Sample.merge
# -------------------------------------------------------------------

class TestSampleMerge:
    def test_merge_combines_components(
        self, tmp_image_path,
    ):
        img = Image(path=tmp_image_path)
        cat = ImageCategory(0)
        a = Sample(img)
        b = Sample(cat)
        merged = a.merge(b)
        assert img in merged.components
        assert cat in merged.components

    def test_merge_returns_new_instance(
        self, tmp_image_path,
    ):
        a = Sample(Image(path=tmp_image_path))
        b = Sample(ImageCategory(0))
        merged = a.merge(b)
        assert merged is not a
        assert merged is not b

    def test_merge_does_not_mutate_originals(
        self, tmp_image_path,
    ):
        img = Image(path=tmp_image_path)
        cat = ImageCategory(0)
        a = Sample(img)
        b = Sample(cat)
        a.merge(b)
        assert a.components == (img,)
        assert b.components == (cat,)

    def test_merge_preserves_order(
        self, tmp_image_path,
    ):
        img = Image(path=tmp_image_path)
        cat = ImageCategory(0)
        a = Sample(img)
        b = Sample(cat)
        merged = a.merge(b)
        assert merged.components == (img, cat)

    def test_merge_empty_sample(
        self, tmp_image_path,
    ):
        img = Image(path=tmp_image_path)
        a = Sample(img)
        b = Sample()
        merged = a.merge(b)
        assert merged.components == (img,)

    def test_add_operator(self, tmp_image_path):
        img = Image(path=tmp_image_path)
        cat = ImageCategory(0)
        merged = Sample(img) + Sample(cat)
        assert merged.components == (img, cat)


# -------------------------------------------------------------------
# Sample.modalities
# -------------------------------------------------------------------

class TestSampleModalities:
    def test_modalities_returns_names(
        self, tmp_image_path,
    ):
        s = Sample(
            Image(path=tmp_image_path, name="video"),
            Text("clip.wav", name="audio"),
        )
        assert s.modalities() == {"video", "audio"}

    def test_modalities_includes_none(
        self, tmp_image_path,
    ):
        s = Sample(
            Image(path=tmp_image_path, name="video"),
            ImageCategory(0),
        )
        assert s.modalities() == {"video", None}

    def test_get_annotation_by_name(self):
        cat_video = ImageCategory(0, "cat", name="video")
        cat_audio = ImageCategory(1, "speech", name="audio")
        s = Sample(cat_video, cat_audio)
        result = s.get_all(ImageCategory, name="video")
        assert result == [cat_video]

    def test_merge_preserves_modality_names(
        self, tmp_image_path,
    ):
        cat = ImageCategory(0, name="video")
        a = Sample(
            Image(path=tmp_image_path, name="video"),
        )
        b = Sample(cat)
        merged = a.merge(b)
        assert merged.get_all(
            ImageCategory, name="video",
        ) == [cat]
        assert merged.modalities() == {"video"}


