"""Tests for component entanglement — transform dispatch to
all applicable components with full component preservation."""
import numpy as np
import pytest

import daugx
from daugx.core.augmentation.image import Resize, Shift, Rotate
from daugx.core.data.components.image import Image
from daugx.core.data.components.text import Text
from daugx.core.data.components.constant import Constant
from daugx.core.data.sample import Sample


IMG_PATH = "daugx/img.png"
_RNG = np.random.default_rng(0)


def _pkg_single() -> Sample:
    """Single Image + bbox + text + constant."""
    return daugx.Sample(
        daugx.Image(path=IMG_PATH),
        daugx.ImageBoundingBox(
            np.array([[10.0, 10.0], [50.0, 50.0]]),
        ),
        daugx.Text(text="label"),
        daugx.Constant(value=99, name="id"),
    ).materialize()


def _pkg_two_images() -> Sample:
    """Two named Images, each with a scoped bbox."""
    return daugx.Sample(
        daugx.Image(path=IMG_PATH, name="left"),
        daugx.Image(path=IMG_PATH, name="right"),
        daugx.ImageBoundingBox(
            np.array([[10.0, 10.0], [50.0, 50.0]]),
            target="left",
        ),
        daugx.ImageBoundingBox(
            np.array([[20.0, 20.0], [80.0, 80.0]]),
            target="right",
        ),
        daugx.Constant(value=7, name="frame"),
    ).materialize()


class TestComponentPreservation:
    def test_resize_preserves_text(self):
        pkg = _pkg_single()
        result = Resize(64, 64).apply(pkg, _RNG)
        assert result.get(Text) is not None
        assert result.get(Text).text == "label"

    def test_resize_preserves_constant(self):
        pkg = _pkg_single()
        result = Resize(64, 64).apply(pkg, _RNG)
        c = result.get(Constant, name="id")
        assert c is not None
        assert c.value == 99

    def test_shift_preserves_text(self):
        pkg = _pkg_single()
        result = Shift(10, 5).apply(pkg, _RNG)
        assert result.get(Text) is not None

    def test_rotate_preserves_constant(self):
        pkg = _pkg_single()
        result = Rotate(45).apply(pkg, _RNG)
        c = result.get(Constant, name="id")
        assert c is not None
        assert c.value == 99


class TestMultiImageDispatch:
    def test_resize_applies_to_all_images(self):
        pkg = _pkg_two_images()
        result = Resize(64, 64).apply(pkg, _RNG)
        images = result.get_all(Image)
        assert len(images) == 2
        for img in images:
            assert img.data.shape[:2] == (64, 64)

    def test_shift_applies_to_all_images(self):
        pkg = _pkg_two_images()
        result = Shift(20, 10).apply(pkg, _RNG)
        images = result.get_all(Image)
        assert len(images) == 2

    def test_multi_image_preserves_constant(self):
        pkg = _pkg_two_images()
        result = Resize(64, 64).apply(pkg, _RNG)
        c = result.get(Constant, name="frame")
        assert c is not None
        assert c.value == 7


class TestAnnotationScoping:
    def test_left_bbox_transformed_with_left_image(self):
        """bbox targeting 'left' is transformed when 'left'
        Image is processed."""
        pkg = _pkg_two_images()
        orig_left_img = pkg.get(Image, name="left")
        orig_h, orig_w = orig_left_img.data.shape[:2]

        result = Resize(64, 64).apply(pkg, _RNG)
        left_bboxes = [
            b for b in result.get_all(daugx.ImageBoundingBox)
            if b.target == "left"
        ]
        assert len(left_bboxes) == 1
        # x_max should be scaled: 50 * (64 / orig_w)
        expected_x_max = 50.0 * (64 / orig_w)
        assert abs(left_bboxes[0].x_max - expected_x_max) < 1

    def test_right_bbox_not_transformed_with_left_image(self):
        """bbox targeting 'right' is NOT transformed when
        processing 'left' Image."""
        pkg = _pkg_two_images()
        result = Shift(100, 0).apply(pkg, _RNG)
        right_bboxes = [
            b for b in result.get_all(daugx.ImageBoundingBox)
            if b.target == "right"
        ]
        assert len(right_bboxes) == 1
        # Right bbox should have been shifted by the right-image transform
        assert right_bboxes[0].x_min == pytest.approx(120.0)

    def test_all_bboxes_present_after_multi_image_resize(self):
        pkg = _pkg_two_images()
        result = Resize(64, 64).apply(pkg, _RNG)
        bboxes = result.get_all(daugx.ImageBoundingBox)
        assert len(bboxes) == 2
        targets = {b.target for b in bboxes}
        assert targets == {"left", "right"}


class TestAppliesTo:
    def test_bbox_applies_to_image(self):
        assert daugx.ImageBoundingBox.applies_to is Image

    def test_polygon_applies_to_image(self):
        assert daugx.ImagePolygon.applies_to is Image

    def test_keypoint_applies_to_image(self):
        assert daugx.ImageKeyPoint.applies_to is Image

    def test_category_applies_to_image(self):
        assert daugx.ImageCategory.applies_to is Image
