"""Tests for Sample.show() visualization method."""
from unittest.mock import patch

import numpy as np
import pytest

from daugx.core.data._visualizer import _label_color
from daugx.core.data.components.bounding_box import (
    ImageBoundingBox,
)
from daugx.core.data.components.image import Image
from daugx.core.data.components.keypoint import ImageKeyPoint
from daugx.core.data.components.polygon import ImagePolygon
from daugx.core.data.sample import Sample


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def blank_image():
    return Image.from_array(
        np.zeros((100, 100, 3), dtype=np.uint8),
    )


@pytest.fixture
def blank_image_named():
    return Image.from_array(
        np.zeros((100, 100, 3), dtype=np.uint8),
        name="main",
    )


@pytest.fixture
def bbox():
    return ImageBoundingBox(
        np.array([[10, 10], [40, 40]]),
        class_id=0,
        class_name="cat",
    )


@pytest.fixture
def bbox_named(bbox):
    return ImageBoundingBox(
        np.array([[10, 10], [40, 40]]),
        class_id=0,
        class_name="cat",
        target="main",
    )


@pytest.fixture
def polygon():
    return ImagePolygon(
        np.array([[10, 10], [50, 10], [50, 50], [10, 50]]),
        class_id=1,
        class_name="dog",
    )


@pytest.fixture
def keypoint():
    return ImageKeyPoint(
        x=25.0, y=25.0, visibility=2,
        class_id=2, class_name="nose",
    )


# -------------------------------------------------------------------
# Ambiguity / error cases
# -------------------------------------------------------------------

class TestShowAmbiguity:
    def test_raises_on_multiple_images_no_filter(self):
        img_a = Image.from_array(
            np.zeros((10, 10, 3), dtype=np.uint8), name="a",
        )
        img_b = Image.from_array(
            np.zeros((10, 10, 3), dtype=np.uint8), name="b",
        )
        s = Sample(img_a, img_b)
        with pytest.raises(ValueError):
            s.show()

    def test_single_image_no_filter_does_not_raise(
        self, blank_image,
    ):
        s = Sample(blank_image)
        with patch("cv2.imshow"), patch("cv2.waitKey"), \
                patch("cv2.destroyAllWindows"):
            s.show()  # must not raise

    def test_multiple_images_with_filter_does_not_raise(self):
        img_a = Image.from_array(
            np.zeros((10, 10, 3), dtype=np.uint8), name="a",
        )
        img_b = Image.from_array(
            np.zeros((10, 10, 3), dtype=np.uint8), name="b",
        )
        s = Sample(img_a, img_b)
        with patch("cv2.imshow"), patch("cv2.waitKey"), \
                patch("cv2.destroyAllWindows"):
            s.show(components=(Image,))  # must not raise


# -------------------------------------------------------------------
# cv2 calls
# -------------------------------------------------------------------

class TestShowCv2Calls:
    def test_imshow_called_once_for_single_image(
        self, blank_image, bbox,
    ):
        s = Sample(blank_image, bbox)
        with patch("cv2.imshow") as mock_show, \
                patch("cv2.waitKey"), \
                patch("cv2.destroyAllWindows"):
            s.show()
        assert mock_show.call_count == 1

    def test_imshow_called_twice_for_two_images_with_filter(
        self,
    ):
        img_a = Image.from_array(
            np.zeros((10, 10, 3), dtype=np.uint8), name="a",
        )
        img_b = Image.from_array(
            np.zeros((10, 10, 3), dtype=np.uint8), name="b",
        )
        s = Sample(img_a, img_b)
        with patch("cv2.imshow") as mock_show, \
                patch("cv2.waitKey"), \
                patch("cv2.destroyAllWindows"):
            s.show(components=(Image,))
        assert mock_show.call_count == 2

    def test_imshow_receives_ndarray(self, blank_image, bbox):
        s = Sample(blank_image, bbox)
        with patch("cv2.imshow") as mock_show, \
                patch("cv2.waitKey"), \
                patch("cv2.destroyAllWindows"):
            s.show()
        _, canvas = mock_show.call_args[0]
        assert isinstance(canvas, np.ndarray)
        assert canvas.shape == (100, 100, 3)

    def test_waitkey_called(self, blank_image):
        s = Sample(blank_image)
        with patch("cv2.imshow"), \
                patch("cv2.waitKey") as mock_wk, \
                patch("cv2.destroyAllWindows"):
            s.show()
        assert mock_wk.call_count == 1

    def test_destroy_windows_called(self, blank_image):
        s = Sample(blank_image)
        with patch("cv2.imshow"), patch("cv2.waitKey"), \
                patch("cv2.destroyAllWindows") as mock_d:
            s.show()
        assert mock_d.call_count == 1


# -------------------------------------------------------------------
# Annotation drawing
# -------------------------------------------------------------------

class TestShowWithAnnotations:
    def test_show_with_bbox(self, blank_image, bbox):
        s = Sample(blank_image, bbox)
        with patch("cv2.imshow") as mock_show, \
                patch("cv2.waitKey"), \
                patch("cv2.destroyAllWindows"):
            s.show()
        _, canvas = mock_show.call_args[0]
        # Canvas must differ from blank (annotations were drawn)
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        assert not np.array_equal(canvas, blank)

    def test_show_with_polygon(self, blank_image, polygon):
        s = Sample(blank_image, polygon)
        with patch("cv2.imshow") as mock_show, \
                patch("cv2.waitKey"), \
                patch("cv2.destroyAllWindows"):
            s.show()
        _, canvas = mock_show.call_args[0]
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        assert not np.array_equal(canvas, blank)

    def test_show_with_keypoint(self, blank_image, keypoint):
        s = Sample(blank_image, keypoint)
        with patch("cv2.imshow") as mock_show, \
                patch("cv2.waitKey"), \
                patch("cv2.destroyAllWindows"):
            s.show()
        _, canvas = mock_show.call_args[0]
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        assert not np.array_equal(canvas, blank)

    def test_components_filter_excludes_bbox(
        self, blank_image, bbox,
    ):
        """With components=(Image,) only, BBox is not drawn."""
        s = Sample(blank_image, bbox)
        with patch("cv2.imshow") as mock_show, \
                patch("cv2.waitKey"), \
                patch("cv2.destroyAllWindows"):
            s.show(components=(Image,))
        _, canvas = mock_show.call_args[0]
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        assert np.array_equal(canvas, blank)

    def test_scoped_annotation_only_on_target_image(self):
        """BBox with target='other' is not drawn on Image 'main'."""
        img_main = Image.from_array(
            np.zeros((50, 50, 3), dtype=np.uint8), name="main",
        )
        img_other = Image.from_array(
            np.zeros((50, 50, 3), dtype=np.uint8), name="other",
        )
        bbox_other = ImageBoundingBox(
            np.array([[5, 5], [20, 20]]),
            class_id=0,
            target="other",
        )
        s = Sample(img_main, img_other, bbox_other)
        captured = {}
        def capture_imshow(name, canvas):
            captured[name] = canvas.copy()
        with patch("cv2.imshow", side_effect=capture_imshow), \
                patch("cv2.waitKey"), \
                patch("cv2.destroyAllWindows"):
            s.show(components=(Image,))
        # "main" image should be blank (bbox belongs to "other")
        blank = np.zeros((50, 50, 3), dtype=np.uint8)
        assert np.array_equal(captured["main"], blank)


# -------------------------------------------------------------------
# _label_color
# -------------------------------------------------------------------

class TestLabelColor:
    def test_reproducible_by_class_id(self):
        c1 = _label_color(0)
        c2 = _label_color(0)
        assert c1 == c2

    def test_reproducible_by_class_name(self):
        c1 = _label_color("cat")
        c2 = _label_color("cat")
        assert c1 == c2

    def test_different_keys_give_different_colors(self):
        colors = {_label_color(i) for i in range(20)}
        assert len(colors) > 10

    def test_color_is_bgr_tuple(self):
        color = _label_color(0)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)
