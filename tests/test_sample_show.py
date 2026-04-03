"""Tests for Sample.show() visualization method."""
from unittest.mock import patch

import numpy as np
import pytest

from daugx.core.data._visualizer import (
    _assemble_grid,
    _label_color,
)
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

class TestShowMultipleImages:
    def test_multiple_images_no_filter_shows_grid(self):
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
            s.show()
        assert mock_show.call_count == 1

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

    def test_imshow_called_once_for_two_images_grid(self):
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
        assert mock_show.call_count == 1

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
        """BBox with target='other' is not drawn on 'main'
        region of the grid."""
        img_main = Image.from_array(
            np.zeros((50, 50, 3), dtype=np.uint8),
            name="main",
        )
        img_other = Image.from_array(
            np.zeros((50, 50, 3), dtype=np.uint8),
            name="other",
        )
        bbox_other = ImageBoundingBox(
            np.array([[5, 5], [20, 20]]),
            class_id=0,
            target="other",
        )
        s = Sample(img_main, img_other, bbox_other)
        with patch("cv2.imshow") as mock_show, \
                patch("cv2.waitKey"), \
                patch("cv2.destroyAllWindows"):
            s.show()
        # Grid is 50x100: main at [0:50, 0:50]
        _, grid = mock_show.call_args[0]
        main_region = grid[:50, :50]
        blank = np.zeros((50, 50, 3), dtype=np.uint8)
        assert np.array_equal(main_region, blank)


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


# -------------------------------------------------------------------
# Grid assembly
# -------------------------------------------------------------------

class TestGridAssembly:
    def test_single_image_passthrough(self):
        canvas = np.zeros((50, 50, 3), dtype=np.uint8)
        result = _assemble_grid([canvas])
        assert result is canvas

    def test_two_images_side_by_side(self):
        c1 = np.full((50, 50, 3), 10, dtype=np.uint8)
        c2 = np.full((50, 50, 3), 20, dtype=np.uint8)
        grid = _assemble_grid([c1, c2])
        assert grid.shape == (50, 100, 3)
        assert np.array_equal(grid[:, :50], c1)
        assert np.array_equal(grid[:, 50:], c2)

    def test_four_images_2x2(self):
        canvases = [
            np.full((30, 30, 3), i * 10, dtype=np.uint8)
            for i in range(4)
        ]
        grid = _assemble_grid(canvases)
        assert grid.shape == (60, 60, 3)

    def test_three_images_empty_cell_is_white(self):
        canvases = [
            np.full((30, 30, 3), 10, dtype=np.uint8)
            for _ in range(3)
        ]
        grid = _assemble_grid(canvases)
        # 2x2 grid, bottom-right cell should be white
        assert grid.shape == (60, 60, 3)
        cell = grid[30:60, 30:60]
        expected = np.full((30, 30, 3), 255, dtype=np.uint8)
        assert np.array_equal(cell, expected)

    def test_different_sizes_no_resize(self):
        big = np.full((100, 100, 3), 10, dtype=np.uint8)
        small = np.full((50, 50, 3), 20, dtype=np.uint8)
        grid = _assemble_grid([big, small])
        # Cell size = 100x100, grid = 100x200
        assert grid.shape == (100, 200, 3)
        # Big image occupies full first cell
        assert np.array_equal(grid[:100, :100], big)
        # Small image top-left of second cell
        assert np.array_equal(grid[:50, 100:150], small)
        # Remaining space in second cell is white
        right_pad = grid[:50, 150:200]
        assert np.all(right_pad == 255)
        bottom_pad = grid[50:100, 100:200]
        assert np.all(bottom_pad == 255)

    def test_nine_images_3x3(self):
        canvases = [
            np.full((20, 20, 3), i * 10, dtype=np.uint8)
            for i in range(9)
        ]
        grid = _assemble_grid(canvases)
        assert grid.shape == (60, 60, 3)


# -------------------------------------------------------------------
# Components normalization
# -------------------------------------------------------------------

class TestComponentsNormalization:
    def test_single_type_works(self):
        """components=Image (not a tuple) shows images only."""
        img = Image.from_array(
            np.zeros((30, 30, 3), dtype=np.uint8),
        )
        bbox = ImageBoundingBox(
            np.array([[5, 5], [20, 20]]), class_id=0,
        )
        s = Sample(img, bbox)
        with patch("cv2.imshow") as mock_show, \
                patch("cv2.waitKey"), \
                patch("cv2.destroyAllWindows"):
            s.show(components=Image)
        _, canvas = mock_show.call_args[0]
        blank = np.zeros((30, 30, 3), dtype=np.uint8)
        assert np.array_equal(canvas, blank)

    def test_none_shows_all_with_annotations(self):
        img = Image.from_array(
            np.zeros((50, 50, 3), dtype=np.uint8),
        )
        bbox = ImageBoundingBox(
            np.array([[5, 5], [20, 20]]), class_id=0,
        )
        s = Sample(img, bbox)
        with patch("cv2.imshow") as mock_show, \
                patch("cv2.waitKey"), \
                patch("cv2.destroyAllWindows"):
            s.show()
        _, canvas = mock_show.call_args[0]
        blank = np.zeros((50, 50, 3), dtype=np.uint8)
        assert not np.array_equal(canvas, blank)

    def test_image_tuple_excludes_annotations(self):
        img = Image.from_array(
            np.zeros((30, 30, 3), dtype=np.uint8),
        )
        bbox = ImageBoundingBox(
            np.array([[5, 5], [20, 20]]), class_id=0,
        )
        s = Sample(img, bbox)
        with patch("cv2.imshow") as mock_show, \
                patch("cv2.waitKey"), \
                patch("cv2.destroyAllWindows"):
            s.show(components=(Image,))
        _, canvas = mock_show.call_args[0]
        blank = np.zeros((30, 30, 3), dtype=np.uint8)
        assert np.array_equal(canvas, blank)

    def test_annotation_only_raises(self):
        """components without Image raises ValueError."""
        img = Image.from_array(
            np.zeros((30, 30, 3), dtype=np.uint8),
        )
        poly = ImagePolygon(
            np.array([[5, 5], [20, 5], [20, 20]]),
            class_id=0,
        )
        s = Sample(img, poly)
        with pytest.raises(ValueError):
            s.show(components=ImagePolygon)
        with pytest.raises(ValueError):
            s.show(components=(ImagePolygon,))

    def test_image_and_bbox_excludes_polygon(self):
        img = Image.from_array(
            np.zeros((100, 100, 3), dtype=np.uint8),
        )
        bbox = ImageBoundingBox(
            np.array([[5, 5], [20, 20]]), class_id=0,
        )
        poly = ImagePolygon(
            np.array([[60, 60], [90, 60], [90, 90]]),
            class_id=1,
        )
        s = Sample(img, bbox, poly)
        with patch("cv2.imshow") as mock_show, \
                patch("cv2.waitKey"), \
                patch("cv2.destroyAllWindows"):
            s.show(components=(Image, ImageBoundingBox))
        _, canvas = mock_show.call_args[0]
        # Polygon region should be untouched (black)
        poly_region = canvas[60:90, 60:90]
        blank_region = np.zeros_like(poly_region)
        assert np.array_equal(poly_region, blank_region)
        # But bbox region should be drawn
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        assert not np.array_equal(canvas, blank)


# -------------------------------------------------------------------
# Window name
# -------------------------------------------------------------------

class TestWindowName:
    def test_window_name_is_sample(self):
        img = Image.from_array(
            np.zeros((10, 10, 3), dtype=np.uint8),
        )
        s = Sample(img)
        with patch("cv2.imshow") as mock_show, \
                patch("cv2.waitKey"), \
                patch("cv2.destroyAllWindows"):
            s.show()
        window_name = mock_show.call_args[0][0]
        assert window_name == "Sample"
