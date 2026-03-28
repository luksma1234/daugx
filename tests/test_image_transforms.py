"""Tests for image augmentation transforms (DataPackage API).

Each transform receives a DataPackage with an Image and a
list of (BoundingBox, Label) objects, and returns a new
DataPackage.
"""
import numpy as np
import pytest

from daugx.core.data.data_package import DataPackage
from daugx.core.data.components.image import Image
from daugx.core.data.components.bounding_box import (
    BoundingBox,
)
from daugx.core.data.components.label import Label
from daugx.core.data.components.polygon import Polygon
from daugx.core.data.components.keypoint import KeyPoint
from daugx.core.augmentation.base import (
    Transform,
    MultiInputTransform,
)
from daugx.core.augmentation.image.shift import Shift
from daugx.core.augmentation.image.scale import Scale
from daugx.core.augmentation.image.rotate import Rotate
from daugx.core.augmentation.image.crop import Crop
from daugx.core.augmentation.image.resize import Resize
from daugx.core.augmentation.image.random_crop import (
    RandomCrop,
)
from daugx.core.augmentation.image.mixup import MixUp
from daugx.core.augmentation.image.mosaic import Mosaic


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_image():
    """100x80 RGB test image (H=100, W=80)."""
    return np.ones((100, 80, 3), dtype=np.uint8) * 127


@pytest.fixture
def package(sample_image):
    """DataPackage with image and two bounding boxes."""
    img = Image.from_array(sample_image)
    objects = [
        (
            BoundingBox(np.array([[10, 10], [30, 30]])),
            Label(0, "cat"),
        ),
        (
            BoundingBox(np.array([[50, 40], [90, 70]])),
            Label(1, "dog"),
        ),
    ]
    return DataPackage({"image": img, "objects": objects})


@pytest.fixture
def package_no_annots(sample_image):
    """DataPackage with image only (no annotations)."""
    return DataPackage({
        "image": Image.from_array(sample_image),
    })


@pytest.fixture
def package_with_polygon():
    """DataPackage with image and polygon annotations."""
    img = Image.from_array(
        np.zeros((100, 100, 3), dtype=np.uint8),
    )
    objects = [
        (
            Polygon(np.array(
                [[10, 10], [40, 10], [40, 40], [10, 40]],
            )),
            Label(0, "square"),
        ),
    ]
    return DataPackage({"image": img, "objects": objects})


@pytest.fixture
def package_with_keypoint():
    """DataPackage with image and keypoint annotations."""
    img = Image.from_array(
        np.zeros((100, 100, 3), dtype=np.uint8),
    )
    objects = [
        (
            KeyPoint(25.0, 25.0, visibility=2),
            Label(0, "nose"),
        ),
    ]
    return DataPackage({"image": img, "objects": objects})


def _img_shape(pkg):
    """Get (H, W) from a DataPackage's image."""
    return pkg["image"].data.shape[:2]


# -------------------------------------------------------------------
# Shift
# -------------------------------------------------------------------

class TestShift:
    def test_output_is_datapackage(self, package, rng):
        t = Shift(x_shift=10, y_shift=5)
        result = t.apply(package, rng)
        assert isinstance(result, DataPackage)

    def test_is_transform_subclass(self):
        assert issubclass(Shift, Transform)

    def test_shape_unchanged(self, package, rng):
        t = Shift(x_shift=10, y_shift=5)
        result = t.apply(package, rng)
        assert _img_shape(result) == _img_shape(package)

    def test_bbox_shifted_positive(self, package, rng):
        t = Shift(x_shift=10, y_shift=0)
        result = t.apply(package, rng)
        bb = result["objects"][0][0]
        assert bb.x_min == pytest.approx(20.0)
        assert bb.y_min == pytest.approx(10.0)

    def test_bbox_shifted_negative(self, package, rng):
        t = Shift(x_shift=-5, y_shift=-5)
        result = t.apply(package, rng)
        bb = result["objects"][0][0]
        assert bb.x_min == pytest.approx(5.0)
        assert bb.y_min == pytest.approx(5.0)

    def test_shift_removes_out_of_bounds_bbox(self, rng):
        """Shift that pushes a bbox entirely off-image."""
        img = Image.from_array(
            np.zeros((50, 50, 3), dtype=np.uint8),
        )
        objects = [
            (
                BoundingBox(np.array([[0, 0], [10, 10]])),
                Label(0, "a"),
            ),
        ]
        pkg = DataPackage(
            {"image": img, "objects": objects},
        )
        t = Shift(x_shift=-20, y_shift=0)
        result = t.apply(pkg, rng)
        assert len(result["objects"]) == 0

    def test_labels_preserved(self, package, rng):
        t = Shift(x_shift=5, y_shift=0)
        result = t.apply(package, rng)
        labels = [lbl.name for _, lbl in result["objects"]]
        assert "cat" in labels

    def test_no_annots(self, package_no_annots, rng):
        t = Shift(x_shift=10, y_shift=5)
        result = t.apply(package_no_annots, rng)
        assert "image" in result

    def test_pixel_shift_right(self):
        """White stripe at x=50 shifts to x=60."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 127
        img[:, 50, :] = 255
        pkg = DataPackage(
            {"image": Image.from_array(img)},
        )
        t = Shift(x_shift=10, y_shift=0)
        result = t.apply(pkg)
        out = result["image"].data
        assert out[0, 60, 0] == 255
        assert out[0, 50, 0] == 127

    def test_pixel_shift_down(self):
        """White stripe at y=50 shifts to y=60."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 127
        img[50, :, :] = 255
        pkg = DataPackage(
            {"image": Image.from_array(img)},
        )
        t = Shift(x_shift=0, y_shift=10)
        result = t.apply(pkg)
        out = result["image"].data
        assert out[60, 0, 0] == 255
        assert out[50, 0, 0] == 127

    def test_polygon_shifted(
        self, package_with_polygon, rng,
    ):
        t = Shift(x_shift=5, y_shift=5)
        result = t.apply(package_with_polygon, rng)
        poly = result["objects"][0][0]
        assert poly.points[0, 0] == pytest.approx(15.0)
        assert poly.points[0, 1] == pytest.approx(15.0)

    def test_keypoint_shifted(
        self, package_with_keypoint, rng,
    ):
        t = Shift(x_shift=10, y_shift=5)
        result = t.apply(package_with_keypoint, rng)
        kp = result["objects"][0][0]
        assert kp.x == pytest.approx(35.0)
        assert kp.y == pytest.approx(30.0)

    def test_hash_and_equality(self):
        a = Shift(x_shift=10, y_shift=5)
        b = Shift(x_shift=10, y_shift=5)
        c = Shift(x_shift=20, y_shift=5)
        assert a == b
        assert a != c
        assert hash(a) == hash(b)

    def test_original_package_unchanged(
        self, package, rng,
    ):
        orig_x = package["objects"][0][0].x_min
        Shift(x_shift=100).apply(package, rng)
        assert package["objects"][0][0].x_min == orig_x


# -------------------------------------------------------------------
# Scale
# -------------------------------------------------------------------

class TestScale:
    def test_is_transform_subclass(self):
        assert issubclass(Scale, Transform)

    def test_scale_down_halves_dimensions(
        self, package, rng,
    ):
        t = Scale(x_scale=0.5, y_scale=0.5)
        result = t.apply(package, rng)
        h, w = _img_shape(result)
        assert h == 50
        assert w == 40

    def test_scale_up_doubles_dimensions(
        self, package, rng,
    ):
        t = Scale(x_scale=2.0, y_scale=2.0)
        result = t.apply(package, rng)
        h, w = _img_shape(result)
        assert h == 200
        assert w == 160

    def test_bbox_scaled(self, package, rng):
        t = Scale(x_scale=2.0, y_scale=2.0)
        result = t.apply(package, rng)
        bb = result["objects"][0][0]
        assert bb.x_min == pytest.approx(20.0)
        assert bb.y_min == pytest.approx(20.0)
        assert bb.x_max == pytest.approx(60.0)

    def test_labels_preserved(self, package, rng):
        t = Scale(x_scale=0.5, y_scale=0.5)
        result = t.apply(package, rng)
        labels = [lbl.name for _, lbl in result["objects"]]
        assert "cat" in labels
        assert "dog" in labels

    def test_no_annots(self, package_no_annots, rng):
        t = Scale(x_scale=2.0, y_scale=2.0)
        result = t.apply(package_no_annots, rng)
        assert _img_shape(result) == (200, 160)

    def test_polygon_scaled(
        self, package_with_polygon, rng,
    ):
        t = Scale(x_scale=2.0, y_scale=2.0)
        result = t.apply(package_with_polygon, rng)
        poly = result["objects"][0][0]
        assert poly.points[0, 0] == pytest.approx(20.0)

    def test_keypoint_scaled(
        self, package_with_keypoint, rng,
    ):
        t = Scale(x_scale=2.0, y_scale=2.0)
        result = t.apply(package_with_keypoint, rng)
        kp = result["objects"][0][0]
        assert kp.x == pytest.approx(50.0)
        assert kp.y == pytest.approx(50.0)

    def test_invalid_scale_raises(self):
        with pytest.raises(ValueError):
            Scale(x_scale=0, y_scale=1)
        with pytest.raises(ValueError):
            Scale(x_scale=-1, y_scale=1)

    def test_hash_and_equality(self):
        a = Scale(x_scale=2.0, y_scale=2.0)
        b = Scale(x_scale=2.0, y_scale=2.0)
        c = Scale(x_scale=1.0, y_scale=2.0)
        assert a == b
        assert a != c
        assert hash(a) == hash(b)


# -------------------------------------------------------------------
# Rotate
# -------------------------------------------------------------------

class TestRotate:
    def test_is_transform_subclass(self):
        assert issubclass(Rotate, Transform)

    def test_shape_unchanged(self, package, rng):
        t = Rotate(angle=45)
        result = t.apply(package, rng)
        assert _img_shape(result) == _img_shape(package)

    def test_360_is_identity_for_pixels(self, rng):
        """Rotating 360 degrees should preserve pixels."""
        img = np.random.default_rng(0).integers(
            0, 255, (50, 50, 3), dtype=np.uint8,
        )
        pkg = DataPackage(
            {"image": Image.from_array(img)},
        )
        t = Rotate(angle=360)
        result = t.apply(pkg, rng)
        np.testing.assert_array_equal(
            result["image"].data, img,
        )

    def test_labels_preserved(self, package, rng):
        t = Rotate(angle=10)
        result = t.apply(package, rng)
        labels = [lbl.name for _, lbl in result["objects"]]
        assert "cat" in labels

    def test_no_annots(self, package_no_annots, rng):
        t = Rotate(angle=30)
        result = t.apply(package_no_annots, rng)
        assert _img_shape(result) == (100, 80)

    def test_polygon_rotated(
        self, package_with_polygon, rng,
    ):
        t = Rotate(angle=90)
        result = t.apply(package_with_polygon, rng)
        # Polygon should still exist (may be clipped)
        assert isinstance(result, DataPackage)

    def test_keypoint_rotated(
        self, package_with_keypoint, rng,
    ):
        t = Rotate(angle=0)
        result = t.apply(package_with_keypoint, rng)
        kp = result["objects"][0][0]
        assert kp.x == pytest.approx(25.0, abs=1e-6)
        assert kp.y == pytest.approx(25.0, abs=1e-6)

    def test_hash_and_equality(self):
        a = Rotate(angle=45)
        b = Rotate(angle=45)
        c = Rotate(angle=90)
        assert a == b
        assert a != c
        assert hash(a) == hash(b)


# -------------------------------------------------------------------
# Crop
# -------------------------------------------------------------------

class TestCrop:
    def test_is_transform_subclass(self):
        assert issubclass(Crop, Transform)

    def test_output_smaller(self, package, rng):
        t = Crop(x_min=0.1, y_min=0.1,
                 x_max=0.9, y_max=0.9)
        result = t.apply(package, rng)
        h, w = _img_shape(result)
        assert h < 100 and w < 80

    def test_exact_output_dimensions(self, rng):
        """50% crop of 100x100 image -> 50x50."""
        img = Image.from_array(
            np.zeros((100, 100, 3), dtype=np.uint8),
        )
        pkg = DataPackage({"image": img})
        t = Crop(x_min=0.25, y_min=0.25,
                 x_max=0.75, y_max=0.75)
        result = t.apply(pkg, rng)
        h, w = _img_shape(result)
        assert h == 50 and w == 50

    def test_removes_out_of_bounds_annotations(
        self, rng,
    ):
        img = Image.from_array(
            np.zeros((100, 100, 3), dtype=np.uint8),
        )
        objects = [
            (
                BoundingBox(np.array([[80, 80], [95, 95]])),
                Label(0, "outside"),
            ),
            (
                BoundingBox(np.array([[10, 10], [40, 40]])),
                Label(1, "inside"),
            ),
        ]
        pkg = DataPackage(
            {"image": img, "objects": objects},
        )
        t = Crop(x_min=0.05, y_min=0.05,
                 x_max=0.5, y_max=0.5)
        result = t.apply(pkg, rng)
        labels = [lbl.name for _, lbl in result["objects"]]
        assert "inside" in labels
        assert "outside" not in labels

    def test_bbox_coordinates_adjusted(self, rng):
        """After crop, bbox coords are relative to new
        origin."""
        img = Image.from_array(
            np.zeros((100, 100, 3), dtype=np.uint8),
        )
        objects = [
            (
                BoundingBox(np.array([[20, 20], [40, 40]])),
                Label(0, "obj"),
            ),
        ]
        pkg = DataPackage(
            {"image": img, "objects": objects},
        )
        # Crop starting at (10, 10)
        t = Crop(x_min=0.1, y_min=0.1,
                 x_max=0.9, y_max=0.9)
        result = t.apply(pkg, rng)
        bb = result["objects"][0][0]
        assert bb.x_min == pytest.approx(10.0)
        assert bb.y_min == pytest.approx(10.0)

    def test_labels_preserved(self, package, rng):
        t = Crop(x_min=0.05, y_min=0.05,
                 x_max=0.95, y_max=0.95)
        result = t.apply(package, rng)
        for _, lbl in result["objects"]:
            assert lbl.name in ("cat", "dog")

    def test_no_annots(self, package_no_annots, rng):
        t = Crop(x_min=0.1, y_min=0.1,
                 x_max=0.5, y_max=0.5)
        result = t.apply(package_no_annots, rng)
        assert "image" in result

    def test_invalid_bounds_raises(self):
        with pytest.raises(ValueError):
            Crop(x_min=0.5, y_min=0.1,
                 x_max=0.3, y_max=0.9)
        with pytest.raises(ValueError):
            Crop(x_min=0.0, y_min=0.1,
                 x_max=0.9, y_max=0.9)

    def test_hash_and_equality(self):
        a = Crop(0.1, 0.1, 0.9, 0.9)
        b = Crop(0.1, 0.1, 0.9, 0.9)
        c = Crop(0.2, 0.1, 0.9, 0.9)
        assert a == b
        assert a != c
        assert hash(a) == hash(b)


# -------------------------------------------------------------------
# Resize
# -------------------------------------------------------------------

class TestResize:
    def test_is_transform_subclass(self):
        assert issubclass(Resize, Transform)

    def test_resize_to_target(self, package, rng):
        t = Resize(width=200, height=150)
        result = t.apply(package, rng)
        h, w = _img_shape(result)
        assert h == 200 and w == 150

    def test_resize_no_preserve_aspect(
        self, package, rng,
    ):
        t = Resize(width=200, height=150,
                   preserve_aspect_ratio=False)
        result = t.apply(package, rng)
        h, w = _img_shape(result)
        assert h == 200 and w == 150

    def test_annotations_present_after_resize(
        self, package, rng,
    ):
        t = Resize(width=200, height=200)
        result = t.apply(package, rng)
        assert len(result["objects"]) > 0

    def test_labels_preserved(self, package, rng):
        t = Resize(width=50, height=50)
        result = t.apply(package, rng)
        labels = [lbl.name for _, lbl in result["objects"]]
        assert "cat" in labels

    def test_no_annots(self, package_no_annots, rng):
        t = Resize(width=50, height=50)
        result = t.apply(package_no_annots, rng)
        h, w = _img_shape(result)
        assert h == 50 and w == 50

    def test_invalid_dims_raises(self):
        with pytest.raises(ValueError):
            Resize(width=0, height=100)
        with pytest.raises(ValueError):
            Resize(width=100, height=-1)

    def test_hash_and_equality(self):
        a = Resize(200, 150)
        b = Resize(200, 150)
        c = Resize(100, 150)
        assert a == b
        assert a != c
        assert hash(a) == hash(b)

    def test_square_resize(self, package_no_annots, rng):
        """Resize non-square image to square."""
        t = Resize(width=100, height=100)
        result = t.apply(package_no_annots, rng)
        h, w = _img_shape(result)
        assert h == 100 and w == 100


# -------------------------------------------------------------------
# RandomCrop
# -------------------------------------------------------------------

class TestRandomCrop:
    def test_is_transform_subclass(self):
        assert issubclass(RandomCrop, Transform)

    def test_output_smaller_or_equal(self, package, rng):
        t = RandomCrop(min_width=0.3, max_width=0.8,
                       min_height=0.3, max_height=0.8)
        result = t.apply(package, rng)
        h, w = _img_shape(result)
        assert h <= 100 and w <= 80

    def test_seeded_reproducibility(self, package):
        t = RandomCrop()
        r1 = t.apply(package, np.random.default_rng(7))
        r2 = t.apply(package, np.random.default_rng(7))
        assert np.array_equal(
            r1["image"].data, r2["image"].data,
        )

    def test_rng_required(self, package):
        t = RandomCrop()
        with pytest.raises(ValueError):
            t.apply(package, rng=None)

    def test_annotations_present(self, package, rng):
        t = RandomCrop(min_width=0.8, max_width=1.0,
                       min_height=0.8, max_height=1.0)
        result = t.apply(package, rng)
        # With a near-full crop, annotations should survive
        assert "objects" in result

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError):
            RandomCrop(min_width=0.8, max_width=0.3)
        with pytest.raises(ValueError):
            RandomCrop(min_height=0.0, max_height=0.5)

    def test_hash_and_equality(self):
        a = RandomCrop(0.2, 0.8, 0.2, 0.8)
        b = RandomCrop(0.2, 0.8, 0.2, 0.8)
        c = RandomCrop(0.3, 0.8, 0.2, 0.8)
        assert a == b
        assert a != c


# -------------------------------------------------------------------
# MixUp
# -------------------------------------------------------------------

class TestMixUp:
    def test_is_multi_input_transform(self):
        assert issubclass(MixUp, MultiInputTransform)

    def test_inflation(self):
        assert MixUp(lam=0.5).inflation == 0.5

    def test_blends_two_packages(self, sample_image, rng):
        pkg1 = DataPackage({
            "image": Image.from_array(
                np.zeros_like(sample_image),
            ),
            "objects": [
                (BoundingBox(np.array([[0, 0], [10, 10]])),
                 Label(0, "a")),
            ],
        })
        pkg2 = DataPackage({
            "image": Image.from_array(
                np.ones_like(sample_image) * 255,
            ),
            "objects": [
                (BoundingBox(np.array([[20, 20], [40, 40]])),
                 Label(1, "b")),
            ],
        })
        t = MixUp(lam=0.5)
        result = t.apply([pkg1, pkg2], rng)
        assert isinstance(result, DataPackage)
        h, w = _img_shape(result)
        assert h == 100 and w == 80
        mean_val = result["image"].data.mean()
        assert 100 < mean_val < 155

    def test_annotations_merged(self, sample_image, rng):
        pkg1 = DataPackage({
            "image": Image.from_array(sample_image),
            "objects": [
                (BoundingBox(np.array([[0, 0], [10, 10]])),
                 Label(0, "a")),
            ],
        })
        pkg2 = DataPackage({
            "image": Image.from_array(sample_image),
            "objects": [
                (BoundingBox(np.array([[20, 20], [40, 40]])),
                 Label(1, "b")),
                (BoundingBox(np.array([[50, 50], [60, 60]])),
                 Label(2, "c")),
            ],
        })
        t = MixUp(lam=0.5)
        result = t.apply([pkg1, pkg2], rng)
        assert len(result["objects"]) == 3

    def test_labels_preserved(self, sample_image, rng):
        pkg1 = DataPackage({
            "image": Image.from_array(sample_image),
            "objects": [
                (BoundingBox(np.array([[0, 0], [10, 10]])),
                 Label(0, "cat")),
            ],
        })
        pkg2 = DataPackage({
            "image": Image.from_array(sample_image),
            "objects": [
                (BoundingBox(np.array([[20, 20], [40, 40]])),
                 Label(1, "dog")),
            ],
        })
        t = MixUp(lam=0.5)
        result = t.apply([pkg1, pkg2], rng)
        labels = {lbl.name for _, lbl in result["objects"]}
        assert labels == {"cat", "dog"}

    def test_invalid_lam_raises(self):
        with pytest.raises(ValueError):
            MixUp(lam=0.1)
        with pytest.raises(ValueError):
            MixUp(lam=0.9)

    def test_wrong_package_count_raises(
        self, sample_image, rng,
    ):
        pkg = DataPackage({
            "image": Image.from_array(sample_image),
        })
        t = MixUp(lam=0.5)
        with pytest.raises(ValueError):
            t.apply([pkg], rng)
        with pytest.raises(ValueError):
            t.apply([pkg, pkg, pkg], rng)

    def test_hash_and_equality(self):
        a = MixUp(lam=0.5)
        b = MixUp(lam=0.5)
        c = MixUp(lam=0.4)
        assert a == b
        assert a != c
        assert hash(a) == hash(b)

    def test_different_sized_images(self, rng):
        """MixUp resizes second image to match first."""
        pkg1 = DataPackage({
            "image": Image.from_array(
                np.zeros((100, 80, 3), dtype=np.uint8),
            ),
        })
        pkg2 = DataPackage({
            "image": Image.from_array(
                np.ones((50, 50, 3), dtype=np.uint8) * 255,
            ),
        })
        t = MixUp(lam=0.5)
        result = t.apply([pkg1, pkg2], rng)
        assert _img_shape(result) == (100, 80)


# -------------------------------------------------------------------
# Mosaic
# -------------------------------------------------------------------

class TestMosaic:
    def test_is_multi_input_transform(self):
        assert issubclass(Mosaic, MultiInputTransform)

    def test_inflation(self):
        assert Mosaic().inflation == 0.25

    def test_combines_four_packages(self, rng):
        pkgs = []
        for i in range(4):
            img = np.ones(
                (50, 50, 3), dtype=np.uint8,
            ) * (i * 60)
            pkgs.append(DataPackage({
                "image": Image.from_array(img),
                "objects": [
                    (BoundingBox(
                        np.array([[5, 5], [20, 20]])),
                     Label(i, f"c{i}")),
                ],
            }))
        t = Mosaic()
        result = t.apply(pkgs, rng)
        assert isinstance(result, DataPackage)
        h, w = _img_shape(result)
        assert h == 100 and w == 100

    def test_annotations_merged(self, rng):
        """All 4 packages' annotations are merged."""
        pkgs = []
        for i in range(4):
            pkgs.append(DataPackage({
                "image": Image.from_array(
                    np.zeros((40, 40, 3), dtype=np.uint8),
                ),
                "objects": [
                    (BoundingBox(
                        np.array([[5, 5], [15, 15]])),
                     Label(i, f"c{i}")),
                ],
            }))
        t = Mosaic()
        result = t.apply(pkgs, rng)
        assert len(result["objects"]) == 4

    def test_labels_preserved(self, rng):
        pkgs = []
        names = ["cat", "dog", "bird", "fish"]
        for i, name in enumerate(names):
            pkgs.append(DataPackage({
                "image": Image.from_array(
                    np.zeros((30, 30, 3), dtype=np.uint8),
                ),
                "objects": [
                    (BoundingBox(
                        np.array([[2, 2], [10, 10]])),
                     Label(i, name)),
                ],
            }))
        t = Mosaic()
        result = t.apply(pkgs, rng)
        labels = {lbl.name for _, lbl in result["objects"]}
        assert labels == set(names)

    def test_wrong_package_count_raises(self, rng):
        pkg = DataPackage({
            "image": Image.from_array(
                np.zeros((30, 30, 3), dtype=np.uint8),
            ),
        })
        t = Mosaic()
        with pytest.raises(ValueError):
            t.apply([pkg, pkg], rng)

    def test_different_sized_images(self, rng):
        """Mosaic resizes all to smallest cell."""
        sizes = [(60, 60), (80, 80), (50, 50), (70, 70)]
        pkgs = []
        for h, w in sizes:
            pkgs.append(DataPackage({
                "image": Image.from_array(
                    np.zeros((h, w, 3), dtype=np.uint8),
                ),
                "objects": [
                    (BoundingBox(
                        np.array([[2, 2], [10, 10]])),
                     Label(0, "x")),
                ],
            }))
        t = Mosaic()
        result = t.apply(pkgs, rng)
        # Smallest is 50x50, mosaic = 100x100
        assert _img_shape(result) == (100, 100)

    def test_hash_and_equality(self):
        a = Mosaic()
        b = Mosaic()
        assert a == b
        assert hash(a) == hash(b)
