"""Tests for InvalidComponentError and Sample validation."""
import numpy as np
import pytest

import daugx
from daugx.errors import InvalidComponentError


class TestOrphanedAnnotation:
    def test_orphaned_target_raises(self):
        """Annotation with target that matches no component name."""
        with pytest.raises(InvalidComponentError):
            daugx.Sample(
                daugx.Image(path="img.jpg"),
                daugx.ImageBoundingBox(
                    np.array([[0, 0], [10, 10]]),
                    target="nonexistent",
                ),
            )

    def test_valid_target_does_not_raise(self):
        """Annotation whose target matches the Image name."""
        daugx.Sample(
            daugx.Image(path="img.jpg", name="cam"),
            daugx.ImageBoundingBox(
                np.array([[0, 0], [10, 10]]),
                target="cam",
            ),
        )

    def test_multiple_valid_targets_do_not_raise(self):
        """Two annotations each targeting their own Image."""
        daugx.Sample(
            daugx.Image(path="a.jpg", name="left"),
            daugx.Image(path="b.jpg", name="right"),
            daugx.ImageBoundingBox(
                np.array([[0, 0], [10, 10]]),
                target="left",
            ),
            daugx.ImageBoundingBox(
                np.array([[5, 5], [20, 20]]),
                target="right",
            ),
        )

    def test_target_none_single_image_does_not_raise(self):
        """target=None with one Image is unambiguous."""
        daugx.Sample(
            daugx.Image(path="img.jpg"),
            daugx.ImageBoundingBox(np.array([[0, 0], [10, 10]])),
        )

    def test_orphaned_polygon_raises(self):
        with pytest.raises(InvalidComponentError):
            daugx.Sample(
                daugx.Image(path="img.jpg"),
                daugx.ImagePolygon(
                    np.array([[0, 0], [5, 0], [5, 5]]),
                    target="missing",
                ),
            )

    def test_orphaned_keypoint_raises(self):
        with pytest.raises(InvalidComponentError):
            daugx.Sample(
                daugx.Image(path="img.jpg"),
                daugx.ImageKeyPoint(x=1, y=2, target="gone"),
            )

    def test_orphaned_category_raises(self):
        with pytest.raises(InvalidComponentError):
            daugx.Sample(
                daugx.Image(path="img.jpg"),
                daugx.ImageCategory(class_id=0, target="nowhere"),
            )


class TestAmbiguousAnnotation:
    def test_target_none_two_images_raises(self):
        """Ambiguous: target=None but two Images exist."""
        with pytest.raises(InvalidComponentError):
            daugx.Sample(
                daugx.Image(path="a.jpg", name="left"),
                daugx.Image(path="b.jpg", name="right"),
                daugx.ImageBoundingBox(
                    np.array([[0, 0], [10, 10]]),
                    # target=None — ambiguous
                ),
            )

    def test_target_none_polygon_two_images_raises(self):
        with pytest.raises(InvalidComponentError):
            daugx.Sample(
                daugx.Image(path="a.jpg", name="cam0"),
                daugx.Image(path="b.jpg", name="cam1"),
                daugx.ImagePolygon(
                    np.array([[0, 0], [5, 0], [5, 5]]),
                ),
            )

    def test_target_none_category_two_images_raises(self):
        with pytest.raises(InvalidComponentError):
            daugx.Sample(
                daugx.Image(path="a.jpg", name="x"),
                daugx.Image(path="b.jpg", name="y"),
                daugx.ImageCategory(class_id=1),
            )

    def test_explicit_target_two_images_no_raise(self):
        """Explicit target resolves ambiguity."""
        daugx.Sample(
            daugx.Image(path="a.jpg", name="left"),
            daugx.Image(path="b.jpg", name="right"),
            daugx.ImageBoundingBox(
                np.array([[0, 0], [10, 10]]),
                target="left",
            ),
        )

    def test_target_none_no_images_does_not_raise(self):
        """No Image components — no ambiguity possible."""
        daugx.Sample(
            daugx.Text(text="hello"),
            daugx.Constant(value=42, name="id"),
        )


class TestErrorMessages:
    def test_orphan_error_message_contains_target(self):
        with pytest.raises(InvalidComponentError, match="nonexistent"):
            daugx.Sample(
                daugx.Image(path="img.jpg"),
                daugx.ImageBoundingBox(
                    np.array([[0, 0], [10, 10]]),
                    target="nonexistent",
                ),
            )

    def test_ambiguous_error_message_mentions_ambiguous(self):
        with pytest.raises(
            InvalidComponentError, match="[Aa]mbiguous"
        ):
            daugx.Sample(
                daugx.Image(path="a.jpg", name="x"),
                daugx.Image(path="b.jpg", name="y"),
                daugx.ImageBoundingBox(
                    np.array([[0, 0], [10, 10]]),
                ),
            )
