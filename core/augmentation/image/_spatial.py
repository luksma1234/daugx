"""Shared helpers for spatial annotation transforms."""
from typing import Optional

from daugx.core.data.components.bounding_box import (
    ImageBoundingBox,
)
from daugx.core.data.components.image import Image
from daugx.core.data.components.keypoint import ImageKeyPoint
from daugx.core.data.components.polygon import ImagePolygon

_SPATIAL_TYPES = (ImageBoundingBox, ImagePolygon, ImageKeyPoint)

# Convenience tuple for image transforms that handle both
# the Image component and all spatial annotation types.
_IMAGE_SPATIAL_OPS = (Image,) + _SPATIAL_TYPES


def apply_and_clip(comp, op, img_h: int, img_w: int):
    """Apply *op* to a spatial annotation then clip to
    image bounds.

    Args:
        comp: Spatial annotation component.
        op: Callable that takes a component and returns
            a transformed component.
        img_h: Image height (pixels).
        img_w: Image width (pixels).

    Returns:
        Clipped annotation.
    """
    transformed = op(comp)
    return transformed.clip(0, 0, img_w, img_h)


def is_valid(
    comp, img_h: int, img_w: int, min_area: float = 0,
) -> bool:
    """Return ``True`` if *comp* is still a valid
    annotation within the image bounds.

    Args:
        comp: Spatial annotation to check.
        img_h: Image height.
        img_w: Image width.
        min_area: Minimum required area for bbox/polygon.
    """
    if isinstance(comp, (ImageBoundingBox, ImagePolygon)):
        return comp.is_valid(min_area)
    if isinstance(comp, ImageKeyPoint):
        return comp.is_valid(0, 0, img_w, img_h)
    return True
