"""Shared helpers for spatial annotation transforms."""
from typing import Callable, List

from daugx.core.data.annotation import Annotation
from daugx.core.data.component import Component
from daugx.core.data.components.bounding_box import (
    ImageBoundingBox,
)
from daugx.core.data.components.keypoint import ImageKeyPoint
from daugx.core.data.components.polygon import ImagePolygon
from daugx.core.data.data_package import DataPackage

_SPATIAL_TYPES = (ImageBoundingBox, ImagePolygon, ImageKeyPoint)


def transform_annots(
    package: DataPackage,
    op: Callable,
    img_h: int,
    img_w: int,
) -> List[Component]:
    """Apply a spatial operation to all annotation components
    in a package.

    For each annotation component, transforms spatial types
    (ImageBoundingBox, ImagePolygon, ImageKeyPoint) using
    *op*, clips to image bounds, and filters out invalid
    results.  Non-spatial annotations are kept unchanged.

    Args:
        package: Source data package.
        op: Callable that takes a spatial component and
            returns a transformed component.
        img_h: New image height (for clipping).
        img_w: New image width (for clipping).

    Returns:
        List of transformed annotation components (invalid
        spatial ones removed).
    """
    result: List[Component] = []
    for comp in package.components:
        if not isinstance(comp, Annotation):
            continue
        if isinstance(comp, _SPATIAL_TYPES):
            transformed = _apply_and_clip(
                comp, op, img_h, img_w,
            )
            if _is_valid(transformed, img_h, img_w):
                result.append(transformed)
        else:
            result.append(comp)
    return result


def _apply_and_clip(comp, op, img_h, img_w):
    """Apply op then clip to image bounds."""
    transformed = op(comp)
    return transformed.clip(0, 0, img_w, img_h)


def _is_valid(comp, img_h, img_w):
    """Check if a spatial component is still valid."""
    if isinstance(comp, (ImageBoundingBox, ImagePolygon)):
        return comp.is_valid()
    if isinstance(comp, ImageKeyPoint):
        return comp.is_valid(0, 0, img_w, img_h)
    return True
