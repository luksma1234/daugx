"""Shared helpers for spatial annotation transforms."""
from typing import Any, Callable, List, Tuple

import numpy as np

from daugx.core.data.components.bounding_box import (
    BoundingBox,
)
from daugx.core.data.components.polygon import Polygon
from daugx.core.data.components.keypoint import KeyPoint
from daugx.core.data.data_package import DataPackage
from daugx.core.data.components.image import Image

_SPATIAL_TYPES = (BoundingBox, Polygon, KeyPoint)


def transform_annots(
    package: DataPackage,
    op: Callable,
    img_h: int,
    img_w: int,
) -> dict:
    """Apply a spatial operation to all annotation lists
    in a package.

    Args:
        package: Source data package.
        op: Callable that takes a spatial component and
            returns a transformed component.
        img_h: New image height (for clipping).
        img_w: New image width (for clipping).

    Returns:
        Dict of key -> transformed annotation list, for
        keys that contain annotation lists.
    """
    replacements: dict = {}
    for key in package.keys:
        val = package[key]
        if not isinstance(val, list):
            continue
        new_list: List[Tuple[Any, ...]] = []
        for tup in val:
            new_tup = tuple(
                _apply_and_clip(comp, op, img_h, img_w)
                if isinstance(comp, _SPATIAL_TYPES)
                else comp
                for comp in tup
            )
            if _all_valid(new_tup, img_h, img_w):
                new_list.append(new_tup)
        replacements[key] = new_list
    return replacements


def _apply_and_clip(comp, op, img_h, img_w):
    """Apply op then clip to image bounds."""
    transformed = op(comp)
    return transformed.clip(0, 0, img_w, img_h)


def _all_valid(tup, img_h, img_w):
    """Check all spatial components in a tuple are valid."""
    for comp in tup:
        if isinstance(comp, (BoundingBox, Polygon)):
            if not comp.is_valid():
                return False
        elif isinstance(comp, KeyPoint):
            if not comp.is_valid(0, 0, img_w, img_h):
                return False
    return True
