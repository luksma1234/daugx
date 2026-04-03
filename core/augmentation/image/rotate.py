"""Rotate augmentation — rotate image around its center."""
from typing import Optional, Tuple, Type

import cv2
import numpy as np

from daugx.core.augmentation.base import Transform
from daugx.core.augmentation.image._spatial import (
    _IMAGE_SPATIAL_OPS,
    apply_and_clip,
    is_valid,
)
from daugx.core.data.component import Component
from daugx.core.data.components.image import Image


class Rotate(Transform):
    """Rotate image around its center.

    Positive angle rotates clockwise; negative rotates
    counterclockwise.  Image dimensions are preserved
    (corners may be cropped).

    Args:
        angle: Rotation angle in degrees.
    """

    operates_on: Tuple[Type[Component], ...] = _IMAGE_SPATIAL_OPS

    def __init__(self, angle: float) -> None:
        self.angle = angle

    def _key(self) -> tuple:
        return (type(self).__name__, self.angle)

    def _apply(
        self,
        component: Component,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[Component]:
        if isinstance(component, Image):
            return self._apply_image(component, rng)
        return self._apply_spatial(component)

    def _apply_image(
        self,
        img: Image,
        rng: Optional[np.random.Generator] = None,
    ) -> Image:
        pixels = img.data
        h, w = pixels.shape[:2]
        center = ((w - 1) / 2.0, (h - 1) / 2.0)
        mat = cv2.getRotationMatrix2D(
            center, self.angle, 1,
        )
        rotated = cv2.warpAffine(pixels, mat, (w, h))
        self._h = h
        self._w = w
        self._center = np.array([center[0], center[1]])
        return Image.from_array(rotated, name=img.name)

    def _apply_spatial(
        self, comp: Component,
    ) -> Optional[Component]:
        def op(c):
            return c.rotate(self.angle, self._center)

        result = apply_and_clip(comp, op, self._h, self._w)
        return result if is_valid(result, self._h, self._w) else None
