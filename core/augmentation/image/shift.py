"""Shift augmentation — translates image pixels."""
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


class Shift(Transform):
    """Shift image pixels by ``(x_shift, y_shift)``.

    Pixels shifted outside the image boundary are lost.
    New pixels are filled with black.

    Args:
        x_shift: Horizontal shift in pixels
            (positive = right).
        y_shift: Vertical shift in pixels
            (positive = down).
    """

    operates_on: Tuple[Type[Component], ...] = _IMAGE_SPATIAL_OPS

    def __init__(
        self,
        x_shift: float = 0,
        y_shift: float = 0,
    ) -> None:
        self.x_shift = x_shift
        self.y_shift = y_shift

    def _key(self) -> tuple:
        return (
            type(self).__name__,
            self.x_shift,
            self.y_shift,
        )

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
        affine = np.float32([
            [1, 0, self.x_shift],
            [0, 1, self.y_shift],
        ])
        shifted = cv2.warpAffine(pixels, affine, (w, h))
        self._h = h
        self._w = w
        return Image.from_array(shifted, name=img.name)

    def _apply_spatial(
        self, comp: Component,
    ) -> Optional[Component]:
        def op(c):
            return c.shift(self.x_shift, self.y_shift)

        result = apply_and_clip(comp, op, self._h, self._w)
        return result if is_valid(result, self._h, self._w) else None
