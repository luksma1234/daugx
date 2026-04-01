"""Crop augmentation — extract a sub-region of the image."""
from typing import Optional, Tuple, Type

import numpy as np

from daugx.core.augmentation.base import Transform
from daugx.core.augmentation.image._spatial import (
    _IMAGE_SPATIAL_OPS,
    apply_and_clip,
    is_valid,
)
from daugx.core.data.component import Component
from daugx.core.data.components.image import Image


class Crop(Transform):
    """Crop an image to a percentage-based region.

    Boundaries are specified as fractions of the image
    dimensions in the range ``[0, 1]``.

    Args:
        x_min: Left boundary (fraction of width).
        y_min: Top boundary (fraction of height).
        x_max: Right boundary (fraction of width).
        y_max: Bottom boundary (fraction of height).
    """

    operates_on: Tuple[Type[Component], ...] = _IMAGE_SPATIAL_OPS

    def __init__(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
    ) -> None:
        if not (0 <= x_min < x_max <= 1):
            raise ValueError(
                f"Invalid x bounds: {x_min}, {x_max}"
            )
        if not (0 <= y_min < y_max <= 1):
            raise ValueError(
                f"Invalid y bounds: {y_min}, {y_max}"
            )
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def _key(self) -> tuple:
        return (
            type(self).__name__,
            self.x_min,
            self.y_min,
            self.x_max,
            self.y_max,
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
        x0 = int(w * self.x_min)
        y0 = int(h * self.y_min)
        x1 = int(w * self.x_max)
        y1 = int(h * self.y_max)
        cropped = pixels[y0:y1, x0:x1, :]
        self._h, self._w = cropped.shape[:2]
        self._x0 = x0
        self._y0 = y0
        return Image.from_array(cropped, name=img.name)

    def _apply_spatial(
        self, comp: Component,
    ) -> Optional[Component]:
        def op(c):
            return c.shift(-self._x0, -self._y0)

        result = apply_and_clip(comp, op, self._h, self._w)
        return result if is_valid(result, self._h, self._w) else None
