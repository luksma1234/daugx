"""Scale augmentation — resize image by a factor."""
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


class Scale(Transform):
    """Scale image by ``(x_scale, y_scale)`` factors.

    Values > 1 enlarge the image; values < 1 shrink it.

    Args:
        x_scale: Horizontal scale factor (> 0).
        y_scale: Vertical scale factor (> 0).
    """

    operates_on: Tuple[Type[Component], ...] = _IMAGE_SPATIAL_OPS

    def __init__(
        self,
        x_scale: float = 1,
        y_scale: float = 1,
    ) -> None:
        if x_scale <= 0 or y_scale <= 0:
            raise ValueError("Scale factors must be > 0.")
        self.x_scale = x_scale
        self.y_scale = y_scale

    def _key(self) -> tuple:
        return (
            type(self).__name__,
            self.x_scale,
            self.y_scale,
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
        scaled = cv2.resize(
            pixels,
            None,
            fx=self.x_scale,
            fy=self.y_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        self._h, self._w = scaled.shape[:2]
        return Image.from_array(scaled, name=img.name)

    def _apply_spatial(
        self, comp: Component,
    ) -> Optional[Component]:
        def op(c):
            return c.scale(self.x_scale, self.y_scale)

        result = apply_and_clip(comp, op, self._h, self._w)
        return result if is_valid(result, self._h, self._w) else None
