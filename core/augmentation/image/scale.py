"""Scale augmentation — resize image by a factor."""
from typing import Optional

import cv2
import numpy as np

from daugx.core.augmentation.base import Transform
from daugx.core.augmentation.image._spatial import (
    transform_annots,
)
from daugx.core.data.components.image import Image
from daugx.core.data.data_package import DataPackage


class Scale(Transform):
    """Scale image by ``(x_scale, y_scale)`` factors.

    Values > 1 enlarge the image; values < 1 shrink it.

    Args:
        x_scale: Horizontal scale factor (> 0).
        y_scale: Vertical scale factor (> 0).
    """

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

    def apply(
        self,
        package: DataPackage,
        rng: Optional[np.random.Generator] = None,
    ) -> DataPackage:
        """Apply scaling to image and annotations.

        Args:
            package: Input data package.
            rng: Unused (deterministic transform).

        Returns:
            New DataPackage with scaled contents.
        """
        pixels = package["image"].data
        scaled = cv2.resize(
            pixels,
            None,
            fx=self.x_scale,
            fy=self.y_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        new_h, new_w = scaled.shape[:2]
        new_img = Image.from_array(scaled)

        def op(comp):
            return comp.scale(self.x_scale, self.y_scale)

        annots = transform_annots(
            package, op, new_h, new_w,
        )
        return package.replace(image=new_img, **annots)
