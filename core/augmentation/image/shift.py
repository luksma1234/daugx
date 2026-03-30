"""Shift augmentation — translates image pixels."""
from typing import Optional

import cv2
import numpy as np

from daugx.core.augmentation.base import Transform
from daugx.core.augmentation.image._spatial import (
    transform_annots,
)
from daugx.core.data.components.image import Image
from daugx.core.data.data_package import DataPackage


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

    def apply(
        self,
        package: DataPackage,
        rng: Optional[np.random.Generator] = None,
    ) -> DataPackage:
        """Apply shift to image and annotations.

        Args:
            package: Input data package.
            rng: Unused (deterministic transform).

        Returns:
            New DataPackage with shifted contents.
        """
        img = package.get(Image)
        pixels = img.data
        h, w = pixels.shape[:2]
        affine = np.float32([
            [1, 0, self.x_shift],
            [0, 1, self.y_shift],
        ])
        shifted = cv2.warpAffine(pixels, affine, (w, h))
        new_img = Image.from_array(
            shifted, name=img.name,
        )

        def op(comp):
            return comp.shift(self.x_shift, self.y_shift)

        annots = transform_annots(package, op, h, w)
        return DataPackage(new_img, *annots)
