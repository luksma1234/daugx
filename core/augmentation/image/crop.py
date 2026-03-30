"""Crop augmentation — extract a sub-region of the image."""
from typing import Optional

import numpy as np

from daugx.core.augmentation.base import Transform
from daugx.core.augmentation.image._spatial import (
    transform_annots,
)
from daugx.core.data.components.image import Image
from daugx.core.data.data_package import DataPackage


class Crop(Transform):
    """Crop an image to a percentage-based region.

    Boundaries are specified as fractions of the image
    dimensions in the range ``(0, 1]``.

    Args:
        x_min: Left boundary (fraction of width).
        y_min: Top boundary (fraction of height).
        x_max: Right boundary (fraction of width).
        y_max: Bottom boundary (fraction of height).
    """

    def __init__(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
    ) -> None:
        if not (0 < x_min < x_max <= 1):
            raise ValueError(
                f"Invalid x bounds: {x_min}, {x_max}"
            )
        if not (0 < y_min < y_max <= 1):
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

    def apply(
        self,
        package: DataPackage,
        rng: Optional[np.random.Generator] = None,
    ) -> DataPackage:
        """Crop image and adjust annotations.

        Args:
            package: Input data package.
            rng: Unused (deterministic transform).

        Returns:
            New DataPackage with cropped contents.
        """
        img = package.get(Image)
        pixels = img.data
        h, w = pixels.shape[:2]
        x0 = int(w * self.x_min)
        y0 = int(h * self.y_min)
        x1 = int(w * self.x_max)
        y1 = int(h * self.y_max)
        cropped = pixels[y0:y1, x0:x1, :]
        new_h, new_w = cropped.shape[:2]
        new_img = Image.from_array(
            cropped, name=img.name,
        )

        def op(comp):
            return comp.shift(-x0, -y0)

        annots = transform_annots(
            package, op, new_h, new_w,
        )
        return DataPackage(new_img, *annots)
