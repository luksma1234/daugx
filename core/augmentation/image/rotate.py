"""Rotate augmentation — rotate image around its center."""
from typing import Optional

import cv2
import numpy as np

from daugx.core.augmentation.base import Transform
from daugx.core.augmentation.image._spatial import (
    transform_annots,
)
from daugx.core.data.components.image import Image
from daugx.core.data.data_package import DataPackage


class Rotate(Transform):
    """Rotate image around its center.

    Positive angle rotates clockwise; negative rotates
    counterclockwise.  Image dimensions are preserved
    (corners may be cropped).

    Args:
        angle: Rotation angle in degrees.
    """

    def __init__(self, angle: float) -> None:
        self.angle = angle

    def _key(self) -> tuple:
        return (type(self).__name__, self.angle)

    def apply(
        self,
        package: DataPackage,
        rng: Optional[np.random.Generator] = None,
    ) -> DataPackage:
        """Apply rotation to image and annotations.

        Args:
            package: Input data package.
            rng: Unused (deterministic transform).

        Returns:
            New DataPackage with rotated contents.
        """
        pixels = package["image"].data
        h, w = pixels.shape[:2]
        center = ((w - 1) / 2.0, (h - 1) / 2.0)
        mat = cv2.getRotationMatrix2D(
            center, self.angle, 1,
        )
        rotated = cv2.warpAffine(pixels, mat, (w, h))
        new_img = Image.from_array(rotated)
        img_center = np.array(
            [center[0], center[1]],
        )

        def op(comp):
            return comp.rotate(self.angle, img_center)

        annots = transform_annots(
            package, op, h, w,
        )
        return package.replace(image=new_img, **annots)
