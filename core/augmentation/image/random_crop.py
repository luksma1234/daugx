"""Random crop augmentation — randomly crop a sub-region."""
from typing import Optional

import numpy as np

from daugx.core.augmentation.base import Transform
from daugx.core.augmentation.image.crop import Crop
from daugx.core.data.data_package import DataPackage


class RandomCrop(Transform):
    """Crop a random sub-region of the image.

    The crop dimensions are sampled uniformly within the
    specified min/max fractions of the original size.

    Args:
        min_width: Minimum crop width as fraction of
            image width.
        max_width: Maximum crop width as fraction of
            image width.
        min_height: Minimum crop height as fraction of
            image height.
        max_height: Maximum crop height as fraction of
            image height.
    """

    def __init__(
        self,
        min_width: float = 0.2,
        max_width: float = 1.0,
        min_height: float = 0.2,
        max_height: float = 1.0,
    ) -> None:
        if not (0 < min_width < max_width <= 1):
            raise ValueError("Invalid width range.")
        if not (0 < min_height < max_height <= 1):
            raise ValueError("Invalid height range.")
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height

    def _key(self) -> tuple:
        return (
            type(self).__name__,
            self.min_width,
            self.max_width,
            self.min_height,
            self.max_height,
        )

    def apply(
        self,
        package: DataPackage,
        rng: Optional[np.random.Generator] = None,
    ) -> DataPackage:
        """Apply a random crop.

        Args:
            package: Input data package.
            rng: Required random number generator.

        Returns:
            New DataPackage with cropped contents.

        Raises:
            ValueError: If *rng* is None.
        """
        if rng is None:
            raise ValueError("rng is required.")

        # Sample crop size as fractions
        crop_w = (
            self.min_width
            + rng.random()
            * (self.max_width - self.min_width)
        )
        crop_h = (
            self.min_height
            + rng.random()
            * (self.max_height - self.min_height)
        )
        # Sample top-left corner
        max_x = 1.0 - crop_w
        max_y = 1.0 - crop_h
        x_min = rng.random() * max_x if max_x > 0 else 0
        y_min = rng.random() * max_y if max_y > 0 else 0
        x_max = x_min + crop_w
        y_max = y_min + crop_h

        # Clamp to valid range
        x_min = max(x_min, 1e-6)
        y_min = max(y_min, 1e-6)
        x_max = min(x_max, 1.0)
        y_max = min(y_max, 1.0)

        cropper = Crop(x_min, y_min, x_max, y_max)
        return cropper.apply(package, rng)
