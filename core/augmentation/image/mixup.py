"""MixUp augmentation — blend two images.

Reference: https://arxiv.org/abs/1710.09412v2
"""
from typing import List, Optional

import cv2
import numpy as np

from daugx.core.augmentation.base import MultiInputTransform
from daugx.core.data.annotation import Annotation
from daugx.core.data.component import Component
from daugx.core.data.components.image import Image
from daugx.core.data.data_package import DataPackage


class MixUp(MultiInputTransform):
    """Blend two images via convex combination.

    The blended image is
    ``lam * image_1 + (1 - lam) * image_2``.
    Annotations from both inputs are concatenated.

    Args:
        lam: Blending weight in ``[0.4, 0.6]``.
    """

    inflation = 0.5

    def __init__(self, lam: float) -> None:
        if not 0.4 <= lam <= 0.6:
            raise ValueError(
                f"lam must be in [0.4, 0.6], got {lam}"
            )
        self.lam = lam

    def _key(self) -> tuple:
        return (type(self).__name__, self.lam)

    def apply(
        self,
        packages: List[DataPackage],
        rng: Optional[np.random.Generator] = None,
    ) -> DataPackage:
        """Blend two data packages.

        Args:
            packages: Exactly two DataPackages.
            rng: Unused (deterministic given lam).

        Returns:
            A new DataPackage with blended image and
            concatenated annotations.

        Raises:
            ValueError: If not exactly 2 packages.
        """
        if len(packages) != 2:
            raise ValueError(
                f"MixUp needs exactly 2 packages, "
                f"got {len(packages)}"
            )
        pkg1, pkg2 = packages
        img1 = pkg1.get(Image).data
        img2 = pkg2.get(Image).data

        # Resize img2 to match img1 if needed
        if img1.shape != img2.shape:
            img2 = cv2.resize(
                img2,
                (img1.shape[1], img1.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        blended = np.array(
            img1 * self.lam + img2 * (1 - self.lam),
            dtype=np.uint8,
        )
        new_img = Image.from_array(blended)

        # Merge annotation components from both packages
        annots: List[Component] = (
            pkg1.get_all(Annotation)
            + pkg2.get_all(Annotation)
        )
        return DataPackage(new_img, *annots)
