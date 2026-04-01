"""MixUp augmentation — blend two images.

Reference: https://arxiv.org/abs/1710.09412v2
"""
from typing import List, Optional

import cv2
import numpy as np

from daugx.core.augmentation.base import MultiInputTransform
from daugx.core.data.component import Component
from daugx.core.data.components.image import Image
from daugx.core.data.sample import Sample


class MixUp(MultiInputTransform):
    """Blend two images via convex combination.

    The blended image is
    ``lam * image_1 + (1 - lam) * image_2``.
    Annotations from both inputs are concatenated.
    Non-Image, non-Annotation components from both
    packages are preserved.

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
        samples: List[Sample],
        rng: Optional[np.random.Generator] = None,
    ) -> Sample:
        """Blend two samples.

        Args:
            samples: Exactly two materialized Samples.
            rng: Unused (deterministic given lam).

        Returns:
            A new Sample with blended image, concatenated
            annotations, and all other components
            preserved.

        Raises:
            ValueError: If not exactly 2 samples.
        """
        if len(samples) != 2:
            raise ValueError(
                f"MixUp needs exactly 2 samples, "
                f"got {len(samples)}"
            )
        s1, s2 = samples
        img1 = s1.get(Image).data
        img2 = s2.get(Image).data

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

        # Collect annotations and non-Image other components
        others: List[Component] = []
        for s in samples:
            for comp in s.components:
                if isinstance(comp, Image):
                    continue
                others.append(comp)

        return Sample(new_img, *others)
