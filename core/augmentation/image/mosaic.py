"""Mosaic augmentation — 2x2 grid of four images."""
from typing import List, Optional

import cv2
import numpy as np

from daugx.core.augmentation.base import MultiInputTransform
from daugx.core.augmentation.image._spatial import (
    _SPATIAL_TYPES,
)
from daugx.core.data.component import Component
from daugx.core.data.components.image import Image
from daugx.core.data.sample import Sample


class Mosaic(MultiInputTransform):
    """Combine four images into a 2x2 mosaic grid.

    All images are resized to the smallest input size
    before stitching.  The output is twice the size of
    each cell.  Annotations are spatially transformed
    to their respective positions in the mosaic.
    Non-Image, non-Annotation components from all
    packages are preserved.

    Layout::

        +------+------+
        | img0 | img3 |
        +------+------+
        | img1 | img2 |
        +------+------+
    """

    inflation = 0.25

    def __init__(self) -> None:
        pass

    def _key(self) -> tuple:
        return (type(self).__name__,)

    def apply(
        self,
        samples: List[Sample],
        rng: Optional[np.random.Generator] = None,
    ) -> Sample:
        """Stitch four samples into a mosaic.

        Args:
            samples: Exactly four materialized Samples.
            rng: Unused.

        Returns:
            A new Sample with the mosaic image, spatially
            adjusted annotations, and all other components
            preserved.

        Raises:
            ValueError: If not exactly 4 samples.
        """
        if len(samples) != 4:
            raise ValueError(
                f"Mosaic needs exactly 4 samples, "
                f"got {len(samples)}"
            )

        # Find smallest cell size
        shapes = [
            s.get(Image).data.shape[:2]
            for s in samples
        ]
        cell_h = min(s[0] for s in shapes)
        cell_w = min(s[1] for s in shapes)

        # Resize each image to cell size
        cells = []
        for s in samples:
            img = s.get(Image).data
            if img.shape[:2] != (cell_h, cell_w):
                img = cv2.resize(
                    img,
                    (cell_w, cell_h),
                    interpolation=cv2.INTER_LINEAR,
                )
            cells.append(img)

        # Stitch: [0|3] over [1|2]
        top = np.hstack([cells[0], cells[3]])
        bot = np.hstack([cells[1], cells[2]])
        mosaic = np.vstack([top, bot])
        out_h, out_w = mosaic.shape[:2]
        new_img = Image.from_array(mosaic)

        # Merge annotation components with offsets
        offsets = [
            (0, 0),
            (0, cell_h),
            (cell_w, cell_h),
            (cell_w, 0),
        ]
        # Scale factors for each sub-image to cell size
        scales = [
            (cell_w / s[1], cell_h / s[0])
            for s in shapes
        ]

        result: List[Component] = [new_img]
        for idx, s in enumerate(samples):
            sx, sy = scales[idx]
            dx, dy = offsets[idx]
            for comp in s.components:
                if isinstance(comp, Image):
                    continue
                if hasattr(comp, 'applies_to'):
                    if isinstance(comp, _SPATIAL_TYPES):
                        result.append(
                            _scale_shift_clip(
                                comp, sx, sy, dx, dy,
                                out_h, out_w,
                            ),
                        )
                    else:
                        result.append(comp)
                else:
                    result.append(comp)

        return Sample(*result)


def _scale_shift_clip(comp, sx, sy, dx, dy, h, w):
    """Scale to cell, shift to position, clip to mosaic."""
    return comp.scale(sx, sy).shift(dx, dy).clip(
        0, 0, w, h,
    )
