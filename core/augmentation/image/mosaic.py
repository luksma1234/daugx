"""Mosaic augmentation — 2x2 grid of four images."""
from typing import List, Optional

import cv2
import numpy as np

from daugx.core.augmentation.base import MultiInputTransform
from daugx.core.augmentation.image._spatial import (
    _SPATIAL_TYPES,
)
from daugx.core.data.annotation import Annotation
from daugx.core.data.component import Component
from daugx.core.data.components.image import Image
from daugx.core.data.data_package import DataPackage


class Mosaic(MultiInputTransform):
    """Combine four images into a 2x2 mosaic grid.

    All images are resized to the smallest input size
    before stitching.  The output is twice the size of
    each cell.

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
        packages: List[DataPackage],
        rng: Optional[np.random.Generator] = None,
    ) -> DataPackage:
        """Stitch four packages into a mosaic.

        Args:
            packages: Exactly four DataPackages.
            rng: Unused.

        Returns:
            A new DataPackage with the mosaic image.

        Raises:
            ValueError: If not exactly 4 packages.
        """
        if len(packages) != 4:
            raise ValueError(
                f"Mosaic needs exactly 4 packages, "
                f"got {len(packages)}"
            )

        # Find smallest cell size
        shapes = [
            p.get(Image).data.shape[:2]
            for p in packages
        ]
        cell_h = min(s[0] for s in shapes)
        cell_w = min(s[1] for s in shapes)

        # Resize each image to cell size
        cells = []
        for pkg in packages:
            img = pkg.get(Image).data
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

        all_annots: List[Component] = []
        for idx, pkg in enumerate(packages):
            sx, sy = scales[idx]
            dx, dy = offsets[idx]
            for comp in pkg.get_all(Annotation):
                if isinstance(comp, _SPATIAL_TYPES):
                    all_annots.append(
                        _scale_shift_clip(
                            comp, sx, sy, dx, dy,
                            out_h, out_w,
                        ),
                    )
                else:
                    all_annots.append(comp)

        return DataPackage(new_img, *all_annots)


def _scale_shift_clip(comp, sx, sy, dx, dy, h, w):
    """Scale to cell, shift to position, clip to mosaic."""
    return comp.scale(sx, sy).shift(dx, dy).clip(
        0, 0, w, h,
    )
