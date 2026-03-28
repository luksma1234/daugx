"""Mosaic augmentation — 2x2 grid of four images."""
from typing import List, Optional

import cv2
import numpy as np

from daugx.core.augmentation.base import MultiInputTransform
from daugx.core.augmentation.image._spatial import (
    transform_annots,
)
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
            p["image"].data.shape[:2] for p in packages
        ]
        cell_h = min(s[0] for s in shapes)
        cell_w = min(s[1] for s in shapes)

        # Resize each image to cell size
        cells = []
        for pkg in packages:
            img = pkg["image"].data
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

        # Merge annotations with offsets
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

        all_annots: dict = {}
        for idx, pkg in enumerate(packages):
            sx, sy = scales[idx]
            dx, dy = offsets[idx]
            for key in pkg.keys:
                if key == "image":
                    continue
                val = pkg[key]
                if not isinstance(val, list):
                    continue
                if key not in all_annots:
                    all_annots[key] = []
                for tup in val:
                    new_tup = tuple(
                        _scale_shift_clip(
                            comp, sx, sy, dx, dy,
                            out_h, out_w,
                        )
                        if hasattr(comp, "scale")
                        else comp
                        for comp in tup
                    )
                    all_annots[key].append(new_tup)

        return DataPackage(
            {"image": new_img, **all_annots},
        )


def _scale_shift_clip(comp, sx, sy, dx, dy, h, w):
    """Scale to cell, shift to position, clip to mosaic."""
    return comp.scale(sx, sy).shift(dx, dy).clip(
        0, 0, w, h,
    )
