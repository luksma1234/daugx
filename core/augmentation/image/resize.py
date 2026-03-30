"""Resize augmentation — resize image to target dims."""
from typing import Optional

import cv2
import numpy as np

from daugx.core.augmentation.base import Transform
from daugx.core.augmentation.image._spatial import (
    transform_annots,
)
from daugx.core.data.components.image import Image
from daugx.core.data.data_package import DataPackage


class Resize(Transform):
    """Resize image to exact ``(width, height)``.

    When ``preserve_aspect_ratio`` is True the image is
    padded with black to match the target aspect ratio
    before scaling.

    Args:
        width: Target width in pixels.
        height: Target height in pixels.
        preserve_aspect_ratio: Whether to preserve the
            original aspect ratio via padding.
    """

    def __init__(
        self,
        width: int,
        height: int,
        preserve_aspect_ratio: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(
                "Width and height must be > 0."
            )
        self.width = width
        self.height = height
        self.preserve_aspect_ratio = preserve_aspect_ratio

    def _key(self) -> tuple:
        return (
            type(self).__name__,
            self.width,
            self.height,
            self.preserve_aspect_ratio,
        )

    def apply(
        self,
        package: DataPackage,
        rng: Optional[np.random.Generator] = None,
    ) -> DataPackage:
        """Resize image and scale annotations.

        Args:
            package: Input data package.
            rng: Unused (deterministic transform).

        Returns:
            New DataPackage with resized contents.
        """
        img = package.get(Image)
        pixels = img.data
        orig_h, orig_w = pixels.shape[:2]

        if not self.preserve_aspect_ratio:
            resized = cv2.resize(
                pixels,
                (self.height, self.width),
                interpolation=cv2.INTER_LINEAR,
            )
            sx = self.height / orig_w
            sy = self.width / orig_h
            new_img = Image.from_array(
                resized, name=img.name,
            )

            def op(comp):
                return comp.scale(sx, sy)

            annots = transform_annots(
                package, op, self.width, self.height,
            )
            return DataPackage(new_img, *annots)

        # Preserve aspect ratio: pad then resize
        fy = self.width / orig_h
        fx = self.height / orig_w
        if fx < fy:
            # Width is the binding dimension — pad top/bot
            scale = fx
            new_w = self.height
            new_h = int(orig_h * scale)
            pad = self.width - new_h
            pad_top = pad // 2
            pad_bot = pad - pad_top
            resized = cv2.resize(
                pixels,
                (new_w, new_h),
                interpolation=cv2.INTER_LINEAR,
            )
            if pad > 0:
                resized = np.vstack([
                    np.zeros(
                        (pad_top, new_w, 3),
                        dtype=np.uint8,
                    ),
                    resized,
                    np.zeros(
                        (pad_bot, new_w, 3),
                        dtype=np.uint8,
                    ),
                ])

            def op(comp):
                return comp.scale(scale, scale).shift(
                    0, pad_top,
                )
        else:
            # Height is the binding dimension — pad sides
            scale = fy
            new_h = self.width
            new_w = int(orig_w * scale)
            pad = self.height - new_w
            pad_left = pad // 2
            pad_right = pad - pad_left
            resized = cv2.resize(
                pixels,
                (new_w, new_h),
                interpolation=cv2.INTER_LINEAR,
            )
            if pad > 0:
                resized = np.hstack([
                    np.zeros(
                        (new_h, pad_left, 3),
                        dtype=np.uint8,
                    ),
                    resized,
                    np.zeros(
                        (new_h, pad_right, 3),
                        dtype=np.uint8,
                    ),
                ])

            def op(comp):
                return comp.scale(scale, scale).shift(
                    pad_left, 0,
                )

        new_img = Image.from_array(
            resized, name=img.name,
        )
        annots = transform_annots(
            package, op, self.width, self.height,
        )
        return DataPackage(new_img, *annots)
