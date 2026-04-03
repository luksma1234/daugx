"""Resize augmentation — resize image to target dims."""
from typing import Callable, Optional, Tuple, Type

import cv2
import numpy as np

from daugx.core.augmentation.base import Transform
from daugx.core.augmentation.image._spatial import (
    _IMAGE_SPATIAL_OPS,
    apply_and_clip,
    is_valid,
)
from daugx.core.data.component import Component
from daugx.core.data.components.image import Image


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

    operates_on: Tuple[Type[Component], ...] = _IMAGE_SPATIAL_OPS

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

    def _apply(
        self,
        component: Component,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[Component]:
        if isinstance(component, Image):
            return self._apply_image(component, rng)
        return self._apply_spatial(component)

    def _apply_image(
        self,
        img: Image,
        rng: Optional[np.random.Generator] = None,
    ) -> Image:
        pixels = img.data
        orig_h, orig_w = pixels.shape[:2]

        if not self.preserve_aspect_ratio:
            # cv2.resize dsize is (width, height)
            resized = cv2.resize(
                pixels,
                (self.width, self.height),
                interpolation=cv2.INTER_LINEAR,
            )
            sx = self.width / orig_w
            sy = self.height / orig_h
            self._op: Callable = lambda c: c.scale(sx, sy)
            self._out_h = self.height
            self._out_w = self.width
            return Image.from_array(resized, name=img.name)

        # Preserve aspect ratio: pad then resize.
        fx = self.width / orig_w
        fy = self.height / orig_h
        if fx < fy:
            # Width is the binding dimension — pad top/bot
            scale = fx
            new_w = self.width
            new_h = int(orig_h * scale)
            pad = self.height - new_h
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
            _scale = scale
            _pad_top = pad_top

            def _op(c, s=_scale, pt=_pad_top):
                return c.scale(s, s).shift(0, pt)

            self._op = _op
        else:
            # Height is the binding dimension — pad sides
            scale = fy
            new_h = self.height
            new_w = int(orig_w * scale)
            pad = self.width - new_w
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
            _scale = scale
            _pad_left = pad_left

            def _op(c, s=_scale, pl=_pad_left):
                return c.scale(s, s).shift(pl, 0)

            self._op = _op

        self._out_h = self.height
        self._out_w = self.width
        return Image.from_array(resized, name=img.name)

    def _apply_spatial(
        self, comp: Component,
    ) -> Optional[Component]:
        result = apply_and_clip(
            comp, self._op, self._out_h, self._out_w,
        )
        return (
            result
            if is_valid(result, self._out_h, self._out_w)
            else None
        )
