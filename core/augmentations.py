from typing import Optional

import numpy as np
import scipy as sp

from .annotations import Annotations
from .transforms import (
    SITransform,
    MITransform,
    IOTransform
)
from ..utils.mat_utils import get_2d_transf_mat


class Shift(SITransform):
    def __init__(
            self,
            x_shift: Optional[float] = None,
            y_shift: Optional[float] = None
    ) -> None:
        super().__init__()
        self.x_shift = x_shift
        self.y_shift = y_shift

    def _apply_on_image(self):
        self.image = sp.ndimage.shift(self.image, (self.x_shift, self.y_shift, 0))

    def _apply_on_annots(self):
        self.annots.shift(self.x_shift, self.y_shift)


class Scale(SITransform):
    def __init__(
            self,
            x_scale: Optional[float] = None,
            y_scale: Optional[float] = None
    ) -> None:
        super().__init__()
        self.x_scale = x_scale if x_scale is not None else 1
        self.y_scale = y_scale if y_scale is not None else 1

    def _apply_on_image(self):
        self.image = sp.ndimage.zoom(self.image, (self.x_scale, self.y_scale, 1))

    def _apply_on_annots(self):
        self.annots.scale(self.x_scale, self.y_scale)


class Rotate(SITransform):
    def __init__(
            self,
            angle: float
    ) -> None:
        super().__init__()
        self.angle = angle

    def _apply_on_image(self):
        self.image = sp.ndimage.rotate(self.image, self.angle, reshape=False)

    def _apply_on_annots(self):
        self.annots.rotate(self.angle)



# def scale():
#     pass
#
#
# def rotate():
#     pass
#
#
# def transform():
#     pass
#
#
# def mosaic():
#     pass
#
#
# def overlay():
#     pass
#
#
# def crop():
#     pass
#
#
# def resize():
#     pass
#
#
# def cutout():
#     pass
#
#
# def mixup():
#     pass
