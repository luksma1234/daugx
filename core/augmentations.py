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
            image: Optional[np.ndarray] = None,
            annotations: Optional[Annotations] = None,
            x_shift: Optional[float] = None,
            y_shift: Optional[float] = None
    ) -> None:
        super().__init__(image, annotations)
        self.x_shift = x_shift
        self.y_shift = y_shift

    def _apply_on_image(self):
        self.image = sp.ndimage.shift(self.image, (self.x_shift, self.y_shift, 0))

    def _apply_on_annots(self):
        self.annots.shift(self.x_shift, self.y_shift)




def scale():
    pass


def rotate():
    pass


def transform():
    pass


def mosaic():
    pass


def overlay():
    pass


def crop():
    pass


def resize():
    pass


def cutout():
    pass


def mixup():
    pass
