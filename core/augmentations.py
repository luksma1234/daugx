import numpy as np
from annotations import Annotations
from transforms import (
    SingleImageTransform,
    MultiImageTransform,
    ImageOnlyTransform
)
import scipy as sp


class Shift(SingleImageTransform):

    def __init__(
            self,
            x_shift: float,
            y_shift: float,
            image: np.ndarray,
            annots: Annotations,
            reflect: bool = False,
            fill: int = 0
    ):
        super().__init__(image, annots)
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.mode = "reflect" if reflect else "constant"
        self.fill = fill

    def apply_on_image(self):
        self.image = sp.ndimage.shift(
            shift=(self.x_shift, self.y_shift),
            mode=self.mode,
            cval=self.fill
        )

    def apply_on_annots(self):
        self.annots.shift((self.x_shift, self.y_shift))


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
