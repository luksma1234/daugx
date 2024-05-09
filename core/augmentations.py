import numpy as np
from .annotations import Annotations
from .transforms import (
    SingleImageTransform,
    MultiImageTransform,
    ImageOnlyTransform
)
import scipy as sp
from ..utils.mat_utils import get_2d_transf_mat

# TODO: Debug affine transformation of image


class Affine(SingleImageTransform):
    def __init__(
            self,
            image: np.ndarray,
            annots: Annotations,
            x_shift: float = 0,
            y_shift: float = 0,
            x_scale: float = 1,
            y_scale: float = 1,
            x_shear: float = 0,
            y_shear: float = 0,
            rot_angle: float = 0,
            fill: int = 0
    ) -> None:
        super().__init__(image, annots)
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.x_shear = x_shear
        self.y_shear = y_shear
        self.rot_angle = rot_angle

        self.fill = fill
        self.output_shape = (self.image_width, self.image_height, 3)
        self.translation_offset = [x_shift, y_shift, 0]

        self.img_center = np.array([self.image_width, self.image_height, 3]) * 0.5
        self.transf_mat = get_2d_transf_mat(
            scale=(self.x_scale, self.y_scale),
            shear=(self.x_shear, self.y_shear),
            angle=self.rot_angle,
        )
        self.offset = self.img_center - np.dot(self.transf_mat, self.img_center)
        self.offset -= self.translation_offset

    def apply_on_image(self) -> np.ndarray:
        print(f"Offset: {self.offset}")
        return sp.ndimage.affine_transform(
            input=self.image,
            matrix=self.transf_mat,
            mode="constant",
            cval=self.fill,
            offset=self.offset,
        )
    # TODO: Something is wrong with the shift or with the distortion
    # maybe its just that the img was permuted
    def apply_on_annots(self) -> Annotations:
        x_shift, y_shift, _ = self.offset
        self.annots.transf(self.transf_mat, (-x_shift, -y_shift))
        return self.annots


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
