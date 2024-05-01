import numpy as np
from .annotations import Annotations
from .transforms import (
    SingleImageTransform,
    MultiImageTransform,
    ImageOnlyTransform
)
import scipy as sp
from ..utils.mat_utils import get_2d_transf_mat, get_3d_transf_mat

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
            rot_angle: float = 0,
            x_shear: float = 0,
            y_shear: float = 0,
            fill: int = 0
    ) -> None:
        super().__init__(image, annots)
        self.fill = fill
        self.output_shape = (self.image_width, self.image_height, 3)
        self.translation_offset = [x_shift, y_shift, 0]
        self.transf_mat_2d = get_2d_transf_mat(
            scale=(x_scale, y_scale),
            distortion=(x_shear, y_shear),
            angle=rot_angle,
        )
        self.norm_transf_mat = get_2d_transf_mat(
            scale=(x_scale/self.image_width, y_scale/self.image_height),
            translation=(x_shift/self.image_width, y_shift/self.image_height),
            distortion=(x_shear/self.image_width, y_shear/self.image_height),
            angle=rot_angle,
        )

    def apply_on_image(self) -> np.ndarray:
        img_shape_array = np.array([self.image_width, self.image_height, 3]) * 0.5
        offset = img_shape_array - np.dot(self.transf_mat_2d, img_shape_array)
        offset -= self.translation_offset
        print(f"Offset: {offset}")
        return sp.ndimage.affine_transform(
            input=self.image,
            matrix=self.transf_mat_2d,
            mode="constant",
            cval=self.fill,
            offset=offset,
        )

    def apply_on_annots(self) -> Annotations:
        return self.annots.transf(self.norm_transf_mat)


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
