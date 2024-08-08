import math
from typing import Optional
from copy import deepcopy

import numpy as np
import scipy as sp

from .transforms import (
    SITransform,
    MITransform,
    IOTransform
)


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
        self.annots.scale_border(self.x_scale, self.y_scale)
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


class Resize(SITransform):
    def __init__(self, target_width: int, target_height: int, preserve_aspect_ratio=True):
        super().__init__()
        self.target_width = target_width
        self.target_height = target_height
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.img_width = None
        self.img_height = None
        self.extend = 0

    def _apply_on_image(self):
        """
        Here may be room for improvements... Its working - but maybe not as efficient as possible
        """
        self.img_width, self.img_height, _ = np.shape(self.image)
        if not self.preserve_aspect_ratio:
            self.image = sp.ndimage.zoom(
                self.image,
                (self.target_width / self.img_width, self.target_height / self.img_height, 1)
            )
        else:
            asp_ratio = self.img_width / self.img_height
            resized_asp_ratio = self.target_width / self.target_height
            self.extend = math.ceil((asp_ratio - resized_asp_ratio) * self.img_width)
            if self.extend > 0:
                stack_a = np.zeros(
                    (self.img_width, int(self.extend / 2), 3),
                    dtype=np.uint8
                )
                stack_b = np.zeros(
                    (self.img_width, self.extend - int(self.extend / 2), 3),
                    dtype=np.uint8
                )
                self.image = np.hstack((stack_a, self.image, stack_b))
            elif self.extend < 0:
                stack_a = np.zeros(
                    (int(-self.extend / 2), self.img_height, 3),
                    dtype=np.uint8
                )
                stack_b = np.zeros(
                    (-self.extend - int(-self.extend / 2), self.img_height, 3),
                    dtype=np.uint8
                )
                self.image = np.vstack((stack_a, self.image, stack_b))
            self.annots.set_border(x_max=np.shape(self.image)[0], y_max=np.shape(self.image)[1])
            self.annots.rebase_border()
            self.image = sp.ndimage.zoom(
                self.image,
                (self.target_width / np.shape(self.image)[0], self.target_height / np.shape(self.image)[1], 1)
            )

    def _apply_on_annots(self):
        if self.preserve_aspect_ratio:
            if self.extend > 0:
                self.annots.scale_border(
                    self.target_width / self.img_width,
                    self.target_height / (self.img_height + self.extend)
                )
                self.annots.scale(
                    self.target_width / self.img_width,
                    self.target_height / (self.img_height + self.extend)
                )
                self.annots.shift(
                    y_shift=int(self.extend * self.target_height / (2 * (self.img_height + self.extend)))
                )

            elif self.extend < 0:
                self.annots.scale(
                    self.target_width / (self.img_width - self.extend),
                    self.target_height / self.img_height
                )
                self.annots.shift(
                    x_shift=int(-self.extend * self.target_width / (2 * (self.img_width - self.extend)))
                )
        else:
            self.annots.scale(self.target_width / self.img_width, self.target_height / self.img_height)


class Mosaic(MITransform):
    def __init__(self, mode: str = "resize"):
        """
        Stitches 4 images together to get one single image.
        Args:
            mode (str): 'resize' or 'crop' - Defines how the images are preprocessed. Images can either be resized or
                        cropped to fit a unify size. The unify size is determined by the dimensions of the smallest
                        image.
        """
        super().__init__()
        self.mode = mode
        self.unify_width = None
        self.unify_height = None

    def _preprocess(self):
        assert len(self.image_list) == 4, (f"Mosaic Augmentation needs exactly 4 images to stitch together. "
                                           f"Found {len(self.image_list)}")
        preprocessed_images, preprocessed_annots = [], []
        img_areas = [annots.border.area for annots in self.annots_list]
        self.unify_width, self.unify_height = self.annots_list[img_areas.index(min(img_areas))].border.corners[1]
        resizer = Resize(self.unify_width, self.unify_height)
        # cropper = Crop()
        for image, annots in zip(self.image_list, self.annots_list):
            if annots.width == self.unify_width and annots.height == self.unify_height:
                prep_img, prep_annots = image, annots
            else:
                if self.mode == "resize":
                    prep_img, prep_annots = resizer.apply(image, annots)
                elif self.mode == "crop":
                    raise "NOT IMPLEMENTED YET"
                else:
                    raise f"Unknown mode for Mosaic '{self.mode}'."
            preprocessed_images.append(prep_img)
            preprocessed_annots.append(prep_annots)
        self.image_list = preprocessed_images
        self.annots_list = preprocessed_annots

    def _apply_on_images(self) -> None:
        self.image = np.vstack((
            np.hstack((self.image_list[0], self.image_list[3])),
            np.hstack((self.image_list[1], self.image_list[2])),
        ))

    def _apply_on_annots(self) -> None:
        for annots in self.annots_list:
            annots.scale_border(2, 2)
            annots.rebase_border()
        self.annots_list[1].shift(y_shift=self.unify_height)
        self.annots_list[2].shift(x_shift=self.unify_width, y_shift=self.unify_height)
        self.annots_list[3].shift(x_shift=self.unify_width)
        for idx, annots in enumerate(self.annots_list):
            if idx == 0:
                self.annots = deepcopy(annots)
            else:
                for annot in annots:
                    self.annots.add(annot.label.id, annot.boundary.points, annot.label.name)


class Crop(SITransform):
    def __init__(self, x_min, y_min, x_max, y_max) -> None:
        """
        Crops an image into the specified boundary.
        Args:
            x_min (float): min value for x in percentage
            y_min (float): min value for y in percentage
            x_max (float): max value for x in percentage
            y_max (float): max value for y in percentage
        """
        super().__init__()
        # percentage x and y values
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

        # validate crop box
        assert self.x_min < self.x_max <= 1 and self.y_min < self.y_max <= 1

        # absolute x and y values
        self.x_min_abs = None
        self.y_min_abs = None
        self.x_max_abs = None
        self.y_max_abs = None

    def _apply_on_image(self):
        image_width = self.annots.width
        image_height = self.annots.height
        self.x_min_abs = int(image_width * self.x_min)
        self.y_min_abs = int(image_height * self.y_min)
        self.x_max_abs = int(image_width * self.x_max)
        self.y_max_abs = int(image_height * self.y_max)
        self.image = self.image[self.x_min_abs:self.x_max_abs, self.y_min_abs:self.y_max_abs, :]

    def _apply_on_annots(self):
        self.annots.crop(self.x_min_abs, self.y_min_abs, self.x_max_abs, self.y_max_abs)


class MixUp(MITransform):

    """
    Implemented from https://arxiv.org/abs/1710.09412v2
    MixUp trains a neural network on convex combinations of pairs of examples and their labels.
    """

    def __init__(self, lam: float):
        """
        Args:
            lam (float): lambda parameter or weight parameter, which sets image blending strength. Higher value leads to
                         more appearance of first image in result. Must be in range 0.4 - 0.6.
        """
        super().__init__()
        # lambda parameter
        self.lam = lam

        assert 0.4 <= self.lam <= 0.6, f"Lambda parameter for MixUp must be in range 0.4 - 0.6. Found {self.lam}."

    def _preprocess(self) -> None:
        assert len(self.image_list) == 2, (f"MixUp Augmentation needs exactly 2 images for blending. "
                                           f"Found {len(self.image_list)}")

    def _apply_on_images(self) -> None:
        self.image = np.array(
            self.image_list[0] * self.lam + self.image_list[1] * (1 - self.lam),
            dtype=np.uint8
        )

    def _apply_on_annots(self) -> None:
        self.annots = self.annots_list[0]
        for annots in self.annots_list[1]:
            self.annots.annots.append(annots)


#
#
# def transform():
#     perspective transform
#     pass
#
#
# def cutout():
#     cutout is essentialy the same as a dropout - therefore maybe not necessary
#     pass
#