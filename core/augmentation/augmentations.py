
import math
from typing import Optional, Tuple
from copy import deepcopy

import numpy as np
import cv2

from daugx.core.augmentation.transforms import (
    SITransform,
    MITransform,
    IOTransform
)
from daugx.core import constants as c
from daugx.core.augmentation.annotations import Annotations

# TODO: This needs documentation
# TODO: There need to be some global value for a background color
# TODO: Random needs to be implemented
# TODO: Remove __eq__ and replace with inspect.signature(foo) to get all input arguments

class Shift(SITransform):
    def __init__(
            self,
            x_shift: float = 0,
            y_shift: float = 0
    ) -> None:
        """

        Shifts all pixels of an image. Pixels that are shifted outside the image are removed. New pixels will be created
        using the defined background color. Shifts the right hand side of the image for positive x. Shifts upwards for
        positive y.

        Args:
            x_shift (Optional[float]): Shifts image towards the right hand side of the image.
                                       Negative numbers shift to the left hand side.
            y_shift (Optional[float]): Shifts image upwards. Negative numbers downwards.

        Returns:
            (None)

        """
        super().__init__()
        self.x_shift = x_shift
        self.y_shift = -y_shift

    def __eq__(self, other):
        if not isinstance(other, Shift):
            return False
        return other.x_shift == self.x_shift and other.y_shift == self.y_shift

    def _apply_on_image(self):
        rows, cols, _ = self.image.shape
        affine = np.float32([[1, 0, self.x_shift], [0, 1, self.y_shift]])
        self.image = cv2.warpAffine(self.image, affine, (cols, rows))

    def _apply_on_annots(self):
        self.annots.shift(self.x_shift, self.y_shift)


class Scale(SITransform):
    def __init__(
            self,
            x_scale: float = 1,
            y_scale: float = 1
    ) -> None:
        """

        Scales an image by resizing it. Uses cv2.resize for scaling of the image. Horizontal scaling is tuned by x
        scaling. Vertical scaling is tunes by the y factor. Values greater than 1 will increase the image size on the
        scaling axis by the scaling factor. Values smaller than 1 will decrease image size on the scaling axis by the
        scaling factor.

        Args:
            x_scale (float): Horizontal scaling factor
            y_scale (float): Vertical scaling factor

        Returns:
            (None)

        """
        super().__init__()
        assert x_scale > 0 and y_scale > 0
        self.x_scale = x_scale
        self.y_scale = y_scale

    def __eq__(self, other):
        if not isinstance(other, Scale):
            return False
        return other.x_scale == self.x_scale and other.y_scale == self.y_scale

    def _apply_on_image(self):
        self.image = cv2.resize(self.image,None,fx=self.y_scale, fy=self.x_scale, interpolation = cv2.INTER_LINEAR)

    def _apply_on_annots(self):
        self.annots.scale(self.x_scale, self.y_scale)


class Rotate(SITransform):
    def __init__(
            self,
            angle: float
    ) -> None:
        """

       Rotates all pixels of an image around the image center. Image rotates clockwise with a positive angle.
       Image rotates counterclockwise with a negative angles.

        Args:
            angle (float): Angle of rotation

        Returns:
            (None)

        """
        super().__init__()
        self.angle = angle

    def __eq__(self, other):
        if not isinstance(other, Rotate):
            return False
        return other.angle == self.angle

    def _apply_on_image(self):
        rows, cols, _ = self.image.shape
        affine = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), self.angle, 1)
        self.image = cv2.warpAffine(self.image, affine, (cols, rows))

    def _apply_on_annots(self):
        self.annots.rotate(self.angle)


class Resize(SITransform):
    def __init__(
            self,
            width: int,
            height: int,
            preserve_aspect_ratio=True
    ):
        """

       Resizes an image to the target width and height. Can preserve aspect ratio. Resulting additional pixels are
       filled with the background color.

        Args:
            width (int): Target width of image
            height (int): Target height of image
            preserve_aspect_ratio (bool): Weather the image aspect ratio must be preserved after resizing.

        Returns:
            (None)

        """
        super().__init__()
        self.width = width
        self.height = height
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.img_width = None
        self.img_height = None
        self.extend = 0

        assert self.width > 0 and self.height > 0
        # Tests have shown that an aspect_ratio of 6 is max
        assert (1 / 6) < (self.width / self.height) < 6

    def __eq__(self, other):
        if not isinstance(other, Resize):
            return False
        return (other.width == self.width and other.height == self.height and
                other.preserve_aspect_ratio == self.preserve_aspect_ratio)

    def _apply_on_image(self):
        self.img_width, self.img_height, _ = self.image.shape
        if not self.preserve_aspect_ratio:
            self.image = cv2.resize(self.image, None, fx=self.height / self.img_height,
                                    fy=self.width / self.img_width, interpolation=cv2.INTER_LINEAR)
        else:
            asp_ratio = self.img_width / self.img_height
            resized_asp_ratio = self.width / self.height
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
            self.image = cv2.resize(
                self.image,
                None,
                fx=self.height / np.shape(self.image)[1],
                fy=self.width / np.shape(self.image)[0],
                interpolation = cv2.INTER_LINEAR
            )

    def _apply_on_annots(self):
        width_ratio = self.width / self.img_width
        height_ratio = self.height / self.img_height
        if self.preserve_aspect_ratio:
            if self.extend > 0:
                self.annots.scale(
                    width_ratio,
                    self.height / (self.img_height + self.extend)
                )
                self.annots.shift(
                    y_shift=int(self.extend * self.height / (2 * (self.img_height + self.extend)))
                )

            elif self.extend < 0:
                self.annots.scale(
                    self.width / (self.img_width - self.extend),
                    height_ratio
                )
                self.annots.shift(
                    x_shift=int(-self.extend * self.width / (2 * (self.img_width - self.extend)))
                )
        else:
            self.annots.scale(width_ratio, height_ratio)


class Mosaic(MITransform):
    def __init__(self, mode: str = c.MOSAIC_RESIZE_MODE):
        """

       Creates a new image from four input images by placing them in a 2x2 order. Resizes or Crops the resulting image.

        Args:
            mode (str): One of 'resize' or 'crop'. Defines weather the output image is resized or cropped to fit the
                        input image size.

        Returns:
            (None)

        """
        super().__init__()
        self.mode = mode
        self.unify_width = None
        self.unify_height = None
        self.inflation = 0.25

    def __eq__(self, other):
        if not isinstance(other, Mosaic):
            return False
        return other.mode == self.mode

    def _preprocess(self):
        preprocessed_images, preprocessed_annots = [], []
        img_areas = [annots.border.area for annots in self.annots_list]
        self.unify_width, self.unify_height = self.annots_list[img_areas.index(min(img_areas))].border.corners[1]
        resizer = Resize(self.unify_width, self.unify_height)
        # TODO: Implement Random Crop
        cropper = RandomCrop()
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
        for idx, annots in enumerate(self.annots_list):
            annots.scale_border(2, 2)
            if idx == 0:
                self.annots = deepcopy(annots)
            else:
                match idx:
                    case 1:
                        annots.shift(x_shift=self.unify_width)
                    case 2:
                        annots.shift(x_shift=self.unify_width, y_shift=self.unify_height)
                    case 3:
                        annots.shift(y_shift=self.unify_height)
                for annot in annots:
                    self.annots.add(annot.boundary.points, annot.label.id, annot.label.name)


class Crop(SITransform):
    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float) -> None:
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
        assert 0 < self.x_min < self.x_max <= 1 and 0 < self.y_min < self.y_max <= 1

        # absolute x and y values
        self.x_min_abs = None
        self.y_min_abs = None
        self.x_max_abs = None
        self.y_max_abs = None

    def __eq__(self, other):
        if not isinstance(other, Crop):
            return False
        return (other.x_min == self.x_min and other.y_min == self.y_min and other.x_max == self.x_max
                and other.y_max == self.y_max)

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


class RandomCrop(SITransform):
    def __init__(
            self,
            min_width: float = 0.2,
            max_width: float = 1,
            min_height: float = 0.2,
            max_height: float = 1,
            preserve_aspect_ratio: bool = True
    ):
        """
        Crops an image by randomly selecting a crop area limited by width and height percentages.
        Args:
            min_width (float): Minimal width percentage of original width
            max_width (float): Maximal width percentage of original width
            min_height (float): Minimal height percentage of original width
            max_height (float): Maximal height percentage of original width
            preserve_aspect_ratio (bool): Weather the aspect ratio of the crop box matches the images aspect ratio
        """
        super().__init__()
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        self.preserve_aspect_ratio = preserve_aspect_ratio

        # validate crop area
        assert 0 < self.min_width < self.max_width <= 1 and 0 < self.min_height < self.max_height <= 1

    def __eq__(self, other):
        if not isinstance(other, RandomCrop):
            return False
        return (other.min_width == self.min_width and other.max_width == self.max_width and
                other.min_height == self.min_height and other.max_height == self.max_height)

    def apply(
            self,
            image: np.ndarray,
            annots: Optional[Annotations] = None,
            rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, Annotations]:
        assert rng is not None
        img_width, img_height, _ = image.shape
        asp_ratio = img_width / img_height
        if self.preserve_aspect_ratio:
            if (self.max_width - self.min_width) / asp_ratio > self.max_height:
                self.max_height = (self.max_width - self.min_width) / asp_ratio
        crop_box_width = rng.random() * (self.max_width - self.min_width) * img_width
        if self.preserve_aspect_ratio:
            crop_box_height = crop_box_width / asp_ratio
            assert crop_box_height <= self.max_height * img_height
        else:
            crop_box_height = rng.random() * (self.max_height - self.min_height) * img_height
        x_min = rng.random() * (img_width - crop_box_width)
        y_min = rng.random() * (img_height - crop_box_height)
        x_max = x_min + crop_box_width
        y_max = y_min + crop_box_height
        cropper = Crop(x_min, y_min, x_max, y_max)
        return cropper.apply(image, annots)

    def _apply_on_image(self):
        pass

    def _apply_on_annots(self):
        pass


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
        self.inflation = 0.5

        assert 0.4 <= self.lam <= 0.6, f"Lambda parameter for MixUp must be in range 0.4 - 0.6. Found {self.lam}."

    def __eq__(self, other):
        if not isinstance(other, MixUp):
            return False
        return other.lam == self.lam

    def _preprocess(self) -> None:
        assert len(self.image_list) == 2, (f"MixUp Augmentation needs exactly 2 images for blending. "
                                           f"Found {len(self.image_list)}")

    def _apply_on_images(self) -> None:
        self.image = np.array(
            self.image_list[0] * self.lam + self.image_list[1] * (1 - self.lam),
            dtype=np.uint8
        )

    def _apply_on_annots(self) -> None:
        for idx, annots in enumerate(self.annots_list):
            if idx == 0:
                self.annots = deepcopy(annots)
            else:
                for annot in annots:
                    self.annots.add(annot.boundary.points, annot.label.id, annot.label.name)
