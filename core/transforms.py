from typing import List, Tuple, Optional

import numpy as np

from .annotations import Annotations


class SITransform:
    """
    Single Image Transform
    """
    def __init__(self, image: Optional[np.ndarray] = None, annots: Optional[Annotations] = None):
        self.image = image
        self.annots = annots

    def apply(
            self,
            image: Optional[np.ndarray] = None,
            annots: Optional[Annotations] = None
    ) -> Tuple[np.ndarray, Annotations]:
        """
        Applies the transformation to the image and its annotations.
        Args:
            image (Optional[np.ndarray]): Any image as numpy array. Only necessary as input if transform was not
                                          initialized with image.
            annots (Optional[Annotations]): Annotations of image. Only necessary as input if transform was not
                                            initialized with annotations.
        Returns:
            (Tuple[np.ndarray, Annotations]): Tuple of transformed image and transformed annotations
        """
        if image is not None:
            self.image = image
        if annots is not None:
            self.annots = annots
        if self.image is None:
            raise ValueError("Unable to perform transformation. Image was not provided.")
        self._apply_on_image()
        self._apply_on_annots()
        return self.image, self.annots

    def _apply_on_image(self):
        """
        -- This method must be overwritten in a subclass --

        Applied the transformation to the image.
        """
        pass

    def _apply_on_annots(self):
        """
        -- This method must be overwritten in a subclass --

        Applied the transformation to the annotations.
        """
        pass


class MITransform:
    """
    Multi Image Transform
    """
    def __init__(self, image_list: List[np.ndarray], annots_list: List[Annotations]):
        self.image_list = image_list
        self.annots_list = annots_list
        self.shape_list = [np.shape(image)[:2] for image in self.image_list]
        self.image = None
        self.annots = None

    def combine_images(self):
        pass

    def apply_on_image(self) -> None:
        pass

    def apply_on_annots(self) -> None:
        pass


class IOTransform:
    """
    Image Only Transform
    """
    def __init__(self, image: np.ndarray):
        self.image = image

    def apply_on_image(self):
        pass

