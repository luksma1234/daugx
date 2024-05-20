from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

import numpy as np

from .annotations import Annotations


class SITransform(ABC):
    """
    Single Image Transform
    """
    def __init__(self):
        self.image = None
        self.annots = None

    def apply(
            self,
            image: np.ndarray,
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
        self.image = image
        self.annots = annots
        if self.image is None:
            raise ValueError("Unable to perform transformation. Image was not provided.")
        self._apply_on_image()
        if self.annots is not None:
            self._apply_on_annots()
        return self.image, self.annots

    @abstractmethod
    def _apply_on_image(self):
        """
        -- This method must be overwritten in a subclass --

        Applied the transformation to the image.
        """
        pass

    @abstractmethod
    def _apply_on_annots(self):
        """
        -- This method must be overwritten in a subclass --

        Applied the transformation to the annotations.
        """
        pass


class MITransform(ABC):
    """
    Multi Image Transform
    """
    def __init__(self):
        self.image_list = []
        self.annots_list = []
        self.image = None
        self.annots = None

    def apply(
            self,
            image_list: List[np.ndarray],
            annots_list: Optional[List[Annotations]] = None
    ) -> Tuple[np.ndarray, Annotations]:
        """
        Applies the transformation to the image and its annotations.
        Args:
            image_list (List[np.ndarray]): Any slist of images as numpy array.
            annots_list (Optional[List[Annotations]]): Annotations of images.
        Returns:
            (Tuple[np.ndarray, Annotations]): Tuple of transformed image and transformed annotations
        """
        self.image_list = image_list
        self.annots_list = annots_list
        self._apply_on_images()
        if self.annots_list is not None:
            self._apply_on_annots()
        return self.image, self.annots

    @abstractmethod
    def _apply_on_images(self) -> None:
        pass

    @abstractmethod
    def _apply_on_annots(self) -> None:
        pass


class IOTransform:
    """
    Image Only Transform
    """
    def __init__(self):
        pass

    def apply_on_image(self):
        pass

