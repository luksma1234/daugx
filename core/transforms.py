import numpy as np
from annotations import Annotations
from typing import List


class SingleImageTransform:
    def __init__(self, image: np.ndarray, annots: Annotations):
        self.image = image
        self.annots = annots

    def apply_on_image(self):
        pass

    def apply_on_annots(self):
        pass


class MultiImageTransform:
    def __init__(self, image_list: List[np.ndarray], annots_list: List[Annotations]):
        self.image_list = image_list
        self.annots_list = annots_list
        self.image = None
        self.annots = None

    def combine_images(self):
        pass

    def apply_on_image(self) -> None:
        pass

    def apply_on_annots(self) -> None:
        pass


class ImageOnlyTransform:
    def __init__(self, image: np.ndarray):
        self.image = image

    def apply_on_image(self):
        pass

