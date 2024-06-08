from copy import deepcopy
from typing import Tuple, List, Dict

from daugx.core.augmentation.annotations import Annotations, Label
from daugx.utils.misc import read_img

import numpy as np


class DataPackage:
    def __init__(self, image_path: str, image_dims: Tuple[int, int], annotations: dict):
        """
        A high level wrapper to wrap annotations and image data. Remains read-only after initialization. Saves image
        and annotation meta information in initialization.
        Args:
            image_path (str): Absolute path to image
            image_dims (Tuple[int, int]): Image dimensions, ignoring channels
            annotations (List[dict]): Raw annotation data from Loader. Data format:
                                      {
                                          type: (str)
                                          annotations:
                                              [
                                                {
                                                    label_name: (str)
                                                    label_id: (int)
                                                    boundary: (np.ndarray)
                                                },
                                                ...
                                              ]
                                      }
        """
        self.__image_path = image_path
        self.__image_dims = image_dims
        self.__annotations_dict = annotations
        self.__annotations = None

        self._annotations_from_dict()

    @property
    def meta_inf(self) -> dict:
        """
        Meta Information of image and its annotations. Meta information is used to filter Data Packages.
        Returns a dict of Meta Information.
        """
        return self._retrieve_meta_inf()

    @property
    def data(self) -> Tuple[np.ndarray, Annotations]:
        """
        Data Property. Returns a Tuple of the loaded image and its annotations.
        """
        return self._load_image(), deepcopy(self.__annotations)

    def _load_image(self):
        return read_img(self.__image_path)

    def _annotations_from_dict(self) -> None:
        """
        Initializes all annotations from dictionary.
        """
        self.__annotations = Annotations(*self.__image_dims, self.__annotations_dict["type"])
        for annotation_dict in self.__annotations_dict["annotations"]:
            self.__annotations.add(**annotation_dict)

    def _retrieve_meta_inf(self) -> dict:
        pass
