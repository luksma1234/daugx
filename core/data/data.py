from copy import deepcopy
from typing import Tuple, List, Union, Optional

from daugx.core.data.meta_inf import MetaInf
from daugx.core.augmentation.annotations import Annotations
from daugx.utils.misc import read_img, get_random


import numpy as np


class DataPackage:

    def __init__(self, image_path: str, annotations: Annotations):
        """
        A high level wrapper to wrap annotations and image data. Remains read-only after initialization. Saves image
        and annotation meta information in initialization.
        Args:
            image_path (str): Absolute path to image
            annotations (Annotations): Annotations of one image reference
        """
        self.__image_path = image_path
        self.__annotations = annotations
        self.__meta_inf: Union[MetaInf, None] = None

    @property
    def meta_inf(self) -> MetaInf:
        """
        Meta Information of image and its annotations. Meta information is used to filter Data Packages.
        Returns a dict of Meta Information.
        """
        if self.__meta_inf is None:
            self._retrieve_meta_inf()
        return self.__meta_inf

    @property
    def data(self) -> Tuple[np.ndarray, Annotations]:
        """
        Data Property. Returns a Tuple of the loaded image and its annotations.
        """
        return self._load_image(), deepcopy(self.__annotations)

    def _load_image(self):
        return read_img(self.__image_path)

    def _retrieve_meta_inf(self) -> None:
        self.__meta_inf = MetaInf(self.__annotations)


class Dataset:
    def __init__(self, data_packages: List[DataPackage], filters: List[str]):
        """
        Filters are applied in the initialization.
        """
        self.data_packages = data_packages
        self.used = []

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.data_packages[index]
        else:
            raise ValueError(f"Unable to parse index with type {type(index)}.")

    def fetch(self, filter_: Optional[str]):
        index = int(get_random() * len(self.data_packages))

    def _filter(self):
        pass

    def reset(self):
        pass

    def _update_used(self, index: int):
        pass
