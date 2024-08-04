from copy import deepcopy
from operator import itemgetter
from typing import Tuple, List, Dict, Union, Optional

from daugx.core.augmentation.annotations import Annotations, Label
from daugx.utils.misc import read_img, get_random

import numpy as np


class MetaInf:
    def __init__(self, annotations: Annotations):
        self.annotations = annotations
        self.img_width = annotations.width
        self.img_height = annotations.height
        self.n_annotations = len(annotations.annots)
        self.annotation_label_ids = [annotation.label.id for annotation in annotations.annots]
        self.annotation_label_names = [annotation.label.name for annotation in annotations.annots]
        self.annotation_max_area = max(
            [annotation.area for annotation in annotations.annots]
        )
        self.annotation_min_area = min(
            [annotation.area for annotation in annotations.annots]
        )

    def min_area_by_label_name(self, label_name):
        annotation_areas = [
            annotation.area for annotation in self.annotations.annots if annotation.label.name == label_name
        ]
        if annotation_areas:
            return min(annotation_areas)
        return None

    def min_area_by_label_id(self, label_id):
        annotation_areas = [
            annotation.area for annotation in self.annotations.annots if annotation.label.id == label_id
        ]
        if annotation_areas:
            return min(annotation_areas)
        return None

    def max_area_by_label_name(self, label_name):
        annotation_areas = [
            annotation.area for annotation in self.annotations.annots if annotation.label.name == label_name
        ]
        if annotation_areas:
            return max(annotation_areas)
        return None

    def max_area_by_label_id(self, label_id):
        annotation_areas = [
            annotation.area for annotation in self.annotations.annots if annotation.label.id == label_id
        ]
        if annotation_areas:
            return max(annotation_areas)
        return None

    def n_annotations_by_label_name(self, label_name):
        annotations = [annotation for annotation in self.annotations if annotation.label.name == label_name]
        return len(annotations)

    def n_annotations_by_label_id(self, label_id):
        annotations = [annotation for annotation in self.annotations if annotation.label.id == label_id]
        return len(annotations)


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

    def fetch(self, filter_: Optional[str]):
        index = int(get_random() * len(self.data_packages))

    def _filter(self):
        pass

    def reset(self):
        pass

    def _update_used(self, index: int):
        pass

class Filter:
    def __init__(self, query: str, name: str):
        self.query = query
        self.name = name

    def filter(self, dataset: Dataset):
        


























