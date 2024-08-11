from copy import deepcopy
from typing import Tuple, List, Union, Optional, Dict

from daugx.core.data.meta_inf import MetaInf
from daugx.core.data.filter import FilterSequence, Filter
from daugx.core.augmentation.annotations import Annotations
from daugx.utils.misc import read_img, get_random, is_in_dict, fetch_by_prob
import daugx.core.constants as c

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
    def __init__(self, id_, data_packages: List[DataPackage], filters: List[FilterSequence]):
        """
        Filters are applied in the initialization.
        What happens if there are no indexes for a filter?
        -> This means that a filtering has no results. Therefore, there should be no path which uses this filter.
        Args:
            data_packages (List[DataPackage]): All data packages for this dataset
            filters (List[dict]): Filters applied for this dataset
        """
        self.__id = id_
        self.data_packages = data_packages
        self.used = []
        self.__filter_indexes: Dict[str, list] = {}
        self._init_filters(filters)

    @property
    def id(self):
        return self.__id

    def fetch(self, filter_: Optional[Union[str, list]]):
        rand = get_random()
        if filter_ is None:
            return fetch_by_prob(self.data_packages, rand)
        elif isinstance(filter_, str):
            return self.data_packages[fetch_by_prob(self.__filter_indexes[filter_], rand)]
        elif isinstance(filter_, list):
            if not is_in_dict(str(filter_), self.__filter_indexes):
                self._combine_filters(filter_)
            return self.data_packages[fetch_by_prob(self.__filter_indexes[str(filter_)], rand)]

    def _init_filters(self, filters: List[FilterSequence]):
        for sequence in filters:
            self.__filter_indexes[sequence.id] = sequence.filter(self.data_packages)

    def _combine_filters(self, filters: List[str]):
        filter_set = set(filters[0])
        for s in filters[1:]:
            filter_set.intersection_update(s)
        filter_list = list(filter_set)
        self.__filter_indexes[str(filters)] = filter_list
