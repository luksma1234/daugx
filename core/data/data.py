import warnings
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
    def __init__(
            self,
            id_: str,
            data_packages: List[DataPackage],
            filters: Optional[List[FilterSequence]],
            background_percentage: Optional[float],
            gen: np.random.Generator
    ):
        """
        Filters are applied in the initialization.
        What happens if there are no indexes for a filter?
        -> This means that a filtering has no results. Therefore, there should be no path which uses this filter.
        Args:
            id_ (str): ID of Dataset
            data_packages (List[DataPackage]): All data packages for this dataset
            filters (List[dict]): Filters applied for this dataset
            background_percentage (Optional[float]): Percentage at what background images are returned from fetch.
                                                     Background images skip filtering.
            gen (np.random.Generator): RNG generator
        """
        self.__id = id_
        self.data_packages = data_packages
        self.__background_percentage = background_percentage
        self.__filter_indexes: Dict[str, list] = {}
        self.__background_filter = FilterSequence(c.FILTER_BACKGROUND_ID)
        self.__background_indexes = []
        self.__gen = gen
        if self.data_packages:
            self._init_filters(filters)

    @property
    def id(self):
        return self.__id

    def fetch(self, filter_: Optional[Union[str, list]] = None) -> Tuple[np.ndarray, Annotations]:
        rand = get_random(self.__gen)
        if self.__background_percentage is not None and get_random(self.__gen) < self.__background_percentage:
            return self.data_packages[fetch_by_prob(self.__background_indexes, rand)].data
        elif filter_ is None:
            return fetch_by_prob(self.data_packages, rand).data
        elif isinstance(filter_, str):
            return self.data_packages[fetch_by_prob(self.__filter_indexes[filter_], rand)].data
        elif isinstance(filter_, list):
            if not is_in_dict(str(sorted(filter_)), self.__filter_indexes):
                self._combine_filters(filter_)
            return self.data_packages[fetch_by_prob(self.__filter_indexes[str(filter_)], rand)].data

    def _init_filters(self, filters: Optional[List[FilterSequence]]):
        self._init_background_filter()
        if filters is not None:
            for sequence in filters:
                self.__filter_indexes[sequence.id] = sequence.filter([data_package.meta_inf for data_package in self.data_packages])

    def _combine_filters(self, filters: List[str]):
        filter_set = set(filters[0])
        for s in filters[1:]:
            filter_set.intersection_update(s)
        filter_list = list(filter_set)
        self.__filter_indexes[str(sorted(filters))] = filter_list

    def _init_background_filter(self):
        """
        Filters all items, where no annotation exists.
        """
        self.__background_filter.add(Filter(
           c.FILTER_TYPE_LABEL,
            # The specifier value is None - because the specifier category is "any"
    {
                c.FILTER_SPECIFIER_CATEGORY: c.FILTER_SPECIFIER_CATEGORY_ANY,
                c.FILTER_SPECIFIER_VALUE: None
             },
           c.FILTER_OPERATOR_NOT_EXISTS,
           None
        ), c.FILTER_SEQUENCE_OPERATOR_NONE)
        self.__background_indexes = self.__background_filter.filter([data_package.meta_inf for data_package in self.data_packages])
