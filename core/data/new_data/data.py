from typing import List, Dict, Union, Optional
import numpy as np
from abc import ABC
from dataclasses import dataclass

from daugx.utils.misc import fetch_by_prob, new_id, fetch_by_prob_list
import daugx.core.constants as c


@dataclass(slots=True)
class DataItem(ABC):
    """
    DataItem Base Class. Data Items are single points of data that act independently.
    """
    id: str
    depends_on: Optional[str]
    is_fully_loaded: bool


@dataclass(slots=True)
class ImageItem(DataItem):
    width: int
    height: int
    path: str
    image: Optional[np.ndarray] = None


class AnnotationItem(DataItem):
    raise NotImplementedError


class LabelItem(DataItem):
    raise NotImplementedError


class AudioItem(DataItem):
    raise NotImplementedError


class DataPackage:
    """
    One Data package loaded from DataSet. Consists of at least one prime modality and n additional modalities.
    """
    pass

from daugx.core.data.new_data.data_loader import (
    DataLoader,
    ImageLoader,
    AnnotationLoader,
    AudioLoader,
    LabelLoader
)


class Modality:
    """
    High level representation of any type of data. Loads data.
    """
    def __init__(
            self,
            rng: np.random.Generator,
            modality_type: Optional[str],
            is_prime: bool = False,
            id_: Optional[str] = None,
            parent: Optional[str] = None,
            loader: Optional[DataLoader] = None,
            loader_args: Optional[tuple] = None
    ):
        # make sure one of loader or loader args is provided
        assert (loader is not None or loader_args is not None)
        self.__rng: np.random.Generator = rng
        self.__id: str = new_id(self.__rng) if id_ is None else id_
        self.__data = {}
        self.__modality_type = modality_type
        assert self.modality_type in c.MODALITY_TYPES
        # prime modalities are modalities without dependencies
        # e.g. can be loaded by themselves (with all their children)
        self.__is_prime = is_prime
        # If parent is set, this modality is a child modality.
        # This Modality is then fully dependent on its parent.
        # e.g. A label of an image annotation is fully dependent on the annotation itself.
        self.__parent = parent
        # TODO: Does this assertion make sense?
        assert (self.__parent is None and self.__is_prime) or (self.__parent is not None and not self.__is_prime)
        self.is_shallow_loaded: bool = False
        if loader is not None:
            self.__loader = loader
        else:
            self.__loader_args = loader_args
            self.__loader: Optional[DataLoader] = None
            self._set_loader()
        assert self.__loader.loader_type == self.modality_type
        # Preload modality data
        self._preload()

    def __len__(self):
        return self.size

    @property
    def id(self):
        return self.__id

    @property
    def size(self):
        return self.__size

    @property
    def is_prime(self):
        return self.__is_prime

    @property
    def parent(self):
        return self.__parent

    @property
    def data(self):
        return self.__data

    @property
    def modality_type(self):
        return self.__modality_type

    def _set_loader(self):
        """
        Initializes the data loader depending on the modality type.
        """
        match self.__modality_type:
            case c.MODALITY_TYPE_IMAGE:
                self.__loader = ImageLoader(*self.__loader_args)
            case c.MODALITY_TYPE_ANNOTATION:
                self.__loader = AnnotationLoader(*self.__loader_args)
            case c.MODALITY_TYPE_LABEL:
                self.__loader = LabelLoader(*self.__loader_args)
            case c.MODALITY_TYPE_AUDIO:
                self.__loader = AudioLoader(*self.__loader_args)

    def _add(self, data_item: DataItem):
        if data_item.id in self.__data:
            self.__data[data_item.id].append(data_item)
        else:
            self.__data[data_item.id] = [data_item]

    def _preload(self):
        self._reset()
        data_list = self.__loader.preload()
        for data_item in data_list:
            self._add(data_item)
        self.__size = len(self.__data)

    def fetch(self, id_: Optional[str] = None) -> Optional[DataItem]:
        """
        Fetches one data item.
        Note: __next__ does not make sense here, because we must have the option to pass an argument.

        Args:
            id_ (Optional[str]): ID of item to fetch. Fetches random item if no id is given.
        """
        if id_ is not None:
            return self.__data.get(id_)
        return fetch_by_prob(list(self.__data.values()), self.__rng.random())

    def _reset(self):
        self.__data = {}
        self.__size = 0


class DataSet:
    """
    High Level class for any data management. Loads data, manages modalities and stores meta information.
    """

    def __init__(self, rng: np.random.Generator, modalities: Optional[List[Modality]] = None):
        self.__rng = rng
        self.__modalities = modalities
        self.__primes: Optional[List[Modality]] = None
        self.__additionals: Optional[List[Modality]] = None
        self.__prime_probs: Optional[List[float]] = None

    @property
    def additionals(self):
        if self.__additionals is None:
            self.__additionals = [modality for modality in self.__modalities if not modality.is_prime]
        return self.__additionals

    @property
    def primes(self) -> list:
        if self.__primes is None:
            self.__primes = [modality for modality in self.__modalities if modality.is_prime]
        return self.__primes

    def fetch(self):
        pass
    # TODO: Yes it makes sense that a modality has grandchildren. Image -> Annotation -> Label
    # TODO: How to handle this? grandchild has to be loaded with id of child

    def _get_fetch_id(self):
        """
        Select one prime id to fetch.
        """
        return fetch_by_prob_list(self.primes, self.__prime_probs, self.__rng).id

    def _load_children(self, parent_id: str) -> List[Modality]:
        children = []
        for child in self._get_children(parent_id):
            children.append(child)
            children.extend(self._load_children(child.id))
        return children

    def _get_children(self, parent_id: str):
        return [modality for modality in self.__modalities if modality.parent == parent_id]


    def _load_additional(self):
        pass

    def add_modality(self, modality):
        self.__modalities.append(modality)
        if modality.is_prime:
            self._calc_prime_probs()

    def _calc_prime_probs(self):
        prime_data_sum = sum([modality.size for modality in self.primes])
        self.__prime_probs = [prime_data_sum / modality.size for modality in self.primes]