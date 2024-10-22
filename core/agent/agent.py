from typing import Tuple, List
import warnings

from daugx.core.augmentation.annotations import Annotations
from daugx.utils.misc import load_json, get_seed, is_api_key, get_config_from_api, is_in_dict
from daugx.utils.visualizer import Visualizer, Colors
from daugx.core.agent.executor import Executor
from daugx.core.data.data import Dataset, DataPackage
from daugx.core.data.loader import InitialLoader
from daugx.core.data.filter import FilterSequence, Filter
from daugx.core import constants as c

import numpy as np


class Agent:
    """
    An Agent which takes an augmentation workflow as input. The augmentation workflow is initially split into all
    possible paths for execution. Each path is assigned an execution probability. Whenever an augmentation is requested
    from the Agent, a path is randomly picked and executed. Each path execution returns a tuple of the resulting image
    and its annotations, both as numpy arrays.

    The Agent initializes either by an API-Key or by a path to a workflow file as .json format.
    When initialized via API-Key, make sure you have connection to the internet. When using offline, please take
    advantage of the option to directly load a .json file.

    You can fetch one image from the Agent by using the 'fetch' method.

        # import daugx
        #
        # agent = daugx.Agent("my_api_key")
        # one_image = agent.fetch()

    You can use this method inside a dataloader e.g. the Tensorflow Dataloader or the PyTorch Dataloader. This ensures
    seamless integration into your training process. Make sure the initialization of the Agent takes place outside the
    loop or the dataloader, to prevent an initialization each time data is fetched.
    """
    def __init__(self, key_or_path: str, seed: int = None):
        """
        Args:
            key_or_path (str): API Key or path to augmentation workflow file
            seed (int): seed for all Agent tasks - this seed will be used throughout all augmentations
        """
        self.input = key_or_path
        self.config = self._get_config()
        self.block_config = self.config[c.CONFIG_KEY_BLOCKS]
        self.datasets_config = self.config[c.CONFIG_KEY_DATASETS]
        self.datasets = []
        self.seed = seed
        if self.seed is None:
            self.seed = get_seed()
        self.__gen = np.random.default_rng(self.seed)
        warnings.warn(f"daugx - Seed for execution: {self.seed}")
        self._init_datasets()
        self.executor = Executor(self.config[c.CONFIG_KEY_BLOCKS], self.datasets, self.__gen)

    def fetch(self, debug=False, wait_key: int = 0) -> Tuple[np.ndarray, Annotations]:
        """
        Gets a random path and executes it. All augmentations are applied in this step.

        Returns:
            (Tuple[np.ndarray, np.ndarray]): Tuple of image as numpy array and its annotations as numpy array
        """
        if debug:
            self._visualize(*self.executor.fetch(), wait_key)
        else:
            return self.executor.fetch()

    def _get_config(self):
        """
        Retrieves the config from self.input. Differentiates between loading from file and
        loading from API Key.
        """
        if is_api_key(self.input):
            return get_config_from_api(self.input)
        else:
            return load_json(self.input)

    def _init_datasets(self):
        """
        Initializes all datasets defined in the self.config file. Loads annotations and filters in RAM.
        """
        for dataset in self.datasets_config:
            initial_loader = InitialLoader(self.__gen, **dataset[c.CONFIG_KEY_INIT])
            data_packages = initial_loader.load()
            if len(data_packages) == 0:
                warnings.warn(f"Loaded an empty dataset. Please verify your loading query: {dataset[c.CONFIG_KEY_INIT][c.CONFIG_KEY_QUERY]}")
            # Filter key does not exist in config if no filters are provided
            if is_in_dict(c.CONFIG_KEY_FILTER, dataset):
                filters = self._init_filters(dataset[c.CONFIG_KEY_FILTER])
            else:
                filters = None
            if is_in_dict(c.CONFIG_KEY_BACKGROUND_PERCENTAGE, dataset):
                self.datasets.append(
                    Dataset(
                        dataset[c.CONFIG_KEY_ID], data_packages, filters, dataset[c.CONFIG_KEY_BACKGROUND_PERCENTAGE], self.__gen
                    )
                )
            else:
                self.datasets.append(
                    Dataset(
                        dataset[c.CONFIG_KEY_ID], data_packages, filters, None, self.__gen
                    )
                )

    @staticmethod
    def _init_filters(filter_list: List[dict]) -> List[FilterSequence]:
        filters = []
        for filter_dict in filter_list:
            sequence = FilterSequence(filter_dict[c.FILTER_DICT_ID])
            for sequence_dict in filter_dict[c.FILTER_DICT_SEQUENCE]:
                filter_ = Filter(
                    sequence_dict[c.FILTER_DICT_TYPE],
                    sequence_dict[c.FILTER_DICT_SPECIFIER],
                    sequence_dict[c.FILTER_DICT_OPERATOR],
                    sequence_dict[c.FILTER_DICT_VALUE]
                )
                sequence.add(filter_, sequence_dict[c.FILTER_DICT_CHAIN_OPERATOR])
            filters.append(sequence)
        return filters

    @staticmethod
    def _visualize(image: np.ndarray, annots: Annotations, wait_key: int):
        vis = Visualizer(wait_key=wait_key)
        vis.show(image, annots)
































