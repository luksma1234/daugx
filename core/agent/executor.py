from typing import List, Union, Tuple
from copy import deepcopy


import daugx.core.constants as c

from daugx.core.agent.block import Blocks, Block, Augment, Input
from daugx.core.augmentation.annotations import Annotations
from daugx.core.data.data import Dataset
from daugx.utils import new_id, is_executed
from daugx.utils.misc import transpose_image


import numpy as np


class Executor:
    def __init__(self, raw_block_list: List[dict], datasets: List[Dataset], gen: np.random.Generator):

        # Build blocks from raw_block_list
        self.__blocks = Blocks(gen)
        self.__blocks.build(raw_block_list)
        self.__datasets = {dataset.id: dataset for dataset in datasets}
        self.__gen = gen
        self.__data = {}
        self.__path = None

    def fetch(self) -> Tuple[np.ndarray, Annotations]:
        """
        Gets one path to execute for the executor.
        """
        self._reset()
        self.__path = deepcopy(self.__blocks.fetch_path())
        image, annotations = self._execute_path()
        return transpose_image(image), annotations

    def _execute_path(self):
        """
        Executes one path.
        Return:
            (Tuple[np.ndarray, Annotations]): Resulting image and its annotations
        """
        for input_block in list(self.__path[c.PATH_INPUTS].values()):
            for _ in range(input_block.uses):
                data_id = new_id(self.__gen)
                self.__data[data_id] = self.__datasets[input_block.dataset].fetch()
                self._propagate(input_block, data_id)
        return self.__data[c.DATA_OUTPUT]

    def _propagate(self, block: Block, data_id: str):
        """
        Executes block after block until a MIT is reached, or an output. Returns None
        """
        new_data_id = new_id(self.__gen)
        if block.is_input:
            self._exec_input_block(block, data_id, new_data_id)
        else:
            self._exec_augment_block(block, data_id, new_data_id)
            # if input images are empty - this block has been reset at this point and therefore executed
            if block.inflation < 1 and len(block.input_image_ids) != 0:
                return
        if block.is_output:
            self.__data[c.DATA_OUTPUT] = self.__data[new_data_id]
            return
        self._propagate(self.__path[c.PATH_AUGMENTATIONS][block.next[0]], new_data_id)

    @staticmethod
    def _execute_block(
            block: Block,
            image: Union[np.ndarray, List[np.ndarray]],
            annotations: Union[Annotations, List[Annotations]]
    ):
        """
        Executes one Block.
        """
        return block.execute(image, annotations)

    def _reset(self):
        self.__data = {}
        self.__path = None

    def _exec_augment_block(self, block: Augment, data_id: str, new_data_id: str):
        """
        Executed an augmentation block.
        """
        if is_executed(block.int_exe_prob, self.__gen):
            if block.inflation < 1:
                self._exec_inflationary_block(block, data_id, new_data_id)
                # data_ids will be deleted after mit is completed
                return
            else:
                self.__data[new_data_id] = self._execute_block(
                    block,
                    *self.__data[data_id]
                )
        else:
            self.__data[new_data_id] = deepcopy(self.__data[data_id])
        del self.__data[data_id]

    def _exec_inflationary_block(self, block: Augment, data_id: str, new_data_id: str):
        """
        Handles propagation of an inflationary Block.
        """
        assert isinstance(block, Augment)
        # adds input image to mit
        block.add_input_image_id(data_id)
        # checks if enough input images are already created
        if len(block.input_image_ids) == round(1 / block.inflation):
            # create the data list from input image ids
            data_list = [deepcopy(self.__data[image_id]) for image_id in block.input_image_ids]
            # execute mit with list of images and list of annotations
            self.__data[new_data_id] = self._execute_block(
                block,
                [data_package[0] for data_package in data_list],
                [data_package[1] for data_package in data_list]
            )
            for data_id in block.input_image_ids:
                del self.__data[data_id]
            block.reset()
        self.__path[c.PATH_AUGMENTATIONS][block.id] = block


    def _exec_input_block(self, block: Input, data_id: str, new_data_id: str):
        """
        Handles propagation of input blocks
        """
        assert isinstance(block, Input)
        self.__data[new_data_id] = deepcopy(self.__data[data_id])
        del self.__data[data_id]