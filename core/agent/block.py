from copy import copy
from typing import Tuple, Optional, Union, Self
from typing import List, Dict

from daugx.core.augmentation import augmentations
from daugx.core.augmentation.annotations import Annotations
from daugx.utils import is_executed, new_id, fetch_by_prob_list, is_in_dict

import numpy as np
import daugx.core.constants as c


class Block:
    def __init__(self, id_: str, prev: List[str], next_: List[str], shares: List[float], category: str, **kwargs):
        """
        Args:
            id_: id of the block
            prev: list of the ids of all previous blocks
            next: list of the ids of all next blocks
            inflation:
            shares: list of the percentage share on the data flowing for each next block id
        """
        self.__next = next_
        self.__prev = prev
        self.__id: str = id_
        # inflation describes how this block interacts with data. A negative inflation essentially deflates the data and
        # results in less output than input. E.g.: If we have an inflation of 0.25, we expect one output per 4 inputs.
        self.__inflation = 1
        self.__is_output = not self.__next
        self.__is_input = not self.__prev
        self.__category = category
        self.__params = kwargs
        # Shares is a list of all splits. Sum of split shares must equal 1.
        self.__shares = shares
        self._normalize_shares()
        self.__variations = len(self.__shares)
        self.__is_set = True if self.__variations == 1 else False
        self.__share = None

        # external execution probability
        # set by external factors like a split or a dataset share
        self.__ext_exe_prob = 1
        # internal execution probability
        # set by the block definition
        self.__int_exe_prob = 1

        self.__prev_ext_exe_prob = None
        self.__input_image_ids = []

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __str__(self):
        return (f"\n---{self.id}---\nexternal_execution_probability: {self.__ext_exe_prob}"
                f"\ninternal_execution_probability: {self.__int_exe_prob}\nis_input: {self.__is_input}\n"
                f"is_output: {self.__is_output}\nnext: {self.__next}\nprev: {self.__prev}\ninflation: {self.inflation}"
                f"\nshare: {self.__share}\n" + "_" * 50)

    @property
    def is_output(self):
        return self.__is_output

    @property
    def is_input(self):
        return self.__is_input

    @property
    def next(self) -> List[str]:
        return self.__next

    @next.setter
    def next(self, value: List[str]):
        assert isinstance(value, list)
        self.__next = value

    @property
    def prev(self) -> List[str]:
        return self.__prev

    @prev.setter
    def prev(self, value: List[str]):
        assert isinstance(value, list)
        self.__prev = value

    @property
    def prev_ext_exe_probs(self) -> List[float]:
        return self.__prev_ext_exe_prob

    @prev_ext_exe_probs.setter
    def prev_ext_exe_probs(self, value: List[float]):
        assert isinstance(value, list)
        self.__prev_ext_exe_prob = value

    @property
    def id(self) -> str:
        return self.__id

    @id.setter
    def id(self, id_: str):
        self.__id = id_

    @property
    def inflation(self) -> float:
        return self.__inflation

    @property
    def is_set(self):
        return self.__is_set

    @property
    def variations(self) -> int:
        return self.__variations

    @property
    def shares(self) -> List[float]:
        return self.__shares

    @property
    def share(self):
        return self.__share

    @property
    def ext_exe_prob(self):
        return self.__ext_exe_prob

    @ext_exe_prob.setter
    def ext_exe_prob(self, value: float):
        assert isinstance(value, float) or value == 1
        assert 0 <= value <= 1
        self.__ext_exe_prob = value

    @property
    def int_exe_prob(self) -> float:
        return self.__int_exe_prob

    @int_exe_prob.setter
    def int_exe_prob(self, value: float):
        assert isinstance(value, float) or value == 1
        assert 0 <= value <= 1
        self.__int_exe_prob = value

    @property
    def input_image_ids(self):
        return self.__input_image_ids

    def add_input_image_id(self, input_image_id: str):
        self.__input_image_ids.append(input_image_id)

    def get_prev(self):
        if self.is_input:
            return None

    def add_prev(self, prev: str):
        self.__prev.append(prev)

    def set_shares(self, shares: List[str]):
        self.__shares = shares

    def mult_exe_prob(self, mult: float):
        """
        Multiplies the external execution probability by any multiplier.
        """
        self.__ext_exe_prob *= mult

    def set(self, index: int):
        assert 0 <= index < self.__variations
        if not self.__is_output:
            self.__next = [self.__next[index]]
        self.__share = self.__shares[index]
        self.mult_exe_prob(self.__share)
        self.__is_set = True

    def _normalize_shares(self):
        for share in self.__shares:
            share *= (1 / sum(self.__shares))

    def execute(self, image: Union[np.ndarray, List[np.ndarray]], annotations: Union[Annotations, List[Annotations]]) \
            -> Tuple[np.ndarray, Optional[Annotations]]:
        """
        Returns image and annotations without changes by default. Executes augmentation if possible.
        """
        return image, annotations

    def update(self) -> Self:
        match self.__category:
            case c.NODE_TYPE_INPUT:
                return Input(self.__id, self.__next, self.__shares, **self.__params)
            case c.NODE_TYPE_AUGMENT:
                return Augment(self.__id, self.__prev, self.__next, self.__shares, **self.__params)

    def reset(self):
        self.__input_image_ids = []


class Input(Block):
    def __init__(self, id_: str, next_: List[str], shares: List[float], dataset: str, n_total_data: int, filters: Optional[List[str]]):
        """
        Dataset Block. This defines what data how to load from where.
        """
        super().__init__(id_, [], next_, shares, c.NODE_TYPE_INPUT)
        self.__n_total_data = n_total_data
        self.__dataset = dataset
        self.__filters = filters
        if self.__filters:
            assert len(self.__filters) == self.variations
        # n data, filter and load_id is defined when InputBlock gets set
        self.__n_data = None
        self.__filter = None
        self.__uses = 1

    def __eq__(self, other):
        # input classes cannot be equal except they are intentionally build to be equal
        return False

    @property
    def dataset(self):
        return self.__dataset

    @property
    def n_data(self):
        return self.__n_data

    @property
    def n_total_data(self):
        return self.__n_total_data

    @property
    def filter(self):
        return self.__filter

    @property
    def uses(self):
        return self.__uses

    def add_use(self):
        self.__uses += 1

    def set(self, index):
        assert 0 <= index < self.variations
        self.next = [self.next[index]]
        self.__share = self.shares[index]
        if self.__filters:
            self.__filter = self.__filters[index]
        self.__n_data = self.__n_total_data * self.__share
        self.__is_set = True

    def reset(self):
        self.__input_image_ids = []
        self.__uses = 0


class Augment(Block):
    def __init__(self, id_: str, prev: List[str], next_: List[str], shares: List[float], class_name: str, exe_prob: float, **kwargs):
        """
        Initializes an Element object.
        Args:
            class_ (str): The name of the augmentation class. See daugx.core.augmentation.augmentations.py for reference
            params (dict):  - The parameters of the augmentation.
            execution_probability (float): The probability of the augmentation being executed.
        """
        super().__init__(id_, prev, next_, shares, c.NODE_TYPE_AUGMENT)
        self.__class_name = class_name
        self.int_exe_prob = exe_prob
        # Try to find class name in augmentations
        try:
            self.augmentation = getattr(augmentations, self.class_name)(**kwargs)
        except AttributeError:
            raise AttributeError(f"The augmentation '{self.class_name}' is unknown. Please make sure your"
                                 f"clients version matches with the library version.")
        except TypeError:
            raise TypeError(f"One or more arguments of '{kwargs}' are not allowed for augmentation {self.class_name}'")
        except Exception as e:
            raise e

        self.__n_inputs = int(1 / self.inflation)

    def __eq__(self, other):
        if not isinstance(other, Augment):
            return False
        return (other.augmentation == self.augmentation and other.int_exe_prob == self.int_exe_prob
                and other.share == self.share)


    @property
    def inflation(self) -> float:
        return self.augmentation.inflation

    @property
    def class_name(self) -> str:
        return self.__class_name

    def execute(self, images: List[np.ndarray], annotations: List[Annotations]) -> Tuple[np.ndarray, Optional[Annotations]]:
        """
        Executes the augmentation.
        """
        # assert len(images) == len(annotations) == self.__n_inputs
        image, annotations = self.augmentation.apply(images, annotations)
        return image, annotations


class Blocks:

    def __init__(self, gen: np.random.Generator):
        self.__blocks = []
        self.__gen = gen

    def __getitem__(self, id_: str):
        return self._get_block_by_id(self.__blocks, id_)

    def __str__(self):
        string = f"n_blocks: {len(self.__blocks)}\n"
        string += "_" * 50
        for block in self.__blocks:
            string += str(block)
        return string

    def fetch_path(self) -> Dict[str, Dict[str, Union[Input, Augment]]]:
        # TODO: Something is wrong here. Uses do not match with actually uses necessary
        """
        Fetches one path. The schema of a path looks like the following:
        {
            "inputs": {
                "input_block_1_ID": Input_Block_1,
                "input_block_1_ID": Input_Block_2,
                ...
            },
            "augmentations": {
                "Augmentation_Block_1_ID": Augmentation_Block_1,
                "Augmentation_Block_2_ID": Augmentation_Block_2,
                ...
            }
        }
        """
        self.reset()
        output_blocks = self._get_output_blocks(self.__blocks)
        # chose one output block
        block = fetch_by_prob_list(
            output_blocks,
            [output_block.ext_exe_prob for output_block in output_blocks],
            self.__gen
        )
        path_blocks = self.root(block)
        return {
            c.PATH_INPUTS: {
                input_block.id: input_block for input_block in self._get_input_blocks(list(path_blocks.values()))
            },
            c.PATH_AUGMENTATIONS: {
                augmentation_block.id: augmentation_block for augmentation_block in self._get_augment_blocks(list(path_blocks.values()))
            }
        }

    def root(self, block: Block) -> Dict[str, Block]:
        """
        Walks downstream until input block is reached. Returns dict of block ID and block object pairs of all blocks
        passed by.
        """
        blocks = {block.id: block}
        if not block.is_input:
            # handle inflationary sub paths
            if block.inflation < 1:
                for variant_index in range(round(1 / block.inflation)):
                    # chose one variant
                    chosen_block_id = fetch_by_prob_list(
                        block.prev,
                        [block.prev_ext_exe_probs[index] / sum(block.prev_ext_exe_probs)
                         for index, _ in enumerate(block.prev_ext_exe_probs)],
                        self.__gen
                    )
                    # add blocks one by one, this makes sure we can add one use to duplicate input blocks
                    for variant_block_id, variant_block in self.root(self._get_block_by_id(self.__blocks, chosen_block_id)).items():
                        if not is_in_dict(variant_block_id, blocks):
                            blocks[variant_block_id] = variant_block
                        # if block duplicate is input block - add use
                        if variant_block.is_input:
                            # now block with variant block id is input block and therefore has the add_use method.
                            blocks[variant_block_id].add_use()
            else:
                chosen_block_id = fetch_by_prob_list(
                    block.prev,
                    [block.prev_ext_exe_probs[index] / sum(block.prev_ext_exe_probs)
                     for index, _ in enumerate(block.prev_ext_exe_probs)],
                    self.__gen
                )
                blocks.update(self.root(self._get_block_by_id(self.__blocks, chosen_block_id)))
        return blocks

    def build(self, raw_block_list: List[dict]):
        raw_blocks = [self._dict_to_block(raw_block).update() for raw_block in raw_block_list]
        raw_ipt_blocks = self._set_ipt_blocks_exe_prob(self._get_input_blocks(raw_blocks))
        for block in raw_ipt_blocks:
            self._build_from_block(block, raw_blocks)
        for block in self._get_output_blocks(self.__blocks):
            self._calc_ext_exe_probs(block)

    def _build_from_block(self, raw_block: Block, raw_blocks: List[Block], share_index=0, prev_block_id=None):
        """
        Build from a given Block upwards.
        1. Assigns a new ID to a copy of the given Block.
        2. Builds all available Block variants
        3. Sets Block with share index
        4. 
        """
        block_index = None
        built_block = copy(raw_block)
        built_block.id = new_id(self.__gen)
        # build all block variations by setting blocks with all available shares
        if len(built_block.shares) > (share_index + 1):
            self._build_from_block(raw_block, raw_blocks, share_index + 1, prev_block_id)
        built_block.set(share_index)
        # if this built block is not unique - replace this with the existing block
        if not self._is_unique(built_block):
            block_index = self._get_duplicate_index(built_block)
            built_block = self.__blocks[block_index]
        # set this blocks ID for previous block as next block ID and add previous ID to this block
        if prev_block_id is not None:
            prev_block = self._get_block_by_id(self.__blocks, prev_block_id)
            prev_block.next = [built_block.id]
            if block_index is None:
                built_block.prev = [prev_block_id]
            elif prev_block_id not in built_block.prev:
                built_block.add_prev(prev_block_id)
        if block_index is not None:
            self.__blocks[block_index] = built_block
        else:
            self.__blocks.append(built_block)
        if built_block.next:
            next_raw_block = self._get_block_by_id(raw_blocks, built_block.next[0])
            if next_raw_block is not None:
                self._build_from_block(next_raw_block, raw_blocks, prev_block_id=built_block.id)

    def _calc_ext_exe_probs(self, block):
        if not block.is_input:
            prev_blocks = [self._get_block_by_id(self.__blocks, id_) for id_ in block.prev]
            for prev_block in prev_blocks:
                self._calc_ext_exe_probs(prev_block)
            prev_ext_exe_probs = [prev_block.ext_exe_prob for prev_block in prev_blocks]
            block.prev_ext_exe_probs = prev_ext_exe_probs
            ext_exe_prob_sum = sum(prev_ext_exe_probs)
            block.ext_exe_prob = ext_exe_prob_sum * block.ext_exe_prob

    def _is_unique(self, new_block: Block) -> bool:
        """
        Verifies uniqueness of provided block.
        """
        for block in self.__blocks:
            if block == new_block:
                return False
        return True

    def _get_duplicate_index(self, block: Block):
        """
        Gets the duplicate of the provided block. Returns None if no duplicate exists
        """
        for index, block_ in enumerate(self.__blocks):
            if block == block_:
                return index


    @staticmethod
    def _dict_to_block(block_config: dict) -> Block:
        """
        Parses the provided dict into a 'Block' object.
        """
        category = block_config[c.NODE_TYPE_STR]
        shares = block_config[c.NODE_SHARE_STR]
        params = block_config[c.NODE_PARAMS_STR]
        id_ = block_config[c.NODE_ID_STR]
        prev = block_config[c.NODE_PREV_STR]
        next_ = block_config[c.NODE_NEXT_STR]
        return Block(id_, prev, next_, shares, category, **params)

    @staticmethod
    def _get_input_blocks(blocks: List[Block]) -> List[Input]:
        """
        Returns all blocks of type 'Input' in the provided list.
        """
        return [block for block in blocks if isinstance(block, Input)]

    @staticmethod
    def _get_output_blocks(blocks: List[Block]) -> List[Block]:
        """
        Returns all output blocks in the provided list.
        """
        output_blocks = [block for block in blocks if block.is_output]
        if not output_blocks:
            raise ValueError("No output blocks defined in config.")
        return output_blocks

    @staticmethod
    def _get_augment_blocks(blocks: List[Block]) -> List[Block]:
        """
        Returns all output blocks in the provided list.
        """
        return [block for block in blocks if isinstance(block, Augment)]

    @staticmethod
    def _set_ipt_blocks_exe_prob(ipt_blocks: List[Input]) -> List[Input]:
        """
        Calculates the execution probability for all input blocks provided,
        by their share on the sum of available data.
        Args:
            ipt_blocks (List[Input]): Provided Input Blocks
        """
        n_total_data = sum([ipt_block.n_total_data for ipt_block in ipt_blocks])
        for ipt_block in ipt_blocks:
            ipt_block.mult_exe_prob(ipt_block.n_total_data / n_total_data)
        return ipt_blocks

    @staticmethod
    def _get_block_by_id(blocks: List[Block], id_: str) -> Optional[Block]:
        """
        Gets a Block by an ID.
        Args:
            id_ (str): ID of the Block
        Returns:
            (Block): if a matching Block was found.
            (None): if no Block with matching id was found.
        """
        for block in blocks:
            if block.id == id_:
                return block

    def reset(self):
        for block in self.__blocks:
            block.reset()