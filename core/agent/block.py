from typing import Tuple, Optional, Union, Self
from typing import List
from daugx.core.augmentation import augmentations
from daugx.core.augmentation.annotations import Annotations
from daugx.utils import is_executed
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
        # Split shares is a list of all splits. Sum of split shares must equal 1.
        self.__shares = shares
        self._normalize_shares()
        self.__variations = len(self.__shares)
        self.__is_set = True if self.__variations == 1 else False
        # The split index is set when Node is implemented into path. It defines what split is used for this path.
        # The split index also defines the execution probability of this node.
        self.__share = None
        self.__exe_prob = 1

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    @property
    def is_output(self):
        return self.__is_output

    @property
    def is_input(self):
        return self.__is_input

    @property
    def next(self) -> List[str]:
        return self.__next

    @property
    def prev(self) -> List[str]:
        return self.__prev

    @property
    def id(self) -> str:
        return self.__id

    @property
    def inflation(self) -> float:
        return self.__inflation

    @property
    def is_set(self):
        return self.__is_set

    @property
    def variations(self):
        return self.__variations

    @property
    def shares(self):
        return self.__shares

    @property
    def share(self):
        return self.__share

    @property
    def exe_prob(self):
        return self.__exe_prob

    def set_prev(self, prev: List[str]):
        self.__prev = prev

    def set_next(self, next_: List[str]):
        self.__next = next_

    def set_shares(self, shares: List[str]):
        self.__shares = shares

    def mult_exe_prob(self, mult: float):
        """
        Multiplies the execution probability with any multiplier.
        """
        self.__exe_prob *= mult

    def set(self, index: int):
        assert 0 <= index < self.__variations
        self.set_next([self.next[index]])
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

    def set(self, index):
        assert 0 <= index < self._Block__variations
        self.set_next([self.next[index]])
        self.__share = self._Block__shares[index]
        if self.__filters:
            self.__filter = self.__filters[index]
        self.__n_data = self.__n_total_data * self.__share
        self.__is_set = True



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
        self.__exe_prob = exe_prob

        # Initialize augmentation class
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

    @property
    def inflation(self):
        return self.augmentation.inflation

    @property
    def class_name(self):
        return self.__class_name

    @property
    def exe_prob(self):
        return self.__exe_prob

    def is_executed(self):
        """
        Returns: bool - Whether the augmentation is executed or not.
        """
        return is_executed(self.__exe_prob)

    def execute(self, images: List[np.ndarray], annotations: List[Annotations]) -> Tuple[np.ndarray, Optional[Annotations]]:
        """
        Executes the augmentation.
        """
        assert len(images) == len(annotations) == self.__n_inputs
        image, annotations = self.augmentation.apply(images, annotations)
        return image, annotations


class Blocks:

    def __init__(self):
        self.__blocks = []

    def __getitem__(self, id_: str):
        return self._get_block_by_id(id_)

    def build(self, raw_block_list: List[dict]):
        self.__blocks = [self.dict_to_block(raw_block).update() for raw_block in raw_block_list]
        self._set_ipt_blocks_exe_prob()

    @property
    def input_blocks(self):
        return self._get_ipt_blocks()

    def _get_block_by_id(self, id_: str) -> Block or None:
        """
        Gets a Block by an ID.

        Args:
            id_ (str): ID of the Block
        Returns:
            (Block): if a matching Block was found.
            (None): if no Block with matching id was found.
        """
        for block in self.__blocks:
            if block.id == id_:
                return block

    def _set_ipt_blocks_exe_prob(self):
        ipt_blocks = self._get_ipt_blocks()
        n_total_data = sum([ipt_block.n_total_data for ipt_block in ipt_blocks])
        for ipt_block in ipt_blocks:
            ipt_block.mult_exe_prob(ipt_block.n_total_data / n_total_data)

    def _get_ipt_blocks(self) -> List[Input]:
        """
        Returns all blocks of type InputBlock.

        Returns:
            (List[Input]): List of input Blocks
        """
        return [block for block in self.__blocks if isinstance(block, Input)]

    @staticmethod
    def dict_to_block(block_config: dict) -> Block:
        category = block_config[c.NODE_TYPE_STR]
        shares = block_config[c.NODE_SHARE_STR]
        params = block_config[c.NODE_PARAMS_STR]
        id_ = block_config[c.NODE_ID_STR]
        prev = block_config[c.NODE_PREV_STR]
        next_ = block_config[c.NODE_NEXT_STR]
        return Block(id_, prev, next_, shares, category, **params)
