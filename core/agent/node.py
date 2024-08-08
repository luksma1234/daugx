from daugx.utils import *
from typing import List
from daugx.core.augmentation import augmentations
from daugx.utils import is_executed
import numpy as np


class Node:
    def __init__(self, id_: int):
        self.next = None
        self.inflation = None
        self.id = id_

    def set_next(self, next_):
        self.next = next_


class InputNode(Node):
    def __init__(self, id: int, n_data: int, path: str, data_type: str):
        super().__init__(id)
        self.inflation = 1
        self.n_data = n_data
        self.path = path
        self.data_type = data_type


class OutputNode(Node):
    def __init__(self, id: int):
        super().__init__(id)
        self.inflation = 1


class MergeNode(Node):
    def __init__(self, id: int):
        super().__init__(id)
        self.inflation = 1


class DividingNode(Node):
    def __init__(self, id: int, split_shares: List[float]):
        super().__init__(id)
        self.split_shares = split_shares
        self.inflation = 1
        # The split index is set when Node is implemented into path. It defines what split is used for this path.
        self.split_idx = None
        # The split index also defines the execution probability of this node.
        self.exe_prob = None

    def set_split_idx(self, split_idx: int):
        self.split_idx = split_idx
        self.exe_prob = self.split_shares[self.split_idx]


class SplitNode(DividingNode):
    def __init__(self, id: int, split_shares: List[float]):
        super().__init__(id, split_shares)
        self.split_shares = split_shares
        self.inflation = 1
        # The split index is set when Node is implemented into path. It defines what split is used for this path.
        self.split_idx = None
        # The split index also defines the execution probability of this node.
        self.exe_prob = None


class FilterNode(DividingNode):
    def __init__(self, id: int, split_shares: List[float]):
        super().__init__(id, split_shares)
        self.split_shares = split_shares
        self.inflation = 1
        # The split index is set when Node is implemented into path. It defines what split is used for this path.
        self.split_idx = None
        # The split index also defines the execution probability of this node.
        self.exe_prob = None


class AugmentationNode(Node):
    def __init__(
            self,
            method: str,
            id: int,
            p: float = 1,
            **kwargs
    ):
        """
        Initializes an Element object.
        Args:
            method: str - The type of the augmentation.
            params: dict - The parameters of the augmentation.
            execution_probability: float - The probability of the augmentation being executed.
        """
        super().__init__(id)
        self.method = method
        self.p = p

        # Initialize augmentation method
        try:
            self.augmentation = getattr(augmentations, self.method)(**kwargs)
        except AttributeError:
            raise UnknownAugmentationError(f"The augmentation method '{self.method}' is unknown. Please make sure your"
                                           f"clients version matches with the library version.")
        except TypeError:
            raise UnknownArgumentError(f"One or more arguments of '{kwargs}' are not allowed for augmentation of "
                                       f"method '{self.method}'")
        except Exception as e:
            raise e

        self.inflation = self.augmentation.inflation

        assert self.inflation <= 1, f"Something went wrong - augmentations can only deflate data."
        self.p *= self.inflation
        self.n_inputs = int(1 / self.inflation)
        self.input = None
        self.output_id = None

    def is_executed(self):
        """
        Returns: bool - Whether the augmentation is executed or not.
        """
        return is_executed(self.p)

    def activate(self, input_ids: List[str]):
        """
        Activates the element -> prepares the element for execution. Sets input and output
        Args:
            input_ids: List of input ids. Ids are generated using the new_id() function.
        Returns: None
        """
        self.input = input_ids
        if len(self.input) > 1:
            self.output_id = new_id()
        else:
            self.output_id = self.input[0]

    def execute(self, data: np.ndarray) -> np.ndarray:
        """
        Executes the augmentation on the given data.
        Args:
            data: np.ndarray - The data to be augmented.

        Returns: np.ndarray - The augmented data.
        """
        return self.augmentation.execute(data)
