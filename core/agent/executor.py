from typing import List, Union


from daugx.core.agent.sequence import Sequence
from daugx.core.agent.option import Option
from daugx.core.augmentation.annotations import Annotations

import numpy as np


class Executor:
    def __init__(self):
        pass

    def execute(self, path: Option, data: Union[List[np.ndarray], np.ndarray]):
        """
        Executes a Path with the given data.
        """


    def execute_single(self, sequence: Sequence):
        """
        Executes all Nodes inside that sequence
        """
        pass

    def execute_parallel(self, sequences: List[Sequence]):
        """
        Executes all elements in the sequences list in parallel. Starts a new process for each sequence.
        """
        pass
