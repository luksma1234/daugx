from typing import List, Union


from daugx.core.agent.sequence import Sequence
from daugx.core.agent.path import Path
from daugx.core.augmentation.annotations import Annotations

import numpy as np

# TODO: How are the sequences embedded inside the paths? How do I know when to execute in parallel? Are there sub paths?


class Executor:
    def __init__(self):
        self.image_data: Union[List[np.ndarray], np.ndarray, None] = None
        self.annotation_data: Union[List[Annotations], Annotations, None] = None

    def execute_path(self, path: Path):
        """
        Executes a Path by executing all sequences.
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
