from typing import List, Union

from daugx.core.agent.sequence import Sequence
from daugx.core.agent.workflow import Workflow
from daugx.core.agent.option import Option
from daugx.core.augmentation.annotations import Annotations

import numpy as np


class Executor:
    def __init__(self, workflow: Workflow):
        self.workflow = Workflow

    def execute(
            self,
            image: Union[np.ndarray, List[np.ndarray]],
            annotations: Union[Annotations, List[Annotations]]
    ):
        """
        Executes one Option. All Nodes inside that Option are executed sequentially.
        """
        blocks = self.workflow.fetch

    def __execute_node(self, sequence: Sequence):
        """
        Executes all Nodes inside that sequence
        """
        pass


