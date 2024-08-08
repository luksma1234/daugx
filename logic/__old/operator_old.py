"""

"""
from typing import List
from ..utils.misc import new_id
from .node import Node


class OperatorNode(Node):
    def __init__(self, succeeding: str):
        super().__init__(succeeding)
        self.inflation = None


class InputOperator(OperatorNode):
    def __init__(self, next_: str, n_data: int):
        super().__init__(next_)
        self.inflation = 1
        self.n_data = n_data


class OutputOperator(OperatorNode):
    def __init__(self, next_=None):
        super().__init__(next_)
        self.inflation = 1


class MergeOperator(OperatorNode):
    def __init__(self, next_: str):
        super().__init__(next_)
        self.inflation = 1


class SplitOperator(OperatorNode):
    def __init__(self, next_: str, splits: int):
        super().__init__(next_)
        self.inflation = 1 / splits


class FilterOperator(OperatorNode):
    def __init__(self, next_: str, splits: int, independent: bool):
        super().__init__(next_)
        self.inflation = 1 / splits
