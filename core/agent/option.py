"""
This module contains the Sequence class, which is a container for a list of elements.
"""

from typing import List
from .node import Node
from .sequence import Sequence


class Option:
    def __init__(self):
        self.nodes: List[List[Node]] = []
        self.inflation = None
        self.inputs = None
        self.exe_prob = None
        self.exe_prob_sum = None
        self.sequences = []
        self.next_node = None
        self.filter: List[str] = []
        self.is_complete = False

    def add_sequence(self, sequence, next_node):
        self.sequences.append(sequence)
        self.next_node = next_node
        if self.next_node is None:
            self.is_complete = True
        if self.exe_prob is None:
            if isinstance(sequence, Sequence):
                self.exe_prob = sequence.exe_prob
            elif isinstance(sequence, list):
                self.exe_prob = sum([path.exe_prob for path in sequence])
        else:
            self.exe_prob *= sequence.exe_prob

