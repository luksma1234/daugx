from .block import Node
from typing import List


class Sequence:
    def __init__(self):
        self.__nodes = []
        self.__next_node: Node or None = None
        self.exe_prob = 1

    def __len__(self):
        return len(self.__nodes)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.__nodes[index]
        else:
            raise IndexError()

    @property
    def next_node(self):
        return self.__next_node

    @property
    def end_node(self):
        return self.__nodes[-1]

    @property
    def start_node(self):
        return self.__nodes[0]

    def add_node(self, node: Node):
        self.__nodes.append(node)

    def set_next_node(self, next_node: Node):
        self.__next_node = next_node
