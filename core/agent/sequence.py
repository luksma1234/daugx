from .node import Node
from typing import List


class Sequence:
    def __init__(self):
        self.nodes = []
        self.next_node: Node or None = None
        self.exe_prob = 1

    def __len__(self):
        return len(self.nodes)

    def add_node(self, node: Node):
        self.nodes.append(node)

    def get_end_node(self):
        return self.nodes[-1]

    def get_start_node(self):
        return self.nodes[0]

    def set_next_node(self, next_node: Node):
        self.next_node = next_node
