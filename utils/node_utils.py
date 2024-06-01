from daugx.core.agent.node import (
    Node,
    InputNode,
    OutputNode,
    MergeNode,
    FilterNode,
    SplitNode,
    AugmentationNode,
    DividingNode
)
from . import constants as c


def is_input(node: Node):
    return isinstance(node, InputNode)


def is_output(node: Node):
    return isinstance(node, OutputNode)


def is_filter(node: Node):
    return isinstance(node, FilterNode)


def is_split(node: Node):
    return isinstance(node, SplitNode)


def is_dividing(node: Node):
    return isinstance(node, DividingNode)


def is_inflationary(node: Node):
    return node.inflation < 1


def config_to_node(node_config: dict) -> Node:
    node_type = node_config[c.TYPE_STR]
    params = node_config[c.PARAMS_STR]
    if node_type == c.NODE_TYPE_INPUT:
        return InputNode(**params)
    elif node_type == c.NODE_TYPE_OUTPUT:
        return OutputNode(**params)
    elif node_type == c.NODE_TYPE_MERGE:
        return MergeNode(**params)
    elif node_type == c.NODE_TYPE_SPLIT:
        return SplitNode(**params)
    elif node_type == c.NODE_TYPE_FILTER:
        return FilterNode(**params)
    elif node_type == c.NODE_TYPE_AUGMENTATION:
        return AugmentationNode(**params)
    else:
        raise ValueError(f"Node Type '{node_type}' is unknown.")
