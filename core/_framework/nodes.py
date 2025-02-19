from typing import Optional, List, Self, Dict

import daugx.core.constants as c
from daugx.utils import new_id, fetch_by_prob_list, norm_list

import numpy as np


class Node:
    # Limit RAM use by predefining all class variables
    # Is it possible to predefine the datatype and therefore use even less RAM?
    __slots__ = (
        "__next", "__prev", "__id", "__data_id", "__inflation", "__category", "__is_output", "__is_input",
         "__is_set", "__shares", "__variations", "__derives_from", "__share", "__int_exe_prob",
         "__ext_exe_prob", "__prev_ext_exe_probs", "__input_origin", "__uses"
    )

    def __init__(
            self,
            id_: str,
            prev: List[str],
            next_: List[str],
            shares: List[float],
            inflation: int,
            category: str,
            p: float = 1.0,
            derives_from: Optional[str] = None,
            data_id: Optional[str] = None
    ):
        """
        Nodes are created for each augmentation and input of the workflow.
        Args:
            id_ (str): ID of the Node
            prev (List[str]): List of all previous node IDs
            next_ (List[str]): List of all next node IDs
            shares (List[float]): Probabilities for each next node
            inflation (int): The amount of data origins necessary to pass this node
            p (float): The execution probability of this node
            derives_from (Optional[str]): The ID of the Node this Node derives from
            data_id (Optional[str]): The id from which the params for this node can be extracted
        """
        self.__next: List[str] = next_
        self.__prev: List[str] = prev
        self.__id: str = id_
        self.__data_id: Optional[str] = data_id
        if self.__data_id is None:
            self.__data_id = self.__id
        self.__inflation: int = inflation
        self.__category = category
        self.__is_output: bool = not self.__next
        self.__is_input: bool = not self.__prev
        # A Node is set if it has None or exactly one next Node
        self.__is_set: bool = len(self.__next) <= 1
        # Shares is a list of all splits. Sum of split shares must equal 1.
        self.__shares: List[float] = norm_list(shares)
        self.__variations: int = len(self.__shares)
        self.__derives_from: Optional[str] = derives_from
        self.__share: Optional[float] = None
        # Internal execution probability
        self.__int_exe_prob: float = p
        # External execution probability
        self.__ext_exe_prob: float = 1.0
        self.__prev_ext_exe_probs = None
        self.__input_origin: Optional[List[str]] = None
        self.__uses = 1

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__key() == other.__key()
        else:
            return False

    def __key(self):
        return self.data_id, self.inflation, self.derives_from, self.share, self.int_exe_prob, self.ext_exe_prob

    @property
    def is_output(self):
        return self.__is_output

    @property
    def is_input(self):
        return self.__is_input

    @property
    def next(self) -> List[str]:
        return self.__next

    @next.setter
    def next(self, value: List[str]):
        assert isinstance(value, list)
        self.__next = value
        self.__is_output = not self.__next

    @property
    def prev(self) -> List[str]:
        return self.__prev

    @prev.setter
    def prev(self, value: List[str]):
        assert isinstance(value, list)
        self.__prev = value
        self.__is_input = not self.__prev

    @property
    def id(self) -> str:
        return self.__id

    @id.setter
    def id(self, id_: str):
        self.__id = id_

    @property
    def inflation(self) -> int:
        return self.__inflation

    @property
    def is_set(self):
        return self.__is_set

    @property
    def variations(self) -> int:
        return self.__variations

    @property
    def shares(self) -> List[float]:
        return self.__shares

    @property
    def share(self):
        return self.__share

    @property
    def ext_exe_prob(self):
        return self.__ext_exe_prob

    @ext_exe_prob.setter
    def ext_exe_prob(self, value: float):
        assert isinstance(value, float) or value == 1
        assert 0 <= value <= 1
        self.__ext_exe_prob = value

    @property
    def int_exe_prob(self) -> float:
        return self.__int_exe_prob

    @int_exe_prob.setter
    def int_exe_prob(self, value: float):
        assert isinstance(value, float) or value == 1
        assert 0 <= value <= 1
        self.__int_exe_prob = value

    @property
    def input_origin(self):
        return self.__input_origin

    @property
    def data_id(self):
        return self.__data_id

    @property
    def derives_from(self):
        return self.__derives_from

    @property
    def prev_ext_exe_probs(self):
        return self.__prev_ext_exe_probs

    @prev_ext_exe_probs.setter
    def prev_ext_exe_probs(self, value: List[float]):
        assert isinstance(value, list)
        self.prev_ext_exe_probs = value

    @property
    def norm_prev_ext_exe_probs(self):
        if self.__prev_ext_exe_probs is not None:
            return norm_list(self.__prev_ext_exe_probs)
        return None

    @property
    def uses(self):
        return self.__uses

    @property
    def category(self):
        return self.__category

    def add_use(self):
        self.__uses += 1

    def add_origin(self, input_id: str):
        self.__input_origin.append(input_id)

    def set(self, index: int):
        """
        Sets the node with a share index. Since multiple nodes can branch off one node, and for each branch a
        probability can be defined, the share index defines into what next node this node will lead. This creates as
        many variations of a node as there are next nodes.

        Args:
            index (int): The share index to be set
        """
        assert 0 <= index < self.__variations
        if self.is_set:
            return
        if not self.__is_output:
            self.__next = [self.__next[index]]
        self.__share = self.__shares[index]
        self.ext_exe_prob *= self.__share
        self.__is_set = True

    def reset(self):
        self.__input_origin = None

    def derive(self, derive_id: str, share_index: int) -> Self:
        """
        Creates a new instance of the Node class which derives from this instance. The derived class is set and the
        share is included in its external execution probability.

        Args:
            derive_id (str): The ID for the derived class (typically a new generated ID)
            share_index (int): The index to set the derived class to
        """
        derivative = Node(
            id_=derive_id,
            prev=self.prev,
            next_=[self.next[share_index]],
            shares=[1],
            inflation=self.inflation,
            p=self.int_exe_prob,
            derives_from=self.id,
            data_id=self.data_id,
            category=self.category
        )
        # set initial external execution probability and set derivative
        derivative.ext_exe_prob = self.ext_exe_prob
        derivative.set(share_index)
        return derivative


class Branch:
    def __init__(self):
        # dict with node IDs as key and the node itself as value
        self.__nodes: Dict[str, Node] = {}
        self.__node_data: Dict[str, dict] = {}
        self.__current_node: Optional[Node] = None
        self.__inputs: Optional[List[Node]] = None
        self.__input_index: int = 0
        self.__input_uses: int = 0

    def __iter__(self):
        return self

    def __next__(self):
        """
        Always returns the next node to be executed. Stops Iteration after Output Node.
        """
        if self.current_node is not None:
            if self.current_node.is_output:
                raise StopIteration
            next_node = self._get_node_by_id(self.current_node.next[0])
            next_node.add_origin(self.current_node.id)
            if len(next_node.input_origin) < next_node.inflation:
                self.__current_node = self._next_input()
                return self.current_node, self.current_data
            self.__current_node = next_node
            return self.current_node
        else:
            self._init_inputs()
            self.__current_node = self._next_input()
            return self.current_node

    @property
    def nodes(self):
        return self.__nodes

    @property
    def current_node(self):
        return self.__current_node

    @property
    def current_data(self):
        if self.current_node is None:
            return None
        return self.__node_data[self.current_node.id]

    @property
    def inputs(self):
        return self.__inputs

    def update(self, branch: Self):
        """
        Merges another branch into this branch. Adds all nodes of the other branch to this branch.
        Adds use to node if it already exists in this branch.

        Args:
            branch (Branch): The branch to be merged into this branch
        """
        assert isinstance(branch, Branch)
        self._reset()
        for node, node_data in zip(branch.nodes.values(), branch.__node_data.values()):
            if self._has_node(node):
                self._add_use(node)
            else:
                self._add(node, node_data)

    def _init_inputs(self):
        self.__inputs = [node for node in self.__nodes.values() if node.is_input]

    def _next_input(self):
        """
        Returns the logical next input from available inputs. Makes sure input uses are taken into account.
        """
        assert len(self.inputs) < self.__input_index
        next_input = self.inputs[self.__input_index]
        if next_input.uses > self.__input_uses:
            self.__input_uses += 1
        else:
            self.__input_uses = 0
            self.__input_index += 1
        return next_input

    def _add(self, node: Node, node_data: dict):
        """
        Adds one node to this branch. Validates dtype of node.

        Args:
            node (Node): The node to be added
            node_data (dict): The data for this node
        """
        assert isinstance(node, Node)
        self.__nodes[node.id] = node
        self.__node_data[node.id] = node_data

    def _has_node(self, node: Node) -> bool:
        """
        Checks if a node ID exists in this branch.
        Args:
            node (Node): The node to be checked

        Returns:
            (bool) True if node was found in this branch
            (bool) False if node was not found in branch
        """
        return node.id in self.__nodes

    def _add_use(self, node: Node):
        """
        Adds one use to the specified Node.
        Args:
            node (Node): The node to add one use to
        """
        assert node.is_input
        self.__nodes[node.id].add_use()

    def _get_node_by_id(self, node_id: str):
        """
        Gets a node by its ID.
        """
        for node in self.__nodes.values():
            if node.id == node_id:
                return node
        raise ValueError

    def _reset(self):
        self.__current_node = None

class Tree:
    def __init__(self, rng: np.random.Generator, raw_nodes: List[dict]):
        self.__rng = rng
        self.__base_nodes = []
        self.__derived_nodes = []
        self.__node_params = {}
        self.__node_mapping = {}
        self._init_base_nodes(raw_nodes)
        self._init_node_mapping()
        self._grow()
        self.__outputs = self._get_outputs(self.__derived_nodes)
        for output in self.__outputs:
            self._init_ext_exe_probs(output)

    def branch(self) -> Branch:
        """
        Fetches one branch.
        """
        return self._root(
            fetch_by_prob_list(
                self.__outputs,
                [output_block.ext_exe_prob for output_block in self.__outputs],
                self.__rng
            )
        )

    def _root(self, node: Node) -> Branch:
        """
        Walks Tree downstream until this node and all its predecessors are satisfied with connections according to their
        inflation value. Returns one branch.

        Args:
            node (Node): The node to start rooting from

        Returns:
            (Branch): The branch downwards from node
        """
        branch = Branch()
        if not node.is_input:
            if node.inflation > 1:
                self._root_inflationary(node, branch)
            else:
                variant_id = self._random_variant_id(node)
                branch.update(self._root(self._get_derived_node_by_id(variant_id)))
        return branch

    def _root_inflationary(self, node: Node, branch: Branch):
        """
        Roots an inflationary node
        """
        for variant_index in range(node.inflation):
            variant_id = self._random_variant_id(node)
            # Update this branch with root-branch of node
            branch.update(self._root(self._get_derived_node_by_id(variant_id)))
        return branch

    def _random_variant_id(self, node):
        """
        Chooses one previous node id from node (variant).

        Args:
            node (Node): Node to chose variant from

        Returns:
            (str): ID of the chosen variant
        """
        assert not node.is_input
        return fetch_by_prob_list(
            node.prev,
            node.norm_prev_ext_exe_probs,
            self.__rng
        )

    def _grow(self):
        """
        Grows the node tree from base input nodes. Explores each base input node. After exploring all nodes, external
        execution probabilities are calculated and initialized for each derived node.

        Returns:
            None
        """
        base_inputs = self._get_inputs(self.__base_nodes)
        for node in base_inputs:
            self._explore(node)
        for opt_node in self._get_outputs(self.__derived_nodes):
            self._init_ext_exe_probs(opt_node)

    def _explore(self, base_node: Node, share_index: int = 0, next_share_index: int = 0):
        """
        Explores a node on a list of nodes. Creates a derivative for each base node.
        """
        derivative = base_node.derive(self.__node_mapping[base_node.id][share_index], share_index)
        # Get all node variations of the next node
        next_base_node = self._get_base_node_by_id(base_node.next[share_index])
        next_shares = next_base_node.shares
        # explore all variations of this node with all combinations of the next node
        if len(base_node.shares) > (share_index + 1):
            for index, share in enumerate(next_shares):
                self._explore(base_node, share_index + 1, index)
        self._map_derivative(derivative, next_share_index)
        self.__derived_nodes.append(derivative)
        self._explore(next_base_node)

    def _map_derivative(self, derivative: Node, next_share_index: int) -> Node:
        """
        Maps base -previous and -next nodes of derivatives to the correct variants.
        """
        # Get previous for derivative, by selecting the variant, where derivative is next
        base_prev = derivative.prev
        base_prev_next_indices = [
            self._get_base_node_by_id(prev).next.index(derivative.derives_from) for prev in base_prev
        ]
        derivative_prev = [
            self.__node_mapping[base_prev_id][prev_next_index] for (base_prev_id, prev_next_index) in
            zip(base_prev, base_prev_next_indices)
        ]
        derivative.prev = derivative_prev
        # Get next for derivative
        base_next = derivative.next
        derivative.next = self.__node_mapping[base_next][next_share_index]
        return derivative

    def _get_base_node_by_id(self, base_node_id: str):
        """
        Gets a base node by its ID.
        """
        for node in self.__base_nodes:
            if node.id == base_node_id:
                return node
        raise ValueError

    def _get_derived_node_by_id(self, derived_node_id: str):
        """
        Gets a derived node by its ID.
        """
        for node in self.__derived_nodes:
            if node.id == derived_node_id:
                return node
        raise ValueError

    def _init_ext_exe_probs(self, node: Node):
        """
        Calculates all external execution probabilities downwards from given node.
        """
        if not node.is_input:
            prev_nodes = [self._get_derived_node_by_id(id_) for id_ in node.prev]
            for prev_node in prev_nodes:
                self._init_ext_exe_probs(prev_node)
            prev_ext_exe_probs = [prev_node.ext_exe_prob for prev_node in prev_nodes]
            node.prev_ext_exe_probs = prev_ext_exe_probs
            ext_exe_prob_sum = sum(prev_ext_exe_probs)
            node.ext_exe_prob = ext_exe_prob_sum * node.ext_exe_prob

    def _dict_to_node(self, raw_node: dict) -> Node:
        """
        Parses the provided dict into a 'Block' object.
        """
        category = raw_node[c.NODE_TYPE_STR]
        params = raw_node[c.NODE_PARAMS_STR]
        id_ = raw_node[c.NODE_ID_STR]
        shares = raw_node[c.NODE_SHARE_STR]
        prev = raw_node[c.NODE_PREV_STR]
        next_ = raw_node[c.NODE_NEXT_STR]
        if category == c.NODE_TYPE_AUGMENT:
            inflation = raw_node[c.NODE_INFLATION_STR]
        else:
            inflation = 1
        self.__node_params[id_] = params
        return Node(id_, prev, next_, shares, inflation, category)

    def _init_base_nodes(self, raw_nodes: List[dict]):
        self.base_nodes = [self._dict_to_node(raw_node) for raw_node in raw_nodes]
        self._init_int_exe_probs()

    def _init_int_exe_probs(self):
        """
        Assigns the internal execution probability to all base nodes. Fetches the internal execution probability from
        the data for each node. The internal execution probability for input nodes is calculated by the amount of data
        this node provides, divided by the sum of data all input nodes provide.
        """
        inputs_data = [self.__node_params[input_.id] for input_ in self._get_inputs(self.__base_nodes)]
        # calculate the sum of all data inputs
        n_total_input_data = sum([input_data[c.NODE_DATA_N_TOTAL_DATA] for input_data in inputs_data])
        for node in self.__base_nodes:
            node_data = self.__node_params.get(node.id)
            if node_data is None:
                continue
            match node.category:
                case c.NODE_TYPE_INPUT:
                    node.int_exe_prob = node_data[c.NODE_DATA_N_TOTAL_DATA] / n_total_input_data
                case c.NODE_TYPE_AUGMENT:
                    node.int_exe_prob = node_data[c.NODE_DATA_EXE_PROB]

    def _init_node_mapping(self):
        """
        Node mapping defines what base node ID refers to which derived node ID(s).
        """
        for node in self.__base_nodes:
            derived_node_ids = [new_id(self.__rng) for _ in node.shares]
            self.__node_mapping[node.id] = derived_node_ids

    @staticmethod
    def _get_inputs(nodes: List[Node]) -> List[Node]:
        """
        Returns all blocks of type 'Input' in the provided list.
        """
        return [node for node in nodes if node.is_input]

    @staticmethod
    def _get_outputs(nodes: List[Node]) -> List[Node]:
        """
        Returns all output blocks in the provided list.
        """
        outputs = [node for node in nodes if node.is_output]
        if not outputs:
            raise ValueError("No output blocks defined in config.")
        return outputs

    @staticmethod
    def _get_augments(nodes: List[Node]) -> List[Node]:
        """
        Returns all augment nodes of the provided list.
        """
        return [node for node in nodes if not node.is_input]





























