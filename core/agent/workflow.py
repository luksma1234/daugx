
from copy import deepcopy

from daugx.core import constants as c
from daugx.utils.node_utils import is_output, is_dividing, is_inflationary, config_to_node
from daugx.core.agent.path import Path
from daugx.core.agent.node import *
from daugx.core.agent.sequence import Sequence

# TODO: Change execution to application. exe_prob -> app_prob


class Workflow:

    def __init__(self, workflow_dict: dict):
        self.workflow_dict = workflow_dict

        self.__nodes: List[Node] = []
        self.__sequences: List[Sequence] = []
        self.__paths: List[Path] = []
        self._build()

    @property
    def paths(self):
        return self.__paths

    @property
    def fetch(self):
        return self._get_path()

    def _build(self) -> None:
        """
        Builds all possible paths of the workflow. First all nodes are initialized, then all sequences are created from
        the Nodes. Afterward, the Sequences are assembled into all Paths possible.
        """
        self._init_nodes()
        self._init_seqs()
        self._init_paths()

    def _init_nodes(self) -> None:
        """
        Initializes all Nodes listed in the workflow. Nodes are initialized by their types. Nodes can be of type
        'input', 'output', 'merge', 'split', 'filter' and 'augment'.
        Sets the next argument for all initialized Nodes. Next node can either be one Node or a list of Nodes.
        """
        # init nodes
        node_configs = self.workflow_dict[c.NODES_STR]
        self.__nodes = [config_to_node(node_config) for node_config in node_configs]
        # set next_nodes
        for node_config, node in zip(node_configs, self.__nodes):
            if isinstance(node_config[c.NODE_NEXT_STR], list):
                next_nodes = []
                for ident in node_config[c.NODE_NEXT_STR]:
                    next_nodes.append(self._get_node_by_id(ident))
                node.next = next_nodes
            else:
                next_node = self._get_node_by_id(node_config[c.NODE_NEXT_STR])
                node.next = next_node

    def _get_node_by_id(self, id_: str) -> Node or None:
        """
        Gets a node by an ID.

        Args:
            id_ (str): ID of the Node
        Returns:
            (Node): if a matching node was found.
            (None): if no node with matching id was found.
        """
        for node in self.__nodes:
            if node.id == id_:
                return node

    def _get_ipt_nodes(self) -> List[InputNode]:
        """
        Returns all nodes in self.node of type InputNode.

        Returns:
            (List[InputNode]): List of input Nodes
        """
        return [node for node in self.__nodes if isinstance(node, InputNode)]

    def _init_seqs(self) -> None:
        """
        Initializes all Sequences which can be created from Nodes. Sequences are essentially lists of succeeding Nodes.
        Sequences take up Nodes until the Node to be incorporated has one of the following 'sequence end properties':
        - is an output node (including)
        - is a dividing node (excluding)
        - is an inflationary node (excluding)
        A new Sequence will start after (including) or with that Node (excluding).

        Sequence initialization starts from the input Node(s). After initialization, the execution probabilities of all
        input sequences is calculated. The execution probability of input sequences is calculated by dividing the
        amount of images the input node provides, by the sum of images of all input nodes.
        """
        ipt_nodes = self._get_ipt_nodes()
        for node in ipt_nodes:
            self._build_seqs(node)
        self._set_ipt_seqs_exe_probs()

    def _build_seqs(self, node: Node, exe_prob=None) -> None:
        """
        Builds a sequence starting from a given node. Takes the next Node from the given Node and incorporates it into
        the sequence, if none of the 'sequence end properties' match.
        If an including 'sequence end property' matches, the Node is added to the sequence, and the sequence is
        finished. A new sequence is then created recursively from the next Node.
        If an excluding 'sequence end property' matches, the sequence is finished without adding the Node. A new
        sequence created recursively from this Node.

        Args:
            node (Node): Any node as entry point for sequence
            exe_prob: The execution probability for the given node (only if the Node has one)
        """
        seq = Sequence()
        if exe_prob is not None:
            seq.exe_prob = exe_prob
        while not is_output(node):
            seq.add_node(node)
            if is_dividing(node):
                node = node.next[node.split_idx]
                continue
            elif is_dividing(node.next):
                if self._finish_seq(seq, node):
                    self._build_div_seqs(node.next)
                return
            elif node.next is not None and is_inflationary(node.next):
                self._build_infl_seqs(seq, node)
                return
            node = node.next
        seq.add_node(node)
        self.add_seq(seq)

    def _finish_seq(self, seq: Sequence, node: Node) -> bool:
        """
        Finishes a sequence by setting the next node and adding the sequence to self.sequences.
        :param seq: A sequence to be finished
        :param node: The last Node of this sequence
        :return: True - if sequence could be added to self.sequences
                 False - if sequence could not be added to self.sequences
        """
        seq.set_next_node(node.next)
        if self.add_seq(seq):
            return True
        return False

    def _build_div_seqs(self, node: DividingNode) -> None:
        """
        Method for building dividing sequences. Dividing sequences start with dividing nodes. Dividing nodes are nodes
        of type split or filter.
        :param node: A dividing node as entry point
        :return: None
        """
        for index, share in enumerate(node.split_shares):
            split_node = deepcopy(node)
            split_node.set_split_idx(index)
            self._build_seqs(split_node, share)
            
    def _build_infl_seqs(self, seq: Sequence, node: Node) -> None:
        """
        Builds inflationary sequences. Finishes current sequence and build upon next and next.next node.
        :param seq: A sequence which ends before an inflationary node.
        :param node: The last node of this sequence
        :return: None
        """
        if self._finish_seq(seq, node):
            self._build_seqs(node.next, node.next.inflation)
        seq = deepcopy(seq)
        if self._finish_seq(seq, node.next):
            self._build_seqs(node.next.next, 1 - node.next.inflation)

    def add_seq(self, seq: Sequence) -> bool:
        """
        Adds sequence to self.sequences if the sequence is unique.
        :param seq: Any sequence
        :return: True if sequence was added
                 False if sequence is already part of self.sequences
        """
        if self._is_unique(seq):
            self.__sequences.append(seq)
            return True
        return False

    def _get_seqs(self, start_node_id: str or int = None, end_node_id: str or int = None) -> List[Sequence]:
        """
        Gets all sequences which start/end node has the given id.
        :param start_node_id: any start node id
        :param end_node_id: any start node is
        :return:
        """
        sequences = []
        for sequence in self.__sequences:
            if start_node_id is not None and sequence.get_start_node().id == start_node_id:
                sequences.append(sequence)
            elif end_node_id is not None and sequence.get_end_node().id == end_node_id:
                sequences.append(sequence)
        return sequences

    def _set_ipt_seqs_exe_probs(self) -> None:
        """
        Sets execution probability for all input sequences.
        :return: None
        """
        ipt_nodes = self._get_ipt_nodes()
        data_sum = sum([node.n_data for node in ipt_nodes])
        for node in ipt_nodes:
            seq = self._get_seqs(start_node_id=node.id)[0]
            seq.exe_prob = node.n_data / data_sum

    def _is_unique(self, new_seq: Sequence) -> bool:
        """
        Checks if a sequence is already existing in self.sequences. Compares all nodes and next nodes.
        :param new_seq: Any sequence
        :return: True if sequence is unique
                 False if sequence is already part of self.sequences
        """
        for seq in self.__sequences:
            # checks if all sequence ids match
            if len(seq) == len(new_seq) and all([n.id == nn.id for n, nn in zip(seq.nodes, new_seq.nodes)]):
                if seq.next_node is None and new_seq.next_node is None:
                    return False
                if (seq.next_node is not None and
                        new_seq.next_node is not None and
                        seq.next_node.id == new_seq.next_node.id):
                    return False
        else:
            return True

    def _init_paths(self) -> None:
        """
        Initializes all paths of the blueprint.
        :return: None
        """
        self._build_ipt_paths()
        while len(self._get_incomplete_paths()) > 0:
            # paths with same next_nodes have to be grouped.
            self._group_incomplete_paths()
            self._build_incomplete_paths()

    def _get_incomplete_paths(self) -> List[Path]:
        return [path for path in self.__paths if not path.is_complete]

    def _build_ipt_paths(self) -> None:
        """
        Builds linear paths from all existing input sequences. Adds paths to self.paths.
        :return: None
        """
        input_sequences = [sequence for sequence in self.__sequences if isinstance(sequence.get_start_node(), InputNode)]
        for input_sequence in input_sequences:
            input_path = Path()
            input_path.add_sequence(input_sequence, input_sequence.next_node)
            self._build_linear_path(input_path)

    def _build_linear_path(self, path: Path) -> None:
        """
        Builds a path from an entry path. A path stops on:
        - next_node is None -> Path is completed
        - next_node is inflationary -> Paths need to be grouped before continuing
        :param path: Any path as entry point
        :return: None
        """
        next_node = path.next_node
        if next_node is not None and not is_inflationary(next_node):
            next_seqs = self._get_seqs(start_node_id=next_node.id)
            for seq in next_seqs:
                next_path = deepcopy(path)
                next_path.add_sequence(seq, seq.next_node)
                # maybe next_path.is_complete?
                if not next_path.is_complete:
                    self._build_linear_path(next_path)
                else:
                    self.__paths.append(next_path)
        else:
            self.__paths.append(path)

    def _group_incomplete_paths(self) -> None:
        """
        Groups incomplete paths with identical next_node. These paths are then combined as variants of one path.
        :return: None
        """
        paths = self._get_incomplete_paths()
        groups = {}
        for path in paths:
            if path.next_node.id not in groups.keys():
                groups[path.next_node.id] = [path]
            else:
                groups[path.next_node.id].append(path)
        self._combine_groups(groups)
        for path in paths:
            self.__paths.remove(path)

    def _combine_groups(self, groups: dict) -> None:
        """
        Combines all paths of the given groups by merging each group into one path with multiple variations.
        Adds combined paths to self.paths.
        :param groups: Dictionary of path groups. Key represents the next_node id. Value a list of paths with
                       this next_node.
        :return: None
        """
        for group in groups.values():
            next_node = group[0].next_node
            new_path = Path()
            new_path.add_sequence(group, next_node)
            self.__paths.append(new_path)

    def _build_incomplete_paths(self) -> None:
        """
        Builds up from paths. To prevent immediate return from linear build-> Adds next sequence first before
        passing to _build_linear_path method.
        :return: None
        """
        for path in self._get_incomplete_paths():
            next_seq = self._get_seqs(start_node_id=path.next_node.id)[0]
            path.add_sequence(next_seq, next_seq.next_node)
            if not path.is_complete and path.next_node.inflation == 1:
                self._build_linear_path(path)

    def _get_path(self) -> Path:
        """
        Gets a path from a list of paths. Paths are sorted by their execution probability. Chooses the method for
        getting paths by the amount of paths available.
        Returns:
            (Path): Chosen path
        """
        prob = get_random()
        if len(self.__paths) > 100:
            return self._get_path_long(prob)
        else:
            return self._get_path_short(prob)

    def _get_path_long(self, prob: float):
        """
        Gets a path from a list of paths. Works faster for a list of >100 paths.
        """
        start_index = int(prob * len(self.__paths))
        if start_index == 0:
            return self._get_path_short(prob)
        elif self.__paths[start_index - 1].exe_prob_sum < prob < self.__paths[start_index].exe_prob_sum:
            # direct hit
            return self.__paths[start_index]
        elif self.__paths[start_index].exe_prob_sum < prob:
            # iterates list from guessed index upwards
            return self._choose_path_by_prob(start_index, False, prob)
        elif self.__paths[start_index].exe_prob_sum > prob:
            # iterates list from guessed index downwards
            return self._choose_path_by_prob(start_index, True, prob)

    def _get_path_short(self, prob: float):
        """
        Gets a path from a list of paths. Works faster for a list of <100 paths.
        """
        return self._choose_path_by_prob(0, False, prob)

    def _choose_path_by_prob(self, start_index: int, inverted: bool, prob: float) -> Path:
        """
        Chooses one path by a given probability

        Returns:
            (Path): Chosen path
        """
        cur_prob = self.__paths[start_index].exe_prob_sum
        if inverted:
            paths = self.__paths[:start_index]
            reversed(paths)
            for path in paths:
                if prob > cur_prob - path.exe_prob:
                    return path
                else:
                    cur_prob -= path.exe_prob
        else:
            paths = self.__paths[start_index + 1:]
            for path in paths:
                if prob < cur_prob + path.exe_prob:
                    return path
                else:
                    cur_prob += path.exe_prob
