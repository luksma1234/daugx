from typing import List
from copy import deepcopy

from daugx.core.agent.block import Block, Blocks
from daugx.utils.node_utils import is_inflationary
from daugx.utils.misc import new_id


class Sequence:
    def __init__(self):
        """
        A Sequence is a collection of blocks, which are connected to each other either by their prev or next attribute.
        Sequences end when there is no next block, or the next block is inflationary. Dividing blocks create multiple
        sequences, one for each choice.
        """
        self.__prev_sequences: List[str] = []
        self.__blocks: List[Block] = []
        self.__exe_prob: float = 1
        self.__id = new_id()
        self.__next: List[str] = []
        self.__prev: List[str] = []
        self.__next_blocks: List[str] = []
        self.__total_exe_prob: float = 0

    def __len__(self):
        return len(self.__blocks)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    @property
    def id(self):
        return self.__id

    @property
    def exe_prob(self):
        return self.__exe_prob

    @property
    def blocks(self):
        return self.__blocks

    @property
    def next_sequences(self):
        return self.__next

    @property
    def prev_sequences(self):
        return self.__prev

    @property
    def next_blocks(self):
        return self.end_block.next

    @property
    def start_block(self):
        if len(self.__blocks) > 0:
            return self.__blocks[0]
        return None

    @property
    def end_block(self):
        if len(self.__blocks) > 0:
            return self.__blocks[-1]
        return None

    @property
    def is_finished(self):
        if self.end_block is None:
            return False
        return self.end_block.is_output

    @property
    def is_input(self):
        return self.start_block.is_input

    @property
    def total_exe_prob(self):
        return self.__total_exe_prob

    def pop_end(self):
        self.__blocks = self.__blocks[:-1]

    def add_to_total_exe_prob(self, exe_prob: float):
        """
        The total exe prob is the absolute probability to reach and execute this sequence
        """
        self.__total_exe_prob += (exe_prob * self.__exe_prob)

    def mult_exe_prob(self, mult: float):
        """
        Multiplies the execution probability with any multiplier.
        """
        self.__exe_prob *= mult

    def add_block(self, block: Block):
        assert block.is_set
        if self._is_suitable_block(block):
            self.__blocks.append(block)

    def add_next_sequence(self, next_sequence: str):
        self.__next.append(next_sequence)

    def add_prev_sequence(self, prev_sequence: str):
        self.__next.append(prev_sequence)

    def _is_suitable_block(self, block: Block):
        """
        Verifies if the block matches the next Block of the last Block added.
        """

        if self.end_block is None or (self.end_block.id in block.prev and block.id in self.next_blocks):
            return True
        return False


class Sequences:

    def __init__(self, raw_block_list: List[dict]):
        self.__raw_block_list = raw_block_list

        self.__blocks = Blocks()
        self.__blocks.build(self.__raw_block_list)

        self.__sequences: List[Sequence] = []
        self.__output_sequences = []

    @property
    def __input_sequences(self):
        return [sequence for sequence in self.__sequences if sequence.is_input]

    def build(self) -> List[Sequence]:
        ipt_blocks = self.__blocks.input_blocks
        # Build sequences from inpout blocks upwards
        for block in ipt_blocks:
            self._build_sequences(block)
        # Connect sequences from input sequence upwards
        for sequence in self.__input_sequences:
            self._connect_sequences(sequence)
        # Calculate sequence total exe prob from output sequence downwards
        self._propagate_exe_prob()
        return self.__sequences

    def _build_sequences(self, block: Block, seq: Sequence = None) -> None:
        """
        Something is off here... Make sure sequence stops on inflationary nodes. Make sure if inflationary nodes have
        execution probabilities, you must build one with that node and once without that node.
        """
        if seq is None:
            seq = Sequence()
        if block.inflation < 1:
            if block.exe_prob < 1:
                # The scenario when mosaic is not executed must be taken into account
                for next_block_id in block.next:
                    # set prev of MIT to prev of next to build scenario without MIT
                    next_block = deepcopy(self.__blocks[next_block_id])
                    next_block.set_prev(block.prev)
                    # set next of prev of MIT to next of MIT
                    seq_end_block = deepcopy(seq.end_block)
                    seq_end_block.set_next(block.next)
                    seq_end_block.set_shares(block.shares)
                    seq.pop_end()
                    for share_index, _ in enumerate(block.shares):
                        seq_end_block.set(share_index)
                        seq.add_block(seq_end_block)
                        # Setup new sequence for current share
                        self._build_sequences(next_block, seq)
            self.__sequences.append(deepcopy(seq))
            seq = Sequence()
        if block.variations > 1:
            # Enumerate over block variations, set block with index and build upon this block.
            for index, (next_block_id, share) in enumerate(zip(block.next, block.shares)):
                next_block = deepcopy(self.__blocks[next_block_id])
                new_seq = deepcopy(seq)
                new_seq.mult_exe_prob(share)
                block.set(index)
                new_seq.add_block(block)
                self._build_sequences(next_block, new_seq)
            return
        # set block with share index 1 due to no variations
        block.set(0)
        seq.add_block(block)
        if seq.is_finished:
            self.__sequences.append(deepcopy(seq))
        else:
            if not block.is_output:
                next_block = deepcopy(self.__blocks[block.next[0]])
                self._build_sequences(next_block, seq)
        return

    def _connect_sequences(self, sequence: Sequence):
        # Get next sequences by matching this sequence end blocks next block with a sequence starting block.
        next_sequences = []
        next_blocks = sequence.end_block.next
        for next_block in next_blocks:
            next_sequence = self._get_sequence_by_start_block(next_block)
            if next_sequence is not None:
                next_sequences.append(next_sequence)
        # Add next sequence ids to this sequence next attribute.
        # Adds this sequence as previous sequence of each next sequence.
        for next_sequence in next_sequences:
            sequence.add_next_sequence(next_sequence.id)
            next_sequence.add_prev_sequence(sequence.id)
            self._connect_sequences(next_sequence)

    def _propagate_exe_prob(self):
        # starts on output sequences
        for sequence in self.__output_sequences:
            self._calc_total_exe_prob(sequence)

    def _calc_total_exe_prob(self, sequence: Sequence):
        if sequence.is_input:
            sequence.mult_exe_prob(sequence.start_block.exe_prob)
        else:
            prev_seqs = [self._get_sequence_by_id(id_) for id_ in sequence.prev_sequences]
            for prev_seq in prev_seqs:
                if prev_seq.total_exe_prob > 0:
                    sequence.add_to_total_exe_prob(prev_seq.exe_prob)
                else:
                    self._calc_total_exe_prob(prev_seq)
                    sequence.add_to_total_exe_prob(prev_seq.exe_prob)

    def _get_sequence_by_start_block(self, start_block_id: str):
        for sequence in self.__sequences:
            if sequence.start_block.id == start_block_id:
                return sequence
        return None

    def _get_sequence_by_id(self, id_: str):
        for sequence in self.__sequences:
            if sequence.id == id_:
                return sequence
        return None

    def add_to_sequences(self, sequence: Sequence):
        if self.is_unique(sequence):
            self.__sequences.append(sequence)



    def is_unique(self, sequence: Sequence):
        """
        Checks if an identical sequence already exists.
        """
        for seq in self.__sequences:
            # compare all blocks for id and next
