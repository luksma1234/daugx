"""

"""
from .operator import Operator
from ..errors.logic_errors import AmbiguousCrownError
from ..utils.misc import choose_by_prob, new_id
from .path import Sequence
import threading
from typing import List
from copy import deepcopy


class Path:
    """
    Each path consists of a list of sequences. Operators are not included in a path. Operators are used to calculate the
    paths execution probability, or to parse special parameters to the data loading process e.g. Filtering.
    """
    def __init__(self):
        self.sequences = []
        self.roots = []
        self.root_exe_probs = None

        # Paving
        self.paved_path = []
        self.paved_data_ids = []

    def pave(self):
        """
        Paves exactly one path.
        Returns: executable, paved path
        """
        paved_path = []
        n_images = 1
        is_crown_reached = False
        step = choose_by_prob(self.roots, self.root_exe_probs)
        while not is_crown_reached:
            next_id = step.starts
            step = self._get_sequence_by_id(next_id)
            paved_sequence, inflation = step.pave()
            # Check if sequence starts with inflation
            if inflation[0] < 1:
                related_sequences = self._get_related_sequences(step.id)
                if len(related_sequences) > 1:
                    pass

    def _paving_process(self, starts: str, previous: str = None) -> None:
        """
        Preprocesses the paving of a sequence.
        Args:
            starts:

        Returns:

        """
        sequence = self._get_sequence_by_id(starts)
        related_sequences = self._get_related_sequences(starts)
        n_related_sequences = len(related_sequences)
        if n_related_sequences == 1:
            if previous is not None and related_sequences[0].id == previous:
                previous_sequence = self._get_sequence_by_id(previous)
                sequence.input_id = previous_sequence.id
                paved_sequence, inflation = sequence.pave()
                # check if inflation of one element of sequence is smaller than 1
                if any([element_inflation < 1 for element_inflation in inflation]):
                    current_input_id = sequence.input_id
                    for element, element_inflation in zip(paved_sequence, inflation):
                        if element_inflation == 1:
                            element.input_id = current_input_id
                            element.output_id = current_input_id
                            self.paved_path.append(element)
                        else:
                            n_iterations = 1 / element_inflation
                            current_paved_path = self.paved_path
                            for i in range(n_iterations):
                                additional_paved_path = deepcopy(current_paved_path)
                                additional_id = new_id()
                                self.paved_data_ids.append(additional_id)
                                for additional_element in additional_paved_path:
                                    if additional_element.input_id == additional_element.output_id:
                                        additional_element.input_id = additional_id
                                        additional_element.output_id = additional_id
                                    else:
                                        additional_element.input_id = additional_id
                                        additional_id = new_id()
                                        additional_element.output_id = additional_id
                                self.paved_path.extend(additional_paved_path)
                if sequence.starts is not None:
                    self._paving_process(starts=sequence.starts, previous=sequence.id)
                
    def _get_sequence_by_id(self, sequence_id: str) -> Sequence:
        """
        Returns the sequence with the given id.
        Args:
            sequence_id: id of the sequence to be returned
        Returns: Sequence
        """
        for sequence in self.sequences:
            if sequence.id == sequence_id:
                return sequence
        else:
            raise ValueError(f"Sequence with id '{sequence_id}' not found in this path.")

    def _get_related_sequences(self, sequence_id: str) -> List[Sequence]:
        """
        Returns all sequences that are related to the given sequence.
        Args:
            sequence_id: id of the sequence to be returned
        Returns: List[Sequence]
        """
        related_sequences = []
        for sequence in self.sequences:
            if sequence_id in sequence.starts:
                related_sequences.append(sequence)
        return related_sequences