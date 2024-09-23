from typing import List
import json

from daugx.core.agent.sequence import Sequence, Sequences


class Workflow:

    def __init__(self, workflow_dict: dict):
        self.workflow_dict = workflow_dict
        self.sequences: List[Sequence] = []
    #
    # @property
    # def sequences(self):
    #     return self.sequences

    def build(self, raw_block_list: List[dict]):
        sequences = Sequences(raw_block_list)
        self.sequences = sequences.build()
        print(self.sequences)

    def fetch(self):
        """
        Gets one task to execute for the executor.
        """
        pass

if __name__ == "__main__":
    wf = Workflow({})
    with open("BlockTesting.json") as f:
        raw = json.load(f)
    wf.build(raw["blocks"])



