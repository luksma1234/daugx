from ..data_utils import DataLoader
from typing import List
from process import Step, PavedProcess


class Executor:
    def __init__(self, data_type: str):
        """
        Universal execution class.
        Args:
            data_type: Type of data used for execution, currently accepted data types: [image]
        """
        self.data_loader = DataLoader(data_type=data_type)
        self.temp_data = []

    def execute(self, paved_process: PavedProcess):
        """
        Executes all steps of the paved process sequentially.
        Args:
            paved_process:

        Returns:
            np.ndarray - result of paved process
        """

    def _get_data_by_id(self):
        pass

