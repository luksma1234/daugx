from ..core.errors.aug_exceptions import UnequalDatatypesError
from ..core.data_utils.data_loader import DataLoader
from typing import List, Callable
import numpy as np


"""

    Geometric:
        - Shift
        - Scale
        - Rotate
        - Crop
        - Transform
        - Resize

    Multi Image:
        - Mosaic
        - Overlay

    Pixel Level:
        - Blur
        - Brighten
        - Saturation
        - Invert color
        - Noise
        - Cutout

    Additive:
        - Insert

    Subtractive:
        - Remove

    Dataset Operators:
        - Merge
        - Split
        - Duplicate
        - Modify
"""


class Shift:
    def __init__(self):
        self.data_inflation = 1


class Mosaic:
    def __init__(self):
        # deflates data by factor 4
        self.data_inflation = 0.25


class AugmentationBase:
    def __init__(
            self,
            data_dir: str,
            data_names: str or List[str],
            data_type: str
    ):
        """
        A base class for all augmentations.
        Loads and validates data when initialized.
        Checks for consistency in arguments.


        Args:
            data_dir: Data directory, where all data to be augmented can be found.
            data_names: Names of data. Name and data directory must build up to the full data path.
            data_type: Type of data to be augmented. Valid types are: "image"
        """
        self.dir = data_dir
        self.names = data_names
        self.data_type = data_type
        self.loader = DataLoader(self.data_type)
        self.data = None

    def load_data(self, path):
        """
        Loads data as np.ndarray into self.data

        Returns:
            None
        """
        self.data = self.loader.load(path)

    def augment(self, aug_func: Callable, params: dict) -> None:
        """
        The execution function for any augmentation. Stores augmented data in self.data.

        Args:
            aug_func: Any augmentation function which can handle the defined datatype. Each augmentation function has to
                      take at least two arguments - data and data_type.
            params: A dictionary of optional parameters to execute the augmentation function

        Returns:
            None
        """
        self.data = aug_func(
            data=self.data,
            data_type=self.data_type,
            **params
        )

    def add_data(self, data: np.ndarray or List[np.ndarray], data_type: str):
        """
        Appends data to self.data.

        Args:
            data: Any data in form of a np.ndarray
            data_type: The data_type of data to add
        Returns:
            None
        Raises:
            UnequalDatatypesError: if input data_type does not match with self.datatype.
            TypeError: if type of data or self.data does not match any of np.ndarray or List[np.ndarray]
        """

        if data_type != self.data_type:
            raise UnequalDatatypesError(f"Datatype {data_type} does not match with type {self.data_type}.")

        elif self.data is None:
            self.data = data

        elif isinstance(self.data, np.ndarray):
            if isinstance(data, np.ndarray):
                self.data = [self.data, data]
            elif isinstance(data, list):
                self.data = [self.data]
                self.data.extend(data)
            else:
                raise TypeError(f"Datatype {type(data)} is not supported as augmentation data. "
                                f"Supported Types are: np.ndarray and List[np.ndarray].")

        elif isinstance(self.data, list):
            if isinstance(data, np.ndarray):
                self.data.append(data)
            elif isinstance(data, list):
                self.data.extend(data)
            else:
                raise TypeError(f"Datatype {type(data)} is not supported as augmentation data. "
                                f"Supported Types are: np.ndarray and List[np.ndarray].")

        else:
            raise TypeError(f"Datatype {type(self.data)} is not supported as augmentation data. "
                            f"Supported Types are: np.ndarray and List[np.ndarray].")

    def get_data(self):
        """
        Used to extract data from this class.

        Returns:
            data and data type as a tuple
        """
        return self.data, self.data_type


class ImageAugmentation(AugmentationBase):
    def __init__(
            self,
            data_dir: str,
            data_names: str or List[str],
    ):
        """
        A subclass for image augmentation.
        Args:
            data_dir:
            data_names:
        """
        self.data_type = "image"
        super().__init__(data_dir, data_names, self.data_type)
