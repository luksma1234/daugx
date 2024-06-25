from typing import List, Optional, Union

import numpy as np

from daugx.utils import new_id

from .boundaries import Boundary, BBoxBoundary, KeyPBoundary, PolyBoundary
from.borders import ImageBorder

BOUNDARY_NAME = "Boundary"
BOUNDARY_TYPE_OBJS = [BBoxBoundary, KeyPBoundary, PolyBoundary]


class Label:
    def __init__(self, label_id: Optional[int] = None, name: Optional[str] = None) -> None:
        """
        Defines a Label for any kind of data.
        Args:
            label_id (int): ID of label
            name (Optional - str): Name of label
        """
        assert label_id is not None or name is not None, "Unable to create label without name and id."
        self.id = label_id
        self.name = name

# TODO: Implement additional Information to annotation. Additional information does not change during augmentations.


class Annotation:
    def __init__(
            self,
            boundary_points: np.ndarray,
            image_border: ImageBorder,
            boundary_type: str,
            label_id: Optional[int] = None,
            label_name: Optional[str] = None
    ) -> None:
        """
        A Generic implementation of annotations.
        Args:
            label_name (str): name of annotation label
            label_id (int): id of annotation label
            boundary_points (np.ndarray): points of boundary as numpy array. Points are always in shape (n, 2).
            boundary_type (str): type of boundary - accepted types are: BBox, KeyP, Poly or an empty string
        """
        self.__border: ImageBorder = image_border
        self.__boundary: Boundary | None = None
        self.set_boundary(boundary_points, boundary_type, self.__border)
        self.__label: Label | None = None
        self.set_label(label_id, label_name)

        self.id = new_id()
        self.valid = True
        self.verify()

    @property
    def boundary(self):
        return self.__boundary

    @property
    def label(self):
        return self.__label

    @property
    def center(self):
        return self.__boundary.center

    @property
    def area(self):
        return self._get_area()

    def set_label(self, label_id: Optional[int] = None, label_name: Optional[str] = None) -> None:
        if self.__label is None:
            self.__label = Label(label_id, label_name)
            return
        if label_id is not None:
            self.__label.id = label_id
        if label_name is not None:
            self.__label.name = label_name

    def set_boundary(self, points: np.ndarray, boundary_type: str, img_border: ImageBorder) -> None:
        if self.__boundary is None:
            for obj in BOUNDARY_TYPE_OBJS:
                if obj.__name__ == boundary_type + BOUNDARY_NAME:
                    self.__boundary = obj(points, img_border)
                    break
            else:
                raise ValueError(f"Boundary type '{boundary_type}' could not be found.")
        else:
            self.__boundary.set(points)

    def verify(self) -> None:
        """
        Sets annotation valid flag based on boundary validity.
        """
        if not self.valid:
            return
        self.valid = self.__boundary.valid

    def _get_area(self) -> float:
        """
        Gets the area which is enclosed by the boundary.
        """
        return self.__boundary.area

    def clip(self):
        """
        Clips boundary to border
        """
        self.__boundary.clip()


class Annotations:

    def __init__(self, image_width: int, image_height: int, boundary_type: str) -> None:
        """
        Docstring missing...
        """
        self.annots: List[Annotation] = []
        self.width = image_width
        self.height = image_height
        self.boundary_type = boundary_type
        self.border = ImageBorder(self.width, self.height)

    def __getitem__(self, index: int) -> Annotation:
        return self.annots[index]

    def clean(self):
        """
        Cleans list of annotations from invalid annots.
        Setting annots as invalid can also be an option to filter annots.
        """
        self.annots = [annot for annot in self.annots if not annot.valid]

    def set_border(
            self,
            x_min: Optional[int] = None,
            y_min: Optional[int] = None,
            x_max: Optional[int] = None,
            y_max: Optional[int] = None
    ) -> None:
        """
        Sets a new boundary for annotations. Can be used to crop/scale/widen image borders.
        Reinitializes width and height.
        Args:
            x_min (Optional - int): new min x value of border
            y_min (Optional - int): new min y value of border
            x_max (Optional - int): new max x value of border
            y_max (Optional - int): new max y value of border
        """
        self.border.set(x_min, y_min, x_max, y_max)
        if x_min is not None:
            self.width -= x_min
        if x_max is not None:
            self.width -= self.width - x_max
        if y_min is not None:
            self.height -= y_min
        if y_max is not None:
            self.height -= self.height - y_max

    def scale_border(self, x_scale: float = 1, y_scale: float = 1):
        """
        Scales the current border by an x and y factor
        """
        self.set_border(
            0,
            0,
            int(x_scale * self.border.width),
            int(y_scale * self.border.height)
        )

    def rebase_border(self):
        self.border.rebase()

    def add(self, label_id: Optional[int], boundary_points: np.ndarray, label_name: Optional[str] = None) -> None:
        """
        Adds a new annotation.
        Args:
            label_id (int): ID of annotation label
            label_name (Optional[str]): name of annotation label
            boundary_points (np.ndarray): numpy array of boundary points. Must be of shape (n, 2).
        """
        self.annots.append(Annotation(
            boundary_points,
            self.border,
            self.boundary_type,
            label_id,
            label_name
        ))

    def filter(self, drop_labels: List[Union[str, int]]):
        """
        Filters annotations by label. Flags annotation as invalid if it belongs to a drop label.
        Args:
            drop_labels (List[Union[str, int]]): List of label names or ids to be dropped. List can be mixed.
        """
        for annotation in self.annots:
            if annotation.label.id or annotation.label.name in drop_labels:
                annotation.valid = False
        self.clean()
    
    def shift(self, x_shift: Optional[float] = 0, y_shift: Optional[float] = 0):
        """
        Shifts boundaries of all annotations. If shift is None - no shift on that axis is performed.
        Args:
            x_shift (float): absolute shift on the x-axis in pixels.
            y_shift (float): absolute shift on the y-axis in pixels.
        """
        for annot in self.annots:
            annot.boundary.shift(x_shift, y_shift)
            annot.clip()

    def scale(self, x_scale: Optional[float] = 1, y_scale: Optional[float] = 1):
        """
        Scales borders and boundaries of all annotations. If scale is None - no scale on that axis is performed.
        Args:
            x_scale (float): scale factor on the x-axis
            y_scale (float): scale factor on the y-axis
        """
        for annot in self.annots:
            annot.boundary.scale(x_scale, y_scale)
            annot.clip()

    def rotate(self, angle: float):
        """
        Rotates boundaries of all annotations.
        Args:
            angle (float): Rotation angle in deg
        """
        for annot in self.annots:
            annot.boundary.rotate(angle)
            annot.clip()

    def crop(self, x_min: int, y_min: int, x_max: int, y_max: int):
        """
        Crops all annotations into the specified boundary.
        Args:
            x_min (float): min value for x in percentage
            y_min (float): min value for y in percentage
            x_max (float): max value for x in percentage
            y_max (float): max value for y in percentage
        """
        self.set_border(x_min, y_min, x_max, y_max)
        for annot in self.annots:
            annot.clip()
            annot.boundary.shift(-x_min, -y_min)
        self.rebase_border()
