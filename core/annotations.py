import inspect
from typing import List, Optional, Union

import numpy as np

from ..utils import new_id
import boundaries
from .boundaries import Boundary

BOUNDARY_NAME = "Boundary"


class Label:
    def __init__(self, label_id: int, name: Optional[str] = None) -> None:
        """
        Defines a Label for any kind of data.
        Args:
            label_id (int): ID of label
            name (Optional - str): Name of label
        """
        self.id = label_id
        self.name = name


class Border:
    def __init__(self, width: int, height: int) -> None:
        """
        Defines a border of an image.
        Args:
            width (int): width of image in pixels
            height (int): height of image in pixels
        """
        self.__width = width
        self.__height = height

        self.__x_min = 0
        self.__x_max = self.__width
        self.__y_min = 0
        self.__y_max = self.__height

    @property
    def width(self):
        return self.__width

    @property
    def height(self):
        return self.__height

    @property
    def corners(self):
        return np.array(
            [
                [self.__x_min, self.__x_max],
                [self.__y_min, self.__y_max]
            ], np.int32
        )

    def set(
            self,
            x_min: Optional[int] = None,
            x_max: Optional[int] = None,
            y_min: Optional[int] = None,
            y_max: Optional[int] = None
    ) -> None:
        """
        Sets corner points of border. This adapts the border to crops or scaling of the image. Input values cannot be
        negative. Corner points are used for further calculation until the border gets rebased. Values which are not
        provided will stay as is.
        Args:
            x_min (Optional - int): new min x value of border
            x_max (Optional - int): new max x value of border
            y_min (Optional - int): new min y value of border
            y_max (Optional - int): new max y value of border

        """
        assert x_max > x_min > 0 and y_min > y_max > 0, "Invalid image border. Border is 0 or negative."
        if x_min is not None:
            self.__x_min = x_min
        if x_max is not None:
            self.__x_max = x_max
        if y_min is not None:
            self.__y_min = y_min
        if y_max is not None:
            self.__y_max = y_max

    def reset(self) -> None:
        """
        Resets the current border Corner points.
        """
        self.__x_min = 0
        self.__x_max = self.__width
        self.__y_min = 0
        self.__y_max = self.__height

    def rebase(self) -> None:
        """
        Rebase the border by calculating the new width and height from the current corner points. Resets the border
        subsequently.
        """
        self.__width = self.__x_max - self.__x_min
        self.__height = self.__y_max - self.__y_min
        self.reset()


class Annotation:
    def __init__(
            self,
            label_id: int,
            boundary_points: np.ndarray,
            image_width: int,
            image_height: int,
            boundary_type: str,
            label_name: Optional[str] = None
    ) -> None:
        """
        A Generic implementation of annotations.
        Args:
            label_name (str): name of annotation label
            label_id (int): id of annotation label
            boundary_points (np.ndarray): points of boundary as numpy array. Points are always in shape (n, 2).
            image_width (int): image width in pixel
            image_height (int): image height in pixel
            boundary_type (str): type of boundary - accepted types are: BBox, KeyP, Poly or an empty string
        """
        self.__boundary: Boundary | None = None
        self.set_boundary(boundary_points, boundary_type)
        self.__label: Label | None = None
        self.set_label(label_id, label_name)
        self.__border: Border | None = None
        self.set_border(0, image_width, 0, image_height)

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
    def border(self):
        return self.__border

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

    def set_boundary(self, points: np.ndarray, boundary_type: Optional[str] = None) -> None:
        if self.__boundary is None:
            if boundary_type is None:
                raise ValueError("Expected boundary type when initializing new boundary. Found None.")
            else:
                for name, obj in inspect.getmembers(boundaries):
                    if name == boundary_type + BOUNDARY_NAME:
                        self.__boundary = obj(points)
                        break
                else:
                    raise ValueError(f"Boundary type '{boundary_type}' could not be found.")
        else:
            self.__boundary.set(points)

    def set_border(
            self,
            x_min: Optional[int] = None,
            x_max: Optional[int] = None,
            y_min: Optional[int] = None,
            y_max: Optional[int] = None
    ) -> None:
        """
        Sets a new border within the coordinate system of the old border. Clips boundary to new border.
        Args:
            x_min (Optional - int): new min x value of border
            x_max (Optional - int): new max x value of border
            y_min (Optional - int): new min y value of border
            y_max (Optional - int): new max y value of border
        """
        self.__border.set(x_min, x_max, y_min, y_max)
        self.__border.rebase()
        # adapt boundary
        self.__boundary.shift(-x_min, -y_min)
        self.__boundary.clip(0, self.__border.width, 0, self.__border.height)

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


class Annotations:
    """
    How does mosaic work?
    -> set border to new size (double) for all 4 Annotations.
    -> affine transform annots with translations
    -> add all annots of all Annotations to first annotation
    """
    def __init__(self, image_width: int, image_height: int) -> None:
        self.annots: List[Annotation] = []
        self.width = image_width
        self.height = image_height

    def __getitem__(self, index: int) -> Annotation:
        return self.annots[index]

    def clean(self):
        """
        Cleans list of annotations from invalid annots.
        Setting annots as invalid can also be an option to filter annots.
        """
        self.annots = [annot for annot in self.annots if not annot.valid]

    def border(
            self,
            x_min: Optional[int] = None,
            x_max: Optional[int] = None,
            y_min: Optional[int] = None,
            y_max: Optional[int] = None
    ) -> None:
        """
        Sets a new boundary for annotations. Can be used to crop/scale/widen image borders.
        Reinitializes width and height.
        Args:
            x_min (Optional - int): new min x value of border
            x_max (Optional - int): new max x value of border
            y_min (Optional - int): new min y value of border
            y_max (Optional - int): new max y value of border
        """
        for annot in self.annots:
            annot.set_border(x_min, x_max, y_min, y_max)
        if x_min is not None:
            self.width -= x_min
        if x_max is not None:
            self.width -= self.width - x_max
        if y_min is not None:
            self.height -= y_min
        if y_max is not None:
            self.height -= self.height - y_max

    def add(self, label_id: int, boundary_points: np.ndarray, boundary_type: str, label_name: Optional[str] = None) -> None:
        """
        Adds a new annotation.
        Args:
            label_id (int): ID of annotation label
            label_name (Optional[str]): name of annotation label
            boundary_points (np.ndarray): numpy array of boundary points. Must be of shape (n, 2).
            boundary_type (str): type of boundary for accepted types refer to Annotation class
        """
        self.annots.append(Annotation(
            label_id,
            boundary_points,
            self.width,
            self.height,
            boundary_type,
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
    
    def shift(self, x_shift: float, y_shift: float):
        """
        Shifts boundaries of all annotations.
        Args:
            x_shift (float): absolute shift on the x-axis in pixels
            y_shift (float): absolute shift on the y-axis in pixels
        """
        for annot in self.annots:
            annot.boundary.shift(x_shift, y_shift)

    def scale(self, x_scale: float, y_scale: float):
        """
        Scales borders and boundaries of all annotations.
        Args:
            x_scale (float): scale factor on the x-axis
            y_scale (float): scale factor on the y-axis
        """
        for annot in self.annots:
            annot.boundary.scale(x_scale, y_scale)
            annot.set_border(
                0,
                int(x_scale * annot.border.width),
                0,
                int(y_scale * annot.border.height)
            )

    def rotate(self, angle: float):
        """
        Rotates boundaries of all annotations.
        Args:
            angle (float): Rotation angle in deg
        """
        for annot in self.annots:
            annot.boundary.rotate(angle)

