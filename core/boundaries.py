import math
from typing import List, Optional, Union

import numpy as np


class Boundary:
    def __init__(self, points: np.ndarray, img_dims: tuple):
        self._points = points
        self.img_dims = img_dims
        # self.validate()
        self.valid = True

    def validate(self):
        """
        Validates shape and points of boundary. Invalidates boundary if point validation fails.
        Raises:
            AssertionError if shape of boundary points does not match with (n, 2)
        """
        self._validate_shape()
        self._boundary_in_border()
        self._validate_points()

    def _validate_shape(self):
        """
        Validates the shape of the boundary points.
        Raises:
            AssertionError if shape of boundary points does not match with (n, 2)
        """
        boundary_shape = np.shape(self._points)
        assert len(boundary_shape) == boundary_shape[1] == 2, (f"Expected boundary of shape (n, 2). "
                                                               f"Received shape {boundary_shape}.")

    def _validate_points(self):
        """

        -- This function may be overwritten in a subclass --

        Validates points of boundary depending on the boundary type.
        """
        pass

    def _boundary_in_border(self):
        """
        Invalidates boundary if all values are equal.
        """
        points_x, points_y = self._points.T
        self.valid = len(np.unique(points_x)) != 1 and len(np.unique(points_y)) != 1

    @property
    def points(self) -> np.ndarray:
        return self._points

    @property
    def center(self):
        return self.__get_center()

    @property
    def area(self):
        return self.__get_area()

    @property
    def visualize(self):
        return self._points

    def __get_center(self):
        """
        Calculates the center of the boundary by the mid-point between min and max x and y.
        """
        points_x, points_y = self._points.T
        return np.array(
            [
                (np.min(points_x) + np.max(points_x)) / 2,
                (np.min(points_y) + np.max(points_y)) / 2
            ]
        )

    def __get_area(self) -> float:
        """

        -- This function may be overwritten in a subclass --

        Calculates the area which is enclosed by the boundary. Area is returned as float of n pixel².
        """

    def set(self, points: np.ndarray, validate: bool = True):
        self._points = points
        if validate:
            self.validate()

    def clip(
            self,
            x_min: Optional[int] = None,
            x_max: Optional[int] = None,
            y_min: Optional[int] = None,
            y_max: Optional[int] = None
    ) -> None:
        """
        Clips all points of boundary to new min and max x and y values.
        Invalids boundary if all points of x or y are equalized.
        Args:
            x_min (Optional - int): new min x value of boundary points
            x_max (Optional - int): new max x value of boundary points
            y_min (Optional - int): new min y value of boundary points
            y_max (Optional - int): new max y value of boundary points
        """
        points_x, points_y = self._points.T
        self.set(
            np.vstack(
                (
                    np.clip(points_x, x_min, x_max),
                    np.clip(points_y, y_min, y_max)
                )
            ).T
        )

    def shift(self, x_shift: float, y_shift: float):
        """
        Shifts boundary by adding x_shift and y_shift for each x and y in points.
        Args:
            x_shift (float): Shift of x coordinates
            y_shift (float): Shift of y coordinates
        """
        self.set(self._points + np.array([x_shift, y_shift]))

    def scale(self, x_scale: float, y_scale: float):
        """
        Scales boundary by multiplying each point by a scaling matrix.
        Args:
            x_scale (float): x or width scale factor
            y_scale (float): y or height scale factor
        """
        matrix = np.array(
            [
                [x_scale, 0],
                [0, y_scale]
            ]
        )
        self.set(np.einsum("bi, ij -> bi", self._points, matrix))

    def rotate(self, angle: float):
        """
        Rotates boundary by an angle.
        Args:
            angle (float): Angle of rotation in deg
        """
        rad_angle = np.deg2rad(-angle)
        matrix = np.array(
            [
                [np.cos(rad_angle), -np.sin(rad_angle)],
                [np.sin(rad_angle), np.cos(rad_angle)]
            ]
        )
        # ox, oy = self.center
        # adj_points_list = []
        # for point in self._points:
        #     x, y = point
        #     qx = ox + math.cos(rad_angle) * (x - ox) + math.sin(rad_angle) * (y - oy)
        #     qy = oy + -math.sin(rad_angle) * (x - ox) + math.cos(rad_angle) * (y - oy)
        #     adj_points_list.append([qx, qy])
        # points = np.array(adj_points_list)

        adj_points = self._points - self.center
        points = np.einsum("bi, ij -> bj", adj_points, matrix) + self.center
        #points = np.vstack([np.matmul(point, matrix) for point in adj_points]) + self.center

        # points = np.einsum("bi, ij -> bi", self.__points, matrix)
        self.set(points)


class BBoxBoundary(Boundary):
    """
    Boundary for bounding boxes
    """
    def __init__(self, points: np.ndarray, img_dims: tuple):
        self.__min_max_points = None
        super().__init__(points, img_dims)

    @property
    def points(self) -> np.ndarray:
        return self.__min_max_points

    @property
    def visualize(self):
        points_x, points_y = self.points.T
        return np.array(
            [
                [min(points_x), min(points_y)],
                [min(points_x), max(points_y)],
                [max(points_x), max(points_y)],
                [max(points_x), min(points_y)]
            ]
        )

    def _validate_points(self):
        """
        Forces boundary as min-max box.
        """
        points_x, points_y = self._points.T
        min_x = min(points_x)
        min_y = min(points_y)
        max_x = max(points_x)
        max_y = max(points_y)
        self.set(
            np.array(
                [
                    [min_x, min_y],
                    [max_x, min_y],
                    [max_x, max_y],
                    [min_x, max_y]
                ]
            ), validate=False
        )
        self.__min_max_points = np.array(
                [
                    [min_x, min_y],
                    [max_x, max_y]
                ]
        )

    def __get_area(self) -> float:
        """
        Calculates box area as product of box width and box height.
        """
        return np.prod(self.points[1] - self.points[0])


class KeyPBoundary(Boundary):
    """
    Boundary for keypoints.
    """
    def __init__(self, points: np.ndarray, img_dims: tuple):
        super().__init__(points, img_dims)

    def __get_area(self) -> float:
        """
        Keypoints have no area. Therefore, 0 is returned.
        """
        return 0


class PolyBoundary(Boundary):
    """
    Boundary for polygons.
    """
    def __init__(self, points: np.ndarray, img_dims: tuple):
        super().__init__(points, img_dims)
