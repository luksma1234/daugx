from typing import Optional

import numpy as np


class ImageBorder:
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
                [self.__x_min, self.__y_min],
                [self.__x_max, self.__y_max]
            ], np.int32
        )

    @property
    def x_min(self):
        return self.__x_min

    @property
    def x_max(self):
        return self.__x_max

    @property
    def y_min(self):
        return self.__y_min

    @property
    def y_max(self):
        return self.__y_max

    @property
    def area(self):
        return self.width * self.height

    def set(
            self,
            x_min: Optional[int] = None,
            y_min: Optional[int] = None,
            x_max: Optional[int] = None,
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
        # Assertion is invalid for NoneType
        # assert x_max > x_min >= 0 and y_max > y_min >= 0, "Invalid image border. Border is 0 or negative."
        if x_min is not None:
            self.__x_min = x_min
        if y_min is not None:
            self.__y_min = y_min
        if x_max is not None:
            self.__x_max = x_max
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
