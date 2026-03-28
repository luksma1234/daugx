"""Bounding box component."""
import numpy as np

from daugx.core.data.component import Component, ComponentState


class BoundingBox(Component):
    """Axis-aligned bounding box.  Always materialized.

    Stores coordinates as a (2, 2) numpy array:
    ``[[x_min, y_min], [x_max, y_max]]``.

    Args:
        points: Array of shape (2, 2).

    Raises:
        ValueError: If *points* is not shape (2, 2).
    """

    def __init__(self, points: np.ndarray) -> None:
        pts = np.asarray(points, dtype=float)
        if pts.shape != (2, 2):
            raise ValueError(
                f"Expected shape (2, 2), got {pts.shape}"
            )
        self._points = pts

    @property
    def points(self) -> np.ndarray:
        return self._points

    @property
    def x_min(self) -> float:
        return float(self._points[0, 0])

    @property
    def y_min(self) -> float:
        return float(self._points[0, 1])

    @property
    def x_max(self) -> float:
        return float(self._points[1, 0])

    @property
    def y_max(self) -> float:
        return float(self._points[1, 1])

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> np.ndarray:
        return np.array([
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
        ])

    def shift(
        self, x_shift: float, y_shift: float,
    ) -> "BoundingBox":
        """Return a new BoundingBox shifted by offsets.

        Args:
            x_shift: Horizontal shift (positive = right).
            y_shift: Vertical shift (positive = down).
        """
        return BoundingBox(
            self._points + np.array([x_shift, y_shift]),
        )

    def scale(
        self, x_scale: float, y_scale: float,
    ) -> "BoundingBox":
        """Return a new BoundingBox scaled by factors.

        Args:
            x_scale: Horizontal scale factor.
            y_scale: Vertical scale factor.
        """
        matrix = np.array([[x_scale, 0], [0, y_scale]])
        return BoundingBox(self._points @ matrix)

    def rotate(
        self, angle: float, center: np.ndarray,
    ) -> "BoundingBox":
        """Return a new axis-aligned BoundingBox after
        rotation.

        Expands the four corners, rotates them, then takes
        the axis-aligned bounding box of the result.

        Args:
            angle: Rotation angle in degrees (positive =
                clockwise).
            center: Rotation center as ``[cx, cy]``.
        """
        corners = np.array([
            [self.x_min, self.y_min],
            [self.x_max, self.y_min],
            [self.x_max, self.y_max],
            [self.x_min, self.y_max],
        ])
        rad = np.deg2rad(-angle)
        cos, sin = np.cos(rad), np.sin(rad)
        rot = np.array([[cos, -sin], [sin, cos]])
        rotated = (corners - center) @ rot + center
        new_min = rotated.min(axis=0)
        new_max = rotated.max(axis=0)
        return BoundingBox(np.array([new_min, new_max]))

    def clip(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
    ) -> "BoundingBox":
        """Return a new BoundingBox clipped to bounds.

        Args:
            x_min: Left bound.
            y_min: Top bound.
            x_max: Right bound.
            y_max: Bottom bound.
        """
        clipped = np.array([
            [max(self.x_min, x_min), max(self.y_min, y_min)],
            [min(self.x_max, x_max), min(self.y_max, y_max)],
        ])
        return BoundingBox(clipped)

    def is_valid(self, min_area: float = 0) -> bool:
        """Check if the box is non-degenerate.

        Args:
            min_area: Minimum required area.
        """
        return (
            self.x_min < self.x_max
            and self.y_min < self.y_max
            and self.area > min_area
        )

    @property
    def state(self) -> ComponentState:
        return ComponentState.MATERIALIZED

    @property
    def is_materialized(self) -> bool:
        return True

    def materialize(self) -> None:
        pass
