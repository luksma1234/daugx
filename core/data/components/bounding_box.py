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

    @property
    def state(self) -> ComponentState:
        return ComponentState.MATERIALIZED

    @property
    def is_materialized(self) -> bool:
        return True

    def materialize(self) -> None:
        pass
