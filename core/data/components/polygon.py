"""Polygon component."""
import numpy as np

from daugx.core.data.component import Component, ComponentState


class Polygon(Component):
    """Polygon boundary.  Always materialized.

    Args:
        points: Array of shape (n, 2) with n >= 3.

    Raises:
        ValueError: If fewer than 3 points or wrong shape.
    """

    def __init__(self, points: np.ndarray) -> None:
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(
                "Expected shape (n, 2), "
                f"got {pts.shape}"
            )
        if pts.shape[0] < 3:
            raise ValueError(
                "Polygon requires at least 3 points, "
                f"got {pts.shape[0]}"
            )
        self._points = pts

    @property
    def points(self) -> np.ndarray:
        return self._points

    @property
    def area(self) -> float:
        """Area via the shoelace formula."""
        x = self._points[:, 0]
        y = self._points[:, 1]
        return float(
            0.5 * abs(
                np.dot(x, np.roll(y, -1))
                - np.dot(y, np.roll(x, -1))
            )
        )

    @property
    def center(self) -> np.ndarray:
        """Centroid (mean of vertices)."""
        return self._points.mean(axis=0)

    @property
    def state(self) -> ComponentState:
        return ComponentState.MATERIALIZED

    @property
    def is_materialized(self) -> bool:
        return True

    def materialize(self) -> None:
        pass
