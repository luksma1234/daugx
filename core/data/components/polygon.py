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

    def shift(
        self, x_shift: float, y_shift: float,
    ) -> "Polygon":
        """Return a new Polygon shifted by offsets.

        Args:
            x_shift: Horizontal shift (positive = right).
            y_shift: Vertical shift (positive = down).
        """
        return Polygon(
            self._points + np.array([x_shift, y_shift]),
        )

    def scale(
        self, x_scale: float, y_scale: float,
    ) -> "Polygon":
        """Return a new Polygon scaled by factors.

        Args:
            x_scale: Horizontal scale factor.
            y_scale: Vertical scale factor.
        """
        matrix = np.array([[x_scale, 0], [0, y_scale]])
        return Polygon(self._points @ matrix)

    def rotate(
        self, angle: float, center: np.ndarray,
    ) -> "Polygon":
        """Return a new Polygon rotated around a center.

        Args:
            angle: Rotation angle in degrees (positive =
                clockwise).
            center: Rotation center as ``[cx, cy]``.
        """
        rad = np.deg2rad(-angle)
        cos, sin = np.cos(rad), np.sin(rad)
        rot = np.array([[cos, -sin], [sin, cos]])
        rotated = (self._points - center) @ rot + center
        return Polygon(rotated)

    def clip(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
    ) -> "Polygon":
        """Return a new Polygon with vertices clipped to
        bounds.

        Args:
            x_min: Left bound.
            y_min: Top bound.
            x_max: Right bound.
            y_max: Bottom bound.
        """
        clipped = np.clip(
            self._points,
            [x_min, y_min],
            [x_max, y_max],
        )
        return Polygon(clipped)

    def is_valid(self, min_area: float = 0) -> bool:
        """Check if the polygon is non-degenerate.

        Args:
            min_area: Minimum required area.
        """
        return self.area > min_area

    @property
    def state(self) -> ComponentState:
        return ComponentState.MATERIALIZED

    @property
    def is_materialized(self) -> bool:
        return True

    def materialize(self) -> None:
        pass
