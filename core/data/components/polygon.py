"""ImagePolygon annotation component."""
from typing import Optional

import numpy as np

from daugx.core.data.component import Component, ComponentState
from daugx.core.data.components.image import Image


class ImagePolygon(Component):
    """Polygon annotation for images.

    Attributes:
        applies_to: Polygons annotate :class:`Image`
            components.

    Args:
        points: Array of shape (n, 2) with n >= 3.
        class_id: Integer class identifier.
        class_name: Human-readable class name.
        target: ``name`` of the parent ``Image`` component
            this annotation belongs to.
        name: Optional disambiguation name.

    Raises:
        ValueError: If fewer than 3 points or wrong shape.
    """

    applies_to = Image

    def __init__(
        self,
        points: np.ndarray,
        class_id: Optional[int] = None,
        class_name: Optional[str] = None,
        target: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._component_name = name
        self._target = target
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
        self._class_id = class_id
        self._class_name = class_name

    @property
    def target(self) -> Optional[str]:
        """Name of the parent Image this annotation belongs to."""
        return self._target

    @property
    def state(self) -> ComponentState:
        return ComponentState.MATERIALIZED

    @property
    def is_materialized(self) -> bool:
        return True

    def materialize(self) -> None:
        pass

    @property
    def points(self) -> np.ndarray:
        return self._points

    @property
    def class_id(self) -> Optional[int]:
        return self._class_id

    @property
    def class_name(self) -> Optional[str]:
        return self._class_name

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
    ) -> "ImagePolygon":
        """Return a new ImagePolygon shifted by offsets.

        Args:
            x_shift: Horizontal shift (positive = right).
            y_shift: Vertical shift (positive = down).
        """
        return ImagePolygon(
            self._points + np.array([x_shift, y_shift]),
            class_id=self._class_id,
            class_name=self._class_name,
            target=self._target,
            name=self._component_name,
        )

    def scale(
        self, x_scale: float, y_scale: float,
    ) -> "ImagePolygon":
        """Return a new ImagePolygon scaled by factors.

        Args:
            x_scale: Horizontal scale factor.
            y_scale: Vertical scale factor.
        """
        matrix = np.array([[x_scale, 0], [0, y_scale]])
        return ImagePolygon(
            self._points @ matrix,
            class_id=self._class_id,
            class_name=self._class_name,
            target=self._target,
            name=self._component_name,
        )

    def rotate(
        self, angle: float, center: np.ndarray,
    ) -> "ImagePolygon":
        """Return a new ImagePolygon rotated around a center.

        Args:
            angle: Rotation angle in degrees (positive =
                clockwise).
            center: Rotation center as ``[cx, cy]``.
        """
        rad = np.deg2rad(-angle)
        cos, sin = np.cos(rad), np.sin(rad)
        rot = np.array([[cos, -sin], [sin, cos]])
        rotated = (self._points - center) @ rot + center
        return ImagePolygon(
            rotated,
            class_id=self._class_id,
            class_name=self._class_name,
            target=self._target,
            name=self._component_name,
        )

    def clip(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
    ) -> "ImagePolygon":
        """Return a new ImagePolygon with vertices clipped to
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
        return ImagePolygon(
            clipped,
            class_id=self._class_id,
            class_name=self._class_name,
            target=self._target,
            name=self._component_name,
        )

    def is_valid(self, min_area: float = 0) -> bool:
        """Check if the polygon is non-degenerate.

        Args:
            min_area: Minimum required area.
        """
        return self.area > min_area
