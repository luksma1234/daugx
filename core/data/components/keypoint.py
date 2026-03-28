"""KeyPoint component."""
from typing import Optional

import numpy as np

from daugx.core.data.component import Component, ComponentState


class KeyPoint(Component):
    """Single keypoint.  Always materialized.

    Args:
        x: X coordinate.
        y: Y coordinate.
        visibility: Optional visibility flag (0/1/2).
    """

    def __init__(
        self,
        x: float,
        y: float,
        visibility: Optional[int] = None,
    ) -> None:
        self._x = float(x)
        self._y = float(y)
        self._visibility = visibility

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def visibility(self) -> Optional[int]:
        return self._visibility

    @property
    def point(self) -> np.ndarray:
        return np.array([self._x, self._y])

    def shift(
        self, x_shift: float, y_shift: float,
    ) -> "KeyPoint":
        """Return a new KeyPoint shifted by offsets.

        Args:
            x_shift: Horizontal shift (positive = right).
            y_shift: Vertical shift (positive = down).
        """
        return KeyPoint(
            self._x + x_shift,
            self._y + y_shift,
            self._visibility,
        )

    def scale(
        self, x_scale: float, y_scale: float,
    ) -> "KeyPoint":
        """Return a new KeyPoint scaled by factors.

        Args:
            x_scale: Horizontal scale factor.
            y_scale: Vertical scale factor.
        """
        return KeyPoint(
            self._x * x_scale,
            self._y * y_scale,
            self._visibility,
        )

    def rotate(
        self, angle: float, center: np.ndarray,
    ) -> "KeyPoint":
        """Return a new KeyPoint rotated around a center.

        Args:
            angle: Rotation angle in degrees (positive =
                clockwise).
            center: Rotation center as ``[cx, cy]``.
        """
        rad = np.deg2rad(-angle)
        cos, sin = np.cos(rad), np.sin(rad)
        rot = np.array([[cos, -sin], [sin, cos]])
        pt = np.array([self._x, self._y]) - center
        rotated = pt @ rot + center
        return KeyPoint(
            float(rotated[0]),
            float(rotated[1]),
            self._visibility,
        )

    def clip(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
    ) -> "KeyPoint":
        """Return a new KeyPoint clipped to bounds.

        Args:
            x_min: Left bound.
            y_min: Top bound.
            x_max: Right bound.
            y_max: Bottom bound.
        """
        return KeyPoint(
            float(np.clip(self._x, x_min, x_max)),
            float(np.clip(self._y, y_min, y_max)),
            self._visibility,
        )

    def is_valid(
        self,
        x_min: float = 0,
        y_min: float = 0,
        x_max: float = float("inf"),
        y_max: float = float("inf"),
    ) -> bool:
        """Check if the keypoint is within bounds.

        Args:
            x_min: Left bound.
            y_min: Top bound.
            x_max: Right bound.
            y_max: Bottom bound.
        """
        return (
            x_min <= self._x <= x_max
            and y_min <= self._y <= y_max
        )

    @property
    def state(self) -> ComponentState:
        return ComponentState.MATERIALIZED

    @property
    def is_materialized(self) -> bool:
        return True

    def materialize(self) -> None:
        pass
