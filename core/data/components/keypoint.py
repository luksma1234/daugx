"""ImageKeyPoint annotation component."""
from typing import Optional

import numpy as np

from daugx.core.data.annotation import Annotation


class ImageKeyPoint(Annotation):
    """Single keypoint annotation for images.

    Args:
        x: X coordinate.
        y: Y coordinate.
        visibility: Optional visibility flag (0/1/2).
        class_id: Integer class identifier.
        class_name: Human-readable class name.
        target: ``name`` of the parent ``Image`` component
            this annotation belongs to.
        name: Optional disambiguation name.
    """

    def __init__(
        self,
        x: float,
        y: float,
        visibility: Optional[int] = None,
        class_id: Optional[int] = None,
        class_name: Optional[str] = None,
        target: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(target=target, name=name)
        self._x = float(x)
        self._y = float(y)
        self._visibility = visibility
        self._class_id = class_id
        self._class_name = class_name

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
    def class_id(self) -> Optional[int]:
        return self._class_id

    @property
    def class_name(self) -> Optional[str]:
        return self._class_name

    @property
    def point(self) -> np.ndarray:
        return np.array([self._x, self._y])

    def shift(
        self, x_shift: float, y_shift: float,
    ) -> "ImageKeyPoint":
        """Return a new ImageKeyPoint shifted by offsets.

        Args:
            x_shift: Horizontal shift (positive = right).
            y_shift: Vertical shift (positive = down).
        """
        return ImageKeyPoint(
            self._x + x_shift,
            self._y + y_shift,
            self._visibility,
            class_id=self._class_id,
            class_name=self._class_name,
            target=self._target,
            name=self._component_name,
        )

    def scale(
        self, x_scale: float, y_scale: float,
    ) -> "ImageKeyPoint":
        """Return a new ImageKeyPoint scaled by factors.

        Args:
            x_scale: Horizontal scale factor.
            y_scale: Vertical scale factor.
        """
        return ImageKeyPoint(
            self._x * x_scale,
            self._y * y_scale,
            self._visibility,
            class_id=self._class_id,
            class_name=self._class_name,
            target=self._target,
            name=self._component_name,
        )

    def rotate(
        self, angle: float, center: np.ndarray,
    ) -> "ImageKeyPoint":
        """Return a new ImageKeyPoint rotated around a center.

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
        return ImageKeyPoint(
            float(rotated[0]),
            float(rotated[1]),
            self._visibility,
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
    ) -> "ImageKeyPoint":
        """Return a new ImageKeyPoint clipped to bounds.

        Args:
            x_min: Left bound.
            y_min: Top bound.
            x_max: Right bound.
            y_max: Bottom bound.
        """
        return ImageKeyPoint(
            float(np.clip(self._x, x_min, x_max)),
            float(np.clip(self._y, y_min, y_max)),
            self._visibility,
            class_id=self._class_id,
            class_name=self._class_name,
            target=self._target,
            name=self._component_name,
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
