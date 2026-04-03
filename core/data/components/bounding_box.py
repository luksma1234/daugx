"""ImageBoundingBox annotation component."""
from typing import Optional

import numpy as np

from daugx.core.data.component import Component, ComponentState
from daugx.core.data.components.image import Image

_BOX_TYPES = ("XYXY", "XYWH", "CXCYWH", "YXYX")


def _xywh_to_xyxy(pts: np.ndarray) -> np.ndarray:
    """Convert ``[[x, y], [w, h]]`` to ``[[x_min, y_min],
    [x_max, y_max]]``."""
    return np.array([
        pts[0],
        pts[0] + pts[1],
    ])


def _cxcywh_to_xyxy(pts: np.ndarray) -> np.ndarray:
    """Convert ``[[cx, cy], [w, h]]`` to ``[[x_min, y_min],
    [x_max, y_max]]``."""
    half = pts[1] / 2
    return np.array([
        pts[0] - half,
        pts[0] + half,
    ])


def _yxyx_to_xyxy(pts: np.ndarray) -> np.ndarray:
    """Convert ``[[y_min, x_min], [y_max, x_max]]`` to
    ``[[x_min, y_min], [x_max, y_max]]``."""
    return pts[:, ::-1]


_CONVERTERS = {
    "XYWH": _xywh_to_xyxy,
    "CXCYWH": _cxcywh_to_xyxy,
    "YXYX": _yxyx_to_xyxy,
}


class ImageBoundingBox(Component):
    """Axis-aligned bounding box annotation for images.

    Stores coordinates as a (2, 2) numpy array:
    ``[[x_min, y_min], [x_max, y_max]]``.

    Attributes:
        applies_to: Bounding boxes annotate :class:`Image`
            components.

    Args:
        points: Array of shape (2, 2). Interpretation depends
            on *box_type*.
        box_type: Coordinate format of *points*. One of
            ``"XYXY"`` (default), ``"XYWH"``, ``"CXCYWH"``,
            ``"YXYX"``. Case-insensitive.
        class_id: Integer class identifier.
        class_name: Human-readable class name.
        target: ``name`` of the parent ``Image`` component
            this annotation belongs to.
        name: Optional disambiguation name.

    Raises:
        ValueError: If *points* is not shape (2, 2) or
            *box_type* is not recognised.
    """

    applies_to = Image

    def __init__(
        self,
        points: np.ndarray,
        class_id: Optional[int] = None,
        class_name: Optional[str] = None,
        target: Optional[str] = None,
        name: Optional[str] = None,
        box_type: str = "XYXY",
    ) -> None:
        super().__init__()
        self._component_name = name
        self._target = target
        pts = np.asarray(points, dtype=float)
        if pts.shape == (4,):
            pts = pts.reshape(2, 2)
        if pts.shape != (2, 2):
            raise ValueError(
                f"Expected shape (2, 2) or (4,), got {pts.shape}"
            )
        box_type = box_type.upper()
        if box_type not in _BOX_TYPES:
            raise ValueError(
                f"Invalid box_type '{box_type}'. "
                f"Expected one of {_BOX_TYPES}."
            )
        converter = _CONVERTERS.get(box_type)
        if converter is not None:
            pts = converter(pts)
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
    ) -> "ImageBoundingBox":
        """Return a new ImageBoundingBox shifted by offsets.

        Args:
            x_shift: Horizontal shift (positive = right).
            y_shift: Vertical shift (positive = down).
        """
        return ImageBoundingBox(
            self._points + np.array([x_shift, y_shift]),
            class_id=self._class_id,
            class_name=self._class_name,
            target=self._target,
            name=self._component_name,
        )

    def scale(
        self, x_scale: float, y_scale: float,
    ) -> "ImageBoundingBox":
        """Return a new ImageBoundingBox scaled by factors.

        Args:
            x_scale: Horizontal scale factor.
            y_scale: Vertical scale factor.
        """
        matrix = np.array([[x_scale, 0], [0, y_scale]])
        return ImageBoundingBox(
            self._points @ matrix,
            class_id=self._class_id,
            class_name=self._class_name,
            target=self._target,
            name=self._component_name,
        )

    def rotate(
        self, angle: float, center: np.ndarray,
    ) -> "ImageBoundingBox":
        """Return a new axis-aligned ImageBoundingBox after
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
        return ImageBoundingBox(
            np.array([new_min, new_max]),
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
    ) -> "ImageBoundingBox":
        """Return a new ImageBoundingBox clipped to bounds.

        Args:
            x_min: Left bound.
            y_min: Top bound.
            x_max: Right bound.
            y_max: Bottom bound.
        """
        clipped = np.array([
            [
                max(self.x_min, x_min),
                max(self.y_min, y_min),
            ],
            [
                min(self.x_max, x_max),
                min(self.y_max, y_max),
            ],
        ])
        return ImageBoundingBox(
            clipped,
            class_id=self._class_id,
            class_name=self._class_name,
            target=self._target,
            name=self._component_name,
        )

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
