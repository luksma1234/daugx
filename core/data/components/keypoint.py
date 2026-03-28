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

    @property
    def state(self) -> ComponentState:
        return ComponentState.MATERIALIZED

    @property
    def is_materialized(self) -> bool:
        return True

    def materialize(self) -> None:
        pass
