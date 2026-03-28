"""Image component."""
from typing import Optional

import cv2
import numpy as np

from daugx.core.data.component import Component, ComponentState


class Image(Component):
    """Raster image component.

    Starts in ``PRELOADED`` state (path only).  Call
    :meth:`materialize` to load pixel data from disk.

    Args:
        path: Filesystem path to the image file.
        format_hint: Optional format string ("png", "jpg").
    """

    def __init__(
        self,
        path: str,
        format_hint: Optional[str] = None,
    ) -> None:
        self._path = path
        self._format_hint = format_hint
        self._data: Optional[np.ndarray] = None
        self._state = ComponentState.PRELOADED

    @property
    def path(self) -> str:
        return self._path

    @property
    def format_hint(self) -> Optional[str]:
        return self._format_hint

    @property
    def data(self) -> np.ndarray:
        """Pixel data (H, W, C).

        Raises:
            RuntimeError: If not yet materialized.
        """
        if self._data is None:
            raise RuntimeError(
                "Image not materialized. "
                "Call materialize() first."
            )
        return self._data

    @property
    def state(self) -> ComponentState:
        return self._state

    @property
    def is_materialized(self) -> bool:
        return self._state == ComponentState.MATERIALIZED

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        format_hint: Optional[str] = None,
    ) -> "Image":
        """Create a materialized Image from pixel data.

        Args:
            data: Pixel array (H, W, C).
            format_hint: Optional format string.

        Returns:
            A new Image in ``MATERIALIZED`` state.
        """
        img = cls.__new__(cls)
        img._path = ""
        img._format_hint = format_hint
        img._data = data
        img._state = ComponentState.MATERIALIZED
        return img

    def materialize(self) -> None:
        """Read image from disk via ``cv2.imread``.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        if self._state == ComponentState.MATERIALIZED:
            return
        img = cv2.imread(self._path)
        if img is None:
            raise FileNotFoundError(
                f"Cannot read image: {self._path}"
            )
        self._data = img
        self._state = ComponentState.MATERIALIZED
