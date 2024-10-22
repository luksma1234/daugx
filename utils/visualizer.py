from typing import Tuple, Union, Dict, Optional

from daugx.core.augmentation.annotations import Annotations
from daugx.utils import is_in_dict

import cv2
import numpy as np


DEFAULT_COLOR = (0, 0, 255)
IS_CLOSED = True
BORDER_THICKNESS = 2


class Colors:
    def __init__(self) -> None:
        self.colors = {}

    def __getitem__(self, label: str) -> Tuple[int, int, int]:
        if is_in_dict(label, self.colors):
            return self.colors[label]
        else:
            return DEFAULT_COLOR

    def __setitem__(self, label: str, color: Union[Tuple[int, int, int], np.ndarray]) -> None:
        self.add(label, color)

    def add(self, label: str, color: Union[Tuple[int, int, int], np.ndarray]) -> None:
        color = np.array(color, dtype=np.uint8)
        assert len(color) == 3
        self.colors[label] = color

    def set_colors(self, colors: Dict[str, Union[Tuple[int, int, int], np.ndarray]]) -> None:
        self._reset()
        for label, color in colors.items():
            self.add(label, color)

    def _reset(self):
        self.colors = {}


class Visualizer:

    def __init__(self, colors: Optional[Colors] = None, wait_key: int = 0) -> None:
        self.image = None
        self.annots = None
        self.colors = colors
        self.image_width = None
        self.image_height = None
        self.wait_key = wait_key

    def show(self, image: np.ndarray, annots: Annotations):
        self.image = image
        self.annots = annots
        self.image_width, self.image_height = np.shape(self.image)[:2]
        self._assemble()
        cv2.imshow("AugmentedImage", self.image)
        cv2.waitKey(self.wait_key)

    def _assemble(self):
        # TODO: Update Visualizer - enhance box visibility
        for annot in self.annots:
            # denormalized boundary
            boundary = annot.boundary.visualize.astype(np.int32)
            boundary = boundary.reshape((-1, 1, 2))
            color = DEFAULT_COLOR if self.colors is None else self.colors[annot.label.name]
            self.image = cv2.polylines(
                self.image.copy(),
                [boundary],
                IS_CLOSED,
                color,
                BORDER_THICKNESS
            )
