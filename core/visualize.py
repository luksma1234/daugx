import cv2
from .annotations import Annotations, Annotation, BoxAnnotation
from ..utils import is_in_dict
import numpy as np
from typing import Tuple, Union, Dict, Optional

DEFAULT_COLOR = (0, 0, 255)
IS_CLOSED = True
BORDER_THICKNESS = 2

# TODO: Boundary is invalid
# TODO: Check if img_size is necessary in annotations
# TODO: Research in Albumentations / augly how they managed annotations


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
        assert len(color) == 3, f"Unable to parse {color} as color. Length does not fit."
        self.colors[label] = color

    def set_colors(self, colors: Dict[str, Union[Tuple[int, int, int], np.ndarray]]) -> None:
        self.reset()
        for label, color in colors.items():
            self.add(label, color)

    def reset(self):
        self.colors = {}


class Visualize:

    def __init__(self, image: np.ndarray, annots: Annotations, colors: Optional[Colors] = None) -> None:
        self.image = image
        self.annots = annots
        self.colors = colors
        self.image_width, self.image_height = np.shape(self.image)[:2]
        self._show()

    def _show(self):
        self._assemble()

        cv2.imshow("AugmentedImage", self.image)
        cv2.waitKey(0)

    def _assemble(self):
        for annot in self.annots:
            # denormalized boundary
            boundary = annot.boundary.astype(np.int32)
            boundary = boundary.reshape((-1, 1, 2))
            color = DEFAULT_COLOR if self.colors is None else self.colors[annot.label.name]
            self.image = cv2.polylines(
                self.image.copy(),
                [boundary],
                IS_CLOSED,
                color,
                BORDER_THICKNESS
            )

























