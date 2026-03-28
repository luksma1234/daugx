"""Data components for the daugx data loading system."""
from daugx.core.data.components.bounding_box import (
    BoundingBox,
)
from daugx.core.data.components.image import Image
from daugx.core.data.components.keypoint import KeyPoint
from daugx.core.data.components.label import Label
from daugx.core.data.components.polygon import Polygon
from daugx.core.data.components.text import Text

__all__ = [
    "BoundingBox",
    "Image",
    "KeyPoint",
    "Label",
    "Polygon",
    "Text",
]
