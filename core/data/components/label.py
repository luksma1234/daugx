"""ImageCategory annotation component."""
from typing import Optional

from daugx.core.data.annotation import Annotation


class ImageCategory(Annotation):
    """Image-level classification annotation.

    Associates an image with a category.  Use this for
    image classification tasks where a category label
    applies to an entire image rather than a specific
    region.

    Args:
        class_id: Integer class identifier.
        class_name: Human-readable class name.
        target: ``name`` of the parent ``Image`` component
            this annotation belongs to.
        name: Optional disambiguation name.
    """

    def __init__(
        self,
        class_id: int,
        class_name: Optional[str] = None,
        target: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(target=target, name=name)
        self._class_id = class_id
        self._class_name = class_name

    @property
    def class_id(self) -> int:
        return self._class_id

    @property
    def class_name(self) -> Optional[str]:
        return self._class_name
