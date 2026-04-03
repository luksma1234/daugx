"""ImageCategory annotation component."""
from typing import Optional

from daugx.core.data.component import Component, ComponentState
from daugx.core.data.components.image import Image


class ImageCategory(Component):
    """Image-level classification annotation.

    Associates an image with a category.  Use this for
    image classification tasks where a category label
    applies to an entire image rather than a specific
    region.

    Attributes:
        applies_to: Categories annotate :class:`Image`
            components.

    Args:
        class_id: Integer class identifier.
        class_name: Human-readable class name.
        target: ``name`` of the parent ``Image`` component
            this annotation belongs to.
        name: Optional disambiguation name.
    """

    applies_to = Image

    def __init__(
        self,
        class_id: int,
        class_name: Optional[str] = None,
        target: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._component_name = name
        self._target = target
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
    def class_id(self) -> int:
        return self._class_id

    @property
    def class_name(self) -> Optional[str]:
        return self._class_name
