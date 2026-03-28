"""Label component."""
from typing import Optional

from daugx.core.data.component import Component, ComponentState


class Label(Component):
    """Classification label.  Always materialized.

    Args:
        class_id: Integer class identifier.
        name: Human-readable class name.
    """

    def __init__(
        self,
        class_id: int,
        name: Optional[str] = None,
    ) -> None:
        self._class_id = class_id
        self._name = name

    @property
    def class_id(self) -> int:
        return self._class_id

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def state(self) -> ComponentState:
        return ComponentState.MATERIALIZED

    @property
    def is_materialized(self) -> bool:
        return True

    def materialize(self) -> None:
        pass
