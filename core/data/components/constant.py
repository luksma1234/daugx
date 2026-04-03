"""Constant component — holds an immutable value."""
from typing import Any, Optional

from daugx.core.data.component import Component, ComponentState


class Constant(Component):
    """Holds any Python value unchanged through augmentation.

    Use this to attach dataset-specific metadata (e.g. image
    id, split name, difficulty score) to a ``Sample`` or
    ``DataPackage`` that transforms must not modify.

    Always ``MATERIALIZED`` — no heavy I/O.

    Args:
        value: The value to store.
        name: Optional disambiguation name.
    """

    def __init__(
        self,
        value: Any,
        name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._component_name = name
        self._value = value

    @property
    def value(self) -> Any:
        """The stored constant value."""
        return self._value

    @property
    def state(self) -> ComponentState:
        """Constants are always MATERIALIZED."""
        return ComponentState.MATERIALIZED

    @property
    def is_materialized(self) -> bool:
        """Always ``True``."""
        return True

    def materialize(self) -> None:
        """No-op — constants are born materialized."""
