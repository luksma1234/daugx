"""Sample — preloaded component holder."""
from __future__ import annotations

from typing import Any, Tuple

from daugx.core.data.component import Component


class Sample:
    """One augmentation unit in preloaded state.

    Users build samples in a loop, assigning components
    by keyword matching the schema keys.

    Args:
        **components: Keyword arguments mapping keys to
            components or lists of component tuples.
    """

    def __init__(self, **components: Any) -> None:
        self._components: dict[str, Any] = dict(components)

    def __getitem__(self, key: str) -> Any:
        return self._components[key]

    def __contains__(self, key: str) -> bool:
        return key in self._components

    @property
    def keys(self) -> Tuple[str, ...]:
        return tuple(self._components.keys())

    @property
    def is_materialized(self) -> bool:
        """True if every component is materialized."""
        for value in self._components.values():
            if isinstance(value, Component):
                if not value.is_materialized:
                    return False
            elif isinstance(value, list):
                for element in value:
                    if isinstance(element, tuple):
                        for comp in element:
                            if (
                                isinstance(comp, Component)
                                and not comp.is_materialized
                            ):
                                return False
                    elif (
                        isinstance(element, Component)
                        and not element.is_materialized
                    ):
                        return False
        return True

    def materialize(self) -> "DataPackage":
        """Materialize all components and return a
        :class:`DataPackage`.

        Calls ``component.materialize()`` on every
        component in-place, then returns a new
        ``DataPackage`` wrapping the same dict.
        """
        for value in self._components.values():
            if isinstance(value, Component):
                value.materialize()
            elif isinstance(value, list):
                for element in value:
                    if isinstance(element, tuple):
                        for comp in element:
                            if isinstance(
                                comp, Component
                            ):
                                comp.materialize()
                    elif isinstance(element, Component):
                        element.materialize()

        from daugx.core.data.data_package import (
            DataPackage,
        )
        return DataPackage(self._components)
