"""DataPackage — a fully materialized sample."""
from __future__ import annotations

from typing import Any, Tuple


class DataPackage:
    """A fully materialized sample.

    All heavy data (images, audio, video) is loaded into
    memory.  Transforms receive ``DataPackage`` instances.

    Constructed by :meth:`Sample.materialize`.  Should not
    be instantiated directly by users.
    """

    def __init__(
        self, components: dict[str, Any],
    ) -> None:
        self._components = components

    def __getitem__(self, key: str) -> Any:
        return self._components[key]

    def __contains__(self, key: str) -> bool:
        return key in self._components

    @property
    def keys(self) -> Tuple[str, ...]:
        return tuple(self._components.keys())

    def get(
        self, key: str, default: Any = None,
    ) -> Any:
        """Return component by key, or *default*."""
        return self._components.get(key, default)
