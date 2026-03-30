"""Sample — preloaded component holder."""
from __future__ import annotations

from typing import (
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
)

from daugx.core.data.annotation import Annotation
from daugx.core.data.component import Component

T = TypeVar("T")


class Sample:
    """One augmentation unit in preloaded state.

    Takes positional ``Component`` instances — including
    annotation components (subclasses of ``Annotation``).
    No string keys.

    Args:
        *components: Components to include in this sample.
    """

    def __init__(self, *components: Component) -> None:
        self._components: List[Component] = list(components)

    @property
    def components(self) -> Tuple[Component, ...]:
        """All top-level components."""
        return tuple(self._components)

    def get(
        self,
        component_type: Type[T],
        name: Optional[str] = None,
    ) -> Optional[T]:
        """Return the first component of *type*.

        Args:
            component_type: The type to search for.
            name: If given, also match ``component.name``.

        Returns:
            The first match, or ``None``.
        """
        for comp in self._components:
            if isinstance(comp, component_type):
                if name is None or comp.name == name:
                    return comp  # type: ignore[return-value]
        return None

    def get_all(
        self,
        component_type: Type[T],
        name: Optional[str] = None,
    ) -> List[T]:
        """Return all components of *type*.

        Args:
            component_type: The type to search for.
            name: If given, also match ``component.name``.

        Returns:
            List of matches (may be empty).
        """
        results: List[T] = []
        for comp in self._components:
            if isinstance(comp, component_type):
                if name is None or comp.name == name:
                    results.append(comp)  # type: ignore[arg-type]
        return results

    def get_annotations(
        self, target: Optional[str] = None,
    ) -> List[Annotation]:
        """Return all annotation components.

        Args:
            target: If given, filter by ``annotation.target``.

        Returns:
            List of matching ``Annotation`` instances.
        """
        results = []
        for comp in self._components:
            if isinstance(comp, Annotation):
                if target is None or comp.target == target:
                    results.append(comp)
        return results

    @property
    def is_materialized(self) -> bool:
        """True if every component is materialized."""
        return all(
            c.is_materialized for c in self._components
        )

    def materialize(self) -> "DataPackage":
        """Materialize all components and return a
        :class:`DataPackage`.

        Calls ``materialize()`` on every component in-place,
        then returns a new ``DataPackage`` wrapping the same
        items.
        """
        for comp in self._components:
            comp.materialize()

        from daugx.core.data.data_package import DataPackage
        return DataPackage(*self._components)

    def merge(self, other: Sample) -> Sample:
        """Return a new Sample combining components from
        both *self* and *other*.

        Neither original is mutated.

        Args:
            other: The Sample to merge into this one.

        Returns:
            A new Sample with all components from both.
        """
        return Sample(*self._components, *other._components)

    def __add__(self, other: Sample) -> Sample:
        return self.merge(other)

    def modalities(self) -> Set[Optional[str]]:
        """Return the set of distinct names across all
        components.

        Useful for discovering which modalities are present
        in a multimodal sample.  Unnamed items contribute
        ``None``.

        Returns:
            Set of name strings (and ``None`` for unnamed
            items).
        """
        return {comp.name for comp in self._components}
