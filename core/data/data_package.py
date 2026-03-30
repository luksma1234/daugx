"""DataPackage — a fully materialized sample."""
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


class DataPackage:
    """A fully materialized sample.

    Stores components as a flat list.  Provides type-based
    retrieval via :meth:`get` and :meth:`get_all`.

    Constructed by :meth:`Sample.materialize` or directly
    by transforms.
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

    def replacing(
        self,
        old: Component,
        new: Component,
    ) -> DataPackage:
        """Return a new DataPackage with *old* swapped for
        *new* (identity match).

        Args:
            old: The component to replace.
            new: The replacement component.

        Returns:
            A new DataPackage with the substitution.
        """
        return DataPackage(*(
            new if c is old else c
            for c in self._components
        ))

    def merge(self, other: DataPackage) -> DataPackage:
        """Return a new DataPackage combining components
        from both *self* and *other*.

        Neither original is mutated.

        Args:
            other: The DataPackage to merge into this one.

        Returns:
            A new DataPackage with all components from both.
        """
        return DataPackage(
            *self._components, *other._components,
        )

    def __add__(self, other: DataPackage) -> DataPackage:
        return self.merge(other)

    def modalities(self) -> Set[Optional[str]]:
        """Return the set of distinct names across all
        components.

        Useful for discovering which modalities are present
        in a multimodal package.  Unnamed items contribute
        ``None``.

        Returns:
            Set of name strings (and ``None`` for unnamed
            items).
        """
        return {comp.name for comp in self._components}
