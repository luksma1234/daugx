"""Sample — component holder with lazy materialization."""
from __future__ import annotations

from typing import (
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from daugx.core.data.component import Component
from daugx.errors.invalid_component import InvalidComponentError

T = TypeVar("T")


class Sample:
    """Component holder with lazy materialization.

    Takes positional ``Component`` instances — including
    annotation components (those with an ``applies_to``
    class attribute).
    Lightweight components (annotations, Text, Constant) are
    born materialized; heavy ones (Image) are materialized by
    calling :meth:`materialize`.

    Args:
        *components: Components to include in this sample.

    Raises:
        InvalidComponentError: If any annotation's ``target``
            references a component name that does not exist
            in the sample (orphaned), or if an annotation
            with ``target=None`` is ambiguous because
            multiple components of its ``applies_to`` type
            exist in the sample.
    """

    def __init__(self, *components: Component) -> None:
        self._components: List[Component] = list(components)
        self._validate_components()

    def _validate_components(self) -> None:
        """Validate annotations for orphans and ambiguity."""
        named = {
            c.name
            for c in self._components
            if c.name is not None
        }
        for comp in self._components:
            if not hasattr(comp, 'applies_to'):
                continue
            if comp.target is not None:
                if comp.target not in named:
                    raise InvalidComponentError(
                        f"{type(comp).__name__} has "
                        f"target={comp.target!r} but no "
                        f"component with that name exists "
                        f"in the sample."
                    )
            else:
                parent_type = comp.applies_to
                count = sum(
                    1
                    for c in self._components
                    if isinstance(c, parent_type)
                    and not hasattr(c, 'applies_to')
                )
                if count > 1:
                    raise InvalidComponentError(
                        f"Ambiguous {type(comp).__name__}: "
                        f"target=None but {count} "
                        f"{parent_type.__name__} components "
                        f"exist. Set target= to specify "
                        f"which one this annotation belongs "
                        f"to."
                    )

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
    ) -> List[Component]:
        """Return all annotation components.

        Args:
            target: If given, filter by ``annotation.target``.

        Returns:
            List of matching annotation ``Component`` instances.
        """
        results = []
        for comp in self._components:
            if hasattr(comp, 'applies_to'):
                if target is None or comp.target == target:
                    results.append(comp)
        return results

    @property
    def is_materialized(self) -> bool:
        """True if every component is materialized."""
        return all(
            c.is_materialized for c in self._components
        )

    def materialize(self) -> "Sample":
        """Materialize all components in-place and return
        *self*.

        Calls ``materialize()`` on every component.  Heavy
        components (e.g. ``Image``) load their data; light
        components treat this as a no-op.  Returns *self* so
        the call can be chained.
        """
        for comp in self._components:
            comp.materialize()
        return self

    def replacing(
        self,
        old: Component,
        new: Component,
    ) -> "Sample":
        """Return a new Sample with *old* swapped for *new*
        (identity match).

        Args:
            old: The component to replace.
            new: The replacement component.

        Returns:
            A new Sample with the substitution applied.
        """
        return Sample(*(
            new if c is old else c
            for c in self._components
        ))

    def __iter__(self) -> Iterator[Component]:
        return iter(self._components)

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

    def show(
        self,
        components: Optional[
            Union[
                Type[Component],
                Tuple[Type[Component], ...],
            ]
        ] = None,
    ) -> None:
        """Visualize this sample using cv2.imshow.

        Paints spatial annotations (``ImageBoundingBox``,
        ``ImagePolygon``, ``ImageKeyPoint``) onto each
        ``Image`` with per-label colors — opaque outline,
        transparent fill — then assembles a grid and opens
        an OpenCV window.  Press any key to close.

        Calls :meth:`materialize` automatically if the
        sample has not been materialized yet.

        Args:
            components: Optional type or tuple of component
                types to include.  When ``None``, all
                ``Image`` and spatial annotation components
                are shown.  Pass ``Image`` to show only
                images.  Pass ``(Image, ImageBoundingBox)``
                to show images with only bounding boxes.
        """
        from daugx.core.data._visualizer import show_sample
        show_sample(self, components)

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
