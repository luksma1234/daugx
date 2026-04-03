"""Transform base classes."""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Type

import numpy as np

from daugx.core.data.component import Component
from daugx.core.data.sample import Sample


class Transform(ABC):
    """Base class for single-input transforms.

    All transforms accept a materialized ``Sample`` and
    return a new materialized ``Sample``.  The
    ``inflation`` attribute controls pipeline sample-count
    semantics (always 1.0 for single-input transforms).

    Subclasses declare ALL component types they operate on
    in ``operates_on`` — including both primary types (e.g.
    ``Image``) and any dependent annotation types (e.g.
    ``ImageBoundingBox``).  The single abstract method
    ``_apply`` receives one component at a time.

    The base ``apply()`` handles dispatch order and
    annotation scoping: primary types are processed first,
    then dependent types (those with an ``applies_to``
    attribute) are processed in the context of the primary
    component they target.

    Class attributes:
        operates_on: All component types this transform
            handles (e.g.
            ``(Image, ImageBoundingBox, ImagePolygon)``).
    """

    inflation: float = 1.0
    operates_on: Tuple[Type[Component], ...] = ()

    def apply(
        self,
        sample: Sample,
        rng: Optional[np.random.Generator] = None,
    ) -> Sample:
        """Dispatch transform to all applicable components.

        Primary component types (those without an
        ``applies_to`` attribute, e.g. ``Image``, ``Text``)
        are processed first.  For each primary component
        transformed, dependent components (those with
        ``applies_to``, e.g. ``ImageBoundingBox``) that are
        scoped to that primary are then processed via
        ``_apply``.  All other components pass through
        unchanged.  Dependents returning ``None`` from
        ``_apply`` are dropped (e.g. a bbox clipped to
        nothing).

        Args:
            sample: Fully materialized sample.
            rng: Random number generator for
                reproducibility.

        Returns:
            A new Sample with the transform applied to
            all matching components.
        """
        result: List[Component] = list(sample.components)

        primary_types = tuple(
            t for t in self.operates_on
            if not hasattr(t, 'applies_to')
        )
        dependent_types = tuple(
            t for t in self.operates_on
            if hasattr(t, 'applies_to')
        )

        for comp_type in primary_types:
            for original in sample.get_all(comp_type):
                transformed = self._apply(original, rng)
                result = [
                    transformed if c is original else c
                    for c in result
                ]
                if not dependent_types:
                    continue
                new_result: List[Component] = []
                for comp in result:
                    if (
                        isinstance(comp, dependent_types)
                        and (
                            comp.target is None
                            or comp.target == original.name
                        )
                    ):
                        new_comp = self._apply(comp, rng)
                        if new_comp is not None:
                            new_result.append(new_comp)
                    else:
                        new_result.append(comp)
                result = new_result

        return Sample(*result)

    @abstractmethod
    def _apply(
        self,
        component: Component,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[Component]:
        """Transform a single component.

        Called for every component whose type is listed in
        ``operates_on``.  For primary types (e.g. ``Image``)
        the return value replaces the original component.
        For dependent types (e.g. ``ImageBoundingBox``)
        returning ``None`` drops the component from the
        output sample.

        Args:
            component: The component to transform.
            rng: Random number generator.

        Returns:
            Transformed component, or ``None`` to drop it
            (only meaningful for dependent types).
        """
        ...

    @abstractmethod
    def _key(self) -> tuple:
        """Return a hashable key identifying this
        transform's configuration."""
        ...

    def __hash__(self) -> int:
        return hash(self._key())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Transform):
            return self._key() == other._key()
        return False


class MultiInputTransform(ABC):
    """Base class for multi-input transforms.

    Multi-input transforms consume multiple materialized
    ``Sample`` instances and produce a single merged
    ``Sample``.  The ``inflation`` attribute is < 1.0
    (e.g. 0.25 for Mosaic, 0.5 for MixUp).

    Subclasses must implement :meth:`apply` and
    :meth:`_key`.
    """

    inflation: float

    @abstractmethod
    def apply(
        self,
        samples: List[Sample],
        rng: Optional[np.random.Generator] = None,
    ) -> Sample:
        """Apply the multi-input transform.

        Args:
            samples: List of fully materialized samples.
            rng: Random number generator.

        Returns:
            A single merged Sample.
        """
        ...

    @abstractmethod
    def _key(self) -> tuple:
        """Return a hashable key identifying this
        transform's configuration."""
        ...

    def __hash__(self) -> int:
        return hash(self._key())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MultiInputTransform):
            return self._key() == other._key()
        return False
