"""Transform base classes for the DataPackage API."""
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from daugx.core.data.data_package import DataPackage


class Transform(ABC):
    """Base class for single-input transforms.

    All transforms accept a ``DataPackage`` and return a
    new ``DataPackage``.  The ``inflation`` attribute
    controls pipeline sample-count semantics (always 1.0
    for single-input transforms).

    Subclasses must implement :meth:`apply` and
    :meth:`_key`.
    """

    inflation: float = 1.0

    @abstractmethod
    def apply(
        self,
        package: DataPackage,
        rng: Optional[np.random.Generator] = None,
    ) -> DataPackage:
        """Apply the transform to a data package.

        Args:
            package: Fully materialized data package.
            rng: Random number generator for
                reproducibility.

        Returns:
            A new DataPackage with the transform applied.
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

    Multi-input transforms consume multiple
    ``DataPackage`` instances and produce a single
    merged ``DataPackage``.  The ``inflation`` attribute
    is < 1.0 (e.g. 0.25 for Mosaic, 0.5 for MixUp).

    Subclasses must implement :meth:`apply` and
    :meth:`_key`.
    """

    inflation: float

    @abstractmethod
    def apply(
        self,
        packages: List[DataPackage],
        rng: Optional[np.random.Generator] = None,
    ) -> DataPackage:
        """Apply the multi-input transform.

        Args:
            packages: List of fully materialized packages.
            rng: Random number generator.

        Returns:
            A single merged DataPackage.
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
