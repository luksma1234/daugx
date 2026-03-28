"""Base component class and lifecycle enum."""
from abc import ABC, abstractmethod
from enum import Enum, auto


class ComponentState(Enum):
    """Lifecycle state of a component."""

    PRELOADED = auto()
    MATERIALIZED = auto()


class Component(ABC):
    """Base class for all data components.

    Components have two lifecycle states:

    - ``PRELOADED``: lightweight metadata only (paths,
      coordinates, class ids).  No heavy I/O performed.
    - ``MATERIALIZED``: heavy data fully loaded into
      memory (pixel arrays, waveforms, etc.).

    Lightweight components (Label, BoundingBox, Polygon,
    KeyPoint) are born MATERIALIZED since they carry no
    heavy data.
    """

    @property
    @abstractmethod
    def state(self) -> ComponentState:
        """Current lifecycle state."""
        ...

    @property
    @abstractmethod
    def is_materialized(self) -> bool:
        """Whether heavy data is loaded."""
        ...

    @abstractmethod
    def materialize(self) -> None:
        """Load heavy data into memory (in-place).

        No-op if already materialized or if the component
        is inherently lightweight.
        """
        ...
