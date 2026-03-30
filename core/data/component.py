"""Base component class and lifecycle enum."""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional


class ComponentState(Enum):
    """Lifecycle state of a component."""

    PRELOADED = 1
    MATERIALIZED = 2


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

    All components carry an optional ``name`` for
    disambiguation when multiple instances of the same
    type exist (e.g. ``Image(path, name="left")``), or to
    group components if components are ambiguous inside a
    sample.
    """

    def __init__(self) -> None:
        self._component_name: Optional[str] = None
        self._component_state: Optional[ComponentState] = None

    @property
    def name(self) -> Optional[str]:
        """Optional disambiguation name."""
        return self._component_name

    @property
    @abstractmethod
    def state(self) -> ComponentState:
        """Current lifecycle state."""
        ...

    @property
    @abstractmethod
    def is_materialized(self) -> bool:
        """True when the component is fully loaded."""
        ...

    @abstractmethod
    def materialize(self) -> None:
        """Load heavy data into memory (in-place).

        No-op if already materialized or if the component
        is inherently lightweight.
        """
        ...
