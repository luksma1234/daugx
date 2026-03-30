"""Annotation base class — base for all annotation components."""
from abc import abstractmethod
from typing import Optional

from daugx.core.data.component import Component, ComponentState


class Annotation(Component):
    """Abstract base for all annotation components.

    Annotations are first-class ``Component`` subclasses that
    live flat inside a ``Sample`` alongside the data they
    annotate.  They use the ``target`` field to associate with
    a parent component by its ``name``.

    All concrete annotation subclasses are lightweight and born
    ``MATERIALIZED``.  Heavy annotations (e.g. segmentation
    masks stored on disk) must override ``state``,
    ``is_materialized``, and ``materialize()`` as needed.

    Args:
        target: The ``name`` of the parent component this
            annotation belongs to (e.g. ``"cam_left"``).
            ``None`` means the annotation is unassociated.
        name: Optional disambiguation name for this annotation
            instance.
    """

    def __init__(
        self,
        *,
        target: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._component_name = name
        self._target = target

    @property
    def target(self) -> Optional[str]:
        """Name of the parent component this annotation
        belongs to, or ``None`` if unassociated."""
        return self._target

    @property
    def state(self) -> ComponentState:
        """Lightweight annotations are always MATERIALIZED."""
        return ComponentState.MATERIALIZED

    @property
    def is_materialized(self) -> bool:
        """Always ``True`` for lightweight annotations."""
        return True

    def materialize(self) -> None:
        """No-op — lightweight annotations are born
        materialized."""
