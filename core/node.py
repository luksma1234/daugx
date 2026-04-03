"""User-facing node handle for pipeline construction.

A ``Node`` is a thin handle returned by ``Pipeline.input()``,
``Node.then()``, ``Node.split()``, and ``Pipeline.merge()``.  It
records the user's intent and delegates all structural bookkeeping
to the owning ``Pipeline`` instance.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from daugx.core.pipeline import Pipeline


class Node:
    """Handle to a single node in the pipeline DAG.

    Users never instantiate ``Node`` directly.  Instances are created
    by ``Pipeline`` builder methods.

    Args:
        pipeline: The owning pipeline.
        node_id: Internal unique identifier for this node.
    """

    def __init__(self, pipeline: Pipeline, node_id: str) -> None:
        self._pipeline = pipeline
        self._node_id = node_id

    @property
    def node_id(self) -> str:
        return self._node_id

    def then(self, transform: object, p: float = 1.0) -> Node:
        """Append an augmentation after this node.

        Args:
            transform: A transform instance (``SITransform``,
                ``IOTransform``, or any object with an ``inflation``
                attribute).
            p: Internal execution probability in ``[0, 1]``.  When
                this node is reached during execution, the transform
                fires with probability *p*.

        Returns:
            A new ``Node`` representing the output of *transform*.

        Raises:
            ValueError: If *p* is not in ``[0, 1]``.
            RuntimeError: If the pipeline has already been compiled.
        """
        self._pipeline.ensure_mutable()
        if p < 0 or p > 1:
            raise ValueError(
                f"Execution probability must be in [0, 1], got {p}."
            )
        return self._pipeline._add_transform(self, transform, p)

    def split(self, *shares: float) -> Tuple[Node, ...]:
        """Branch into multiple paths with given probability shares.

        Shares are normalised so they sum to 1.0.

        Args:
            *shares: Positive probability weights, one per branch.

        Returns:
            A tuple of ``Node`` handles, one for each branch.

        Raises:
            ValueError: If fewer than two shares are given or any
                share is negative.
            RuntimeError: If the pipeline has already been compiled.
        """
        self._pipeline.ensure_mutable()
        if len(shares) < 2:
            raise ValueError("split() requires at least two shares.")
        if any(s < 0 for s in shares):
            raise ValueError("Shares must be non-negative.")
        return self._pipeline._add_split(self, shares)
