"""Compiled pipeline — the callable exit point.

A ``CompiledPipeline`` is produced by ``Pipeline.compile()``.  It
encapsulates the frozen DAG and RNG state.  Each call loads data
from the registered datasets, executes the selected augmentation
path, and returns the result as a dictionary of modalities.
"""
from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional, Set

import numpy as np


class CompiledPipeline:
    """Callable augmentation pipeline.

    Args:
        seed: The seed used to initialise the RNG.
        nodes: The frozen node graph from ``Pipeline``.
    """

    def __init__(
        self,
        seed: int,
        nodes: Dict[str, Any],
    ) -> None:
        self._seed = seed
        self._nodes = nodes
        self._rng = np.random.default_rng(seed)
        self._output_ids = self._find_output_ids()

    @property
    def seed(self) -> int:
        return self._seed

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the RNG to reproduce the same sequence.

        Args:
            seed: New seed.  If ``None``, reuses the original seed.
        """
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)

    def __call__(self) -> Dict[str, Any]:
        """Produce one augmented sample.

        Returns:
            A dictionary mapping modality names to their data
            (e.g. ``{'image': np.ndarray, 'bboxes': [...]}``).

        Raises:
            NotImplementedError: Execution engine is not yet built.
        """
        raise NotImplementedError(
            "Pipeline execution is not yet implemented."
        )

    def stream(
        self, n: int,
    ) -> Generator[Dict[str, Any], None, None]:
        """Yield *n* augmented samples.

        Args:
            n: Number of samples to generate.

        Yields:
            Dictionaries in the same format as ``__call__``.
        """
        for _ in range(n):
            yield self()

    # ------------------------------------------------------------------
    # Path tracing
    # ------------------------------------------------------------------

    def trace(self) -> Dict[str, Any]:
        """Simulate one path selection without loading data.

        Selects an output node weighted by ``ext_exe_prob``, walks
        backward to find required inputs, then walks forward
        deciding at each transform node whether it fires (using
        ``int_exe_prob``).

        Returns:
            A dict with keys:

            - ``output``: node_id of the selected output node.
            - ``inputs``: set of input node_ids that would be loaded.
            - ``visited``: set of all node_ids on the selected path.
            - ``fired``: set of transform node_ids that fired
              (not bypassed).
        """
        output_id = self._select_output()
        visited: Set[str] = set()
        inputs: Set[str] = set()
        fired: Set[str] = set()
        self._walk_backward(output_id, visited, inputs)
        self._walk_forward(inputs, visited, fired)
        return {
            "output": output_id,
            "inputs": inputs,
            "visited": visited,
            "fired": fired,
        }

    def _select_output(self) -> str:
        """Choose one output node weighted by ext_exe_prob."""
        if len(self._output_ids) == 1:
            return self._output_ids[0]
        probs = np.array([
            self._nodes[oid].ext_exe_prob
            for oid in self._output_ids
        ])
        probs = probs / probs.sum()
        idx = self._rng.choice(len(self._output_ids), p=probs)
        return self._output_ids[idx]

    def _walk_backward(
        self,
        node_id: str,
        visited: Set[str],
        inputs: Set[str],
    ) -> None:
        """Walk from *node_id* back to all required inputs."""
        visited.add(node_id)
        rec = self._nodes[node_id]
        if rec.kind == "input":
            inputs.add(node_id)
            return
        for pid in rec.prev:
            if pid not in visited:
                self._walk_backward(pid, visited, inputs)

    def _walk_forward(
        self,
        inputs: Set[str],
        visited: Set[str],
        fired: Set[str],
    ) -> None:
        """Walk forward from inputs, deciding int_exe_prob."""
        # Process nodes in topological order by following next
        # pointers, restricted to the visited set.
        queue: List[str] = list(inputs)
        processed: Set[str] = set()
        while queue:
            nid = queue.pop(0)
            if nid in processed:
                continue
            processed.add(nid)
            rec = self._nodes[nid]
            # Decide if this transform fires.
            if rec.kind in ("transform", "merge"):
                if self._rng.random() < rec.exe_prob:
                    fired.add(nid)
            for child_id in rec.next:
                if child_id in visited and child_id not in processed:
                    queue.append(child_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_output_ids(self) -> List[str]:
        """Find all terminal nodes (no successors, not a merge input).

        A node is an output if it has no children AND it is not
        consumed by a merge node.
        """
        merge_inputs: Set[str] = set()
        for rec in self._nodes.values():
            if rec.kind == "merge":
                merge_inputs.update(rec.prev)
        return [
            nid for nid, rec in self._nodes.items()
            if not rec.next and nid not in merge_inputs
        ]
