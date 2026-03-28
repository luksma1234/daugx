"""Pipeline builder for augmentation DAGs.

``Pipeline`` is the main entry point for constructing an augmentation
workflow.  Users register datasets, chain transforms, and split /
merge paths.  Calling ``compile()`` freezes the DAG and returns a
``CompiledPipeline`` that can be called to produce augmented samples.
"""
from __future__ import annotations

import secrets
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from daugx.core.compiled_pipeline import CompiledPipeline
from daugx.core.dataset import Dataset
from daugx.core.node import Node


class _NodeRecord:
    """Internal bookkeeping for a single DAG node."""

    def __init__(
        self,
        node_id: str,
        kind: str,
        *,
        dataset: Optional[Dataset] = None,
        transform: Optional[object] = None,
        exe_prob: float = 1.0,
        prev: Optional[List[str]] = None,
        shares: Optional[List[float]] = None,
    ) -> None:
        self.node_id = node_id
        self.kind = kind  # "input", "transform", "split", "merge"
        self.dataset = dataset
        self.transform = transform
        self.exe_prob = exe_prob
        self.prev: List[str] = prev or []
        self.next: List[str] = []
        self.shares = shares
        # Computed by Pipeline._calc_ext_exe_probs() during compile().
        self.ext_exe_prob: float = 1.0


class Pipeline:
    """Builder for an augmentation DAG.

    Args:
        seed: Explicit seed for reproducibility.  If ``None``, a
            random seed is generated.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        if seed is None:
            seed = secrets.randbelow(2**32)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._nodes: Dict[str, _NodeRecord] = {}
        self._compiled = False

    @property
    def seed(self) -> int:
        return self._seed

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """Render the pipeline DAG as an ASCII tree.

        Merge nodes break the tree structure (multiple parents), so
        they are rendered as separate sections.  Nodes that feed
        into a merge show ``-> MergeName`` instead of ``[Output]``.
        The merge is then rendered below with its input sources.
        """
        lines: List[str] = []
        rendered: set = set()
        # Build a set of nodes that feed directly into a merge.
        self._merge_targets: Dict[str, str] = {}
        for nid, rec in self._nodes.items():
            if rec.kind == "merge":
                for pid in rec.prev:
                    self._merge_targets[pid] = nid
        # Collect inputs that feed *directly* into a merge
        # with no augmentations in between.  These are rendered
        # compactly as part of the merge section, not standalone.
        direct_merge_inputs: set = set()
        for nid, rec in self._nodes.items():
            if rec.kind == "merge":
                for pid in rec.prev:
                    if self._nodes[pid].kind == "input":
                        direct_merge_inputs.add(pid)
        # Render from each input node.
        input_ids = [
            nid for nid, r in self._nodes.items()
            if r.kind == "input"
            and nid not in direct_merge_inputs
        ]
        for nid in input_ids:
            if nid not in rendered:
                self._render_node(nid, lines, "", True, rendered)
        # Render merge nodes that haven't been reached yet.
        for nid, rec in self._nodes.items():
            if rec.kind == "merge" and nid not in rendered:
                self._render_merge_section(
                    nid, lines, rendered,
                )
        return "\n".join(lines)

    def _render_merge_section(
        self,
        node_id: str,
        lines: List[str],
        rendered: set,
    ) -> None:
        """Render a merge node as a top-level section."""
        rec = self._nodes[node_id]
        rendered.add(node_id)
        label = self._node_label(rec)
        # Collect short source labels for the header.
        source_labels: List[str] = []
        for pid in rec.prev:
            source_labels.append(self._source_label(pid))
            rendered.add(pid)
        sources = ", ".join(source_labels)
        lines.append("")
        lines.append(f"{label} \u2190 [{sources}]")
        self._render_children(node_id, lines, "", rendered)

    def _source_label(self, node_id: str) -> str:
        """Short label for a merge input source."""
        rec = self._nodes[node_id]
        if rec.kind == "input":
            return rec.dataset.name if rec.dataset else "?"
        # Use this node's own label (e.g. "Shift (p=0.5)")
        return self._node_label(rec)

    def _render_node(
        self,
        node_id: str,
        lines: List[str],
        prefix: str,
        is_last: bool,
        rendered: set,
    ) -> None:
        """Recursively render a node and its descendants."""
        if node_id in rendered:
            return
        rendered.add(node_id)

        rec = self._nodes[node_id]
        connector = (
            "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
        )
        label = self._node_label(rec)

        if rec.kind == "input" and not prefix:
            lines.append(label)
            child_prefix = ""
        else:
            lines.append(f"{prefix}{connector}{label}")
            child_prefix = prefix + (
                "    " if is_last else "\u2502   "
            )

        self._render_children(
            node_id, lines, child_prefix, rendered,
        )

    def _render_children(
        self,
        node_id: str,
        lines: List[str],
        child_prefix: str,
        rendered: set,
    ) -> None:
        """Render the children of a node."""
        rec = self._nodes[node_id]
        children = rec.next

        # Check if this node feeds into a merge.
        if node_id in self._merge_targets:
            merge_id = self._merge_targets[node_id]
            merge_name = self._node_label(
                self._nodes[merge_id]
            )
            lines.append(
                f"{child_prefix}"
                f"\u2514\u2500\u2500 \u2192 {merge_name}"
            )
            return

        if not children:
            ext = ""
            if self._compiled:
                ext = f"  ext={rec.ext_exe_prob:.4g}"
            lines.append(
                f"{child_prefix}\u2514\u2500\u2500 [Output]{ext}"
            )
            return

        # Check if all children are split nodes.
        split_children = [
            cid for cid in children
            if self._nodes[cid].kind == "split"
        ]
        if split_children and len(split_children) == len(children):
            shares_str = ", ".join(
                f"{self._nodes[c].shares[0]:.4g}"
                for c in split_children
            )
            lines.append(
                f"{child_prefix}"
                f"\u2514\u2500\u2500 split({shares_str})"
            )
            split_prefix = child_prefix + "    "
            for i, cid in enumerate(split_children):
                is_last_branch = i == len(split_children) - 1
                share = self._nodes[cid].shares[0]
                branch_conn = (
                    "\u2514\u2500\u2500 " if is_last_branch
                    else "\u251c\u2500\u2500 "
                )
                lines.append(
                    f"{split_prefix}{branch_conn}"
                    f"Branch {i + 1} (share={share:.4g})"
                )
                branch_prefix = split_prefix + (
                    "    " if is_last_branch
                    else "\u2502   "
                )
                split_rec = self._nodes[cid]
                for j, gc in enumerate(split_rec.next):
                    # Check if grandchild is a merge node.
                    gc_rec = self._nodes[gc]
                    if gc_rec.kind == "merge":
                        self._render_merge(
                            gc, lines, branch_prefix, rendered,
                        )
                    else:
                        self._render_node(
                            gc, lines, branch_prefix,
                            j == len(split_rec.next) - 1,
                            rendered,
                        )
        else:
            for i, cid in enumerate(children):
                child_rec = self._nodes[cid]
                if child_rec.kind == "merge":
                    self._render_merge(
                        cid, lines, child_prefix, rendered,
                    )
                else:
                    self._render_node(
                        cid, lines, child_prefix,
                        i == len(children) - 1, rendered,
                    )

    def _node_label(self, rec: _NodeRecord) -> str:
        """Build the display label for a node."""
        if rec.kind == "input":
            name = rec.dataset.name if rec.dataset else "Dataset"
            return name
        if rec.kind in ("transform", "merge"):
            name = type(rec.transform).__name__
            if rec.exe_prob < 1.0:
                return f"{name} (p={rec.exe_prob:.4g})"
            return name
        if rec.kind == "split":
            return f"split (share={rec.shares[0]:.4g})"
        return rec.kind

    # ------------------------------------------------------------------
    # Mutability guard
    # ------------------------------------------------------------------

    def ensure_mutable(self) -> None:
        """Raise ``RuntimeError`` if the pipeline is already compiled."""
        if self._compiled:
            raise RuntimeError(
                "Pipeline has been compiled and cannot be modified."
            )

    # ------------------------------------------------------------------
    # Builder methods
    # ------------------------------------------------------------------

    def input(self, dataset: Dataset) -> Node:
        """Register a dataset as a source node.

        Args:
            dataset: A ``Dataset`` descriptor.

        Returns:
            A ``Node`` handle for the new input.
        """
        self.ensure_mutable()
        node_id = self._new_id()
        self._nodes[node_id] = _NodeRecord(
            node_id, "input", dataset=dataset,
        )
        return Node(self, node_id)

    def merge(
        self,
        inputs: Sequence[Node],
        transform: object,
    ) -> Node:
        """Combine multiple nodes with a multi-input transform.

        The number of *inputs* must match the transform's expected
        input count, derived from ``transform.inflation``.

        Args:
            inputs: Sequence of upstream ``Node`` handles.
            transform: A multi-image transform with
                ``inflation < 1``.

        Returns:
            A ``Node`` handle for the merged output.

        Raises:
            ValueError: If the input count does not match.
            RuntimeError: If the pipeline has been compiled.
        """
        self.ensure_mutable()
        expected = round(1 / transform.inflation)
        if len(inputs) != expected:
            raise ValueError(
                f"Transform expects {expected} inputs "
                f"(inflation={transform.inflation}), "
                f"got {len(inputs)}."
            )
        node_id = self._new_id()
        prev_ids = [n.node_id for n in inputs]
        self._nodes[node_id] = _NodeRecord(
            node_id, "merge",
            transform=transform,
            prev=prev_ids,
        )
        for pid in prev_ids:
            self._nodes[pid].next.append(node_id)
        return Node(self, node_id)

    def compile(self) -> CompiledPipeline:
        """Freeze the DAG and return a callable pipeline.

        Computes external execution probabilities for every node
        before freezing.

        Raises:
            ValueError: If the pipeline has no input nodes.
            RuntimeError: If already compiled.
        """
        self.ensure_mutable()
        if not any(
            r.kind == "input" for r in self._nodes.values()
        ):
            raise ValueError("Pipeline has no input nodes.")
        self._calc_ext_exe_probs()
        self._compiled = True
        return CompiledPipeline(
            seed=self._seed,
            nodes=self._nodes,
        )

    # ------------------------------------------------------------------
    # Probability queries (available after compile)
    # ------------------------------------------------------------------

    def get_ext_exe_prob(self, node_id: str) -> float:
        """Return the external execution probability for a node.

        The ext_exe_prob is the probability that this node's path
        is *selected* during execution.  It is the product of all
        branch shares from the input down to this node.
        """
        return self._nodes[node_id].ext_exe_prob

    def get_int_exe_prob(self, node_id: str) -> float:
        """Return the internal execution probability for a node.

        This is the ``p`` value passed to ``Node.then()``.  When
        the path is selected and data reaches this node, the
        transform fires with this probability.
        """
        return self._nodes[node_id].exe_prob

    def get_effective_prob(self, node_id: str) -> float:
        """Return the effective probability for a node.

        effective_prob = ext_exe_prob * int_exe_prob.  This is the
        overall probability that this specific augmentation actually
        runs on any given ``fetch()`` call from this branch.
        """
        rec = self._nodes[node_id]
        return rec.ext_exe_prob * rec.exe_prob

    # ------------------------------------------------------------------
    # Internal helpers called by Node
    # ------------------------------------------------------------------

    def add_transform(
        self, source: Node, transform: object, p: float,
    ) -> Node:
        """Add a transform node after *source*. Called by ``Node.then``."""
        node_id = self._new_id()
        self._nodes[node_id] = _NodeRecord(
            node_id, "transform",
            transform=transform,
            exe_prob=p,
            prev=[source.node_id],
        )
        self._nodes[source.node_id].next.append(node_id)
        return Node(self, node_id)

    def add_split(
        self, source: Node, shares: Sequence[float],
    ) -> Tuple[Node, ...]:
        """Add split branches after *source*. Called by ``Node.split``."""
        total = sum(shares)
        normalised = [s / total for s in shares]
        branch_nodes: List[Node] = []
        for share in normalised:
            node_id = self._new_id()
            self._nodes[node_id] = _NodeRecord(
                node_id, "split",
                prev=[source.node_id],
                shares=[share],
            )
            self._nodes[source.node_id].next.append(node_id)
            branch_nodes.append(Node(self, node_id))
        return tuple(branch_nodes)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _calc_ext_exe_probs(self) -> None:
        """Compute ext_exe_prob for every node.

        Walks forward from input nodes (ext_exe_prob=1.0).  Each
        successor inherits the parent's ext_exe_prob, multiplied by
        the branch share if the successor is a split node.  Merge
        nodes receive the minimum of their predecessors' probs
        (all predecessors must be reached for the merge to fire).

        Reuses the propagation logic from the original
        ``Blocks._calc_ext_exe_probs`` in ``block.py``.
        """
        # Reset all to 1.0.
        for rec in self._nodes.values():
            rec.ext_exe_prob = 1.0

        # Topological walk: process nodes whose predecessors are
        # all resolved.
        resolved: Dict[str, float] = {}
        # Seed: input nodes have ext_exe_prob = 1.0.
        queue: List[str] = [
            nid for nid, rec in self._nodes.items()
            if rec.kind == "input"
        ]
        for nid in queue:
            resolved[nid] = 1.0

        while queue:
            nid = queue.pop(0)
            rec = self._nodes[nid]
            parent_prob = resolved[nid]

            for child_id in rec.next:
                child = self._nodes[child_id]
                if child.kind == "split":
                    # Split node: inherit parent prob * share.
                    child_prob = parent_prob * child.shares[0]
                elif child.kind == "merge":
                    # Merge node: all predecessors must be reached.
                    # Use min of predecessors (they all must fire).
                    child_prob = min(
                        resolved.get(pid, 1.0)
                        for pid in child.prev
                    )
                    # Only resolve when all predecessors are done.
                    if not all(
                        pid in resolved for pid in child.prev
                    ):
                        continue
                else:
                    # Transform node: inherits parent prob directly.
                    child_prob = parent_prob

                child.ext_exe_prob = child_prob
                if child_id not in resolved:
                    resolved[child_id] = child_prob
                    queue.append(child_id)

    def _new_id(self) -> str:
        return f"node_{len(self._nodes)}"
