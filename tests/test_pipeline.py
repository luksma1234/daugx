"""Tests for the pipeline API surface.

Tests use mocks for transforms and datasets to isolate pipeline
construction logic from data loading and augmentation execution.
"""
from collections import Counter
from unittest.mock import MagicMock

import pytest

from daugx.core.dataset import Dataset
from daugx.core.node import Node
from daugx.core.pipeline import Pipeline
from daugx.core.compiled_pipeline import CompiledPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_si_transform(name: str = "Transform") -> MagicMock:
    """Creates a mock single-image transform (inflation=1)."""
    t = MagicMock()
    t.inflation = 1
    type(t).__name__ = name
    return t


def _make_mi_transform(
    inflation: float, name: str = "MITransform",
) -> MagicMock:
    """Creates a mock multi-image transform with given inflation."""
    t = MagicMock()
    t.inflation = inflation
    type(t).__name__ = name
    return t


def _make_dataset(name: str = "Dataset") -> Dataset:
    return Dataset(samples=[], name=name)


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

class TestPipelineConstruction:
    """Tests for Pipeline builder methods."""

    def test_creates_with_explicit_seed(self):
        p = Pipeline(seed=42)
        assert p.seed == 42

    def test_creates_with_random_seed_when_omitted(self):
        p = Pipeline()
        assert isinstance(p.seed, int)

    def test_two_unseeded_pipelines_get_different_seeds(self):
        p1 = Pipeline()
        p2 = Pipeline()
        assert p1.seed != p2.seed

    def test_input_returns_node(self):
        p = Pipeline(seed=0)
        ds = _make_dataset()
        node = p.input(ds)
        assert isinstance(node, Node)

    def test_multiple_inputs_return_distinct_nodes(self):
        p = Pipeline(seed=0)
        ds = _make_dataset()
        n1 = p.input(ds)
        n2 = p.input(ds)
        assert n1 is not n2


# ---------------------------------------------------------------------------
# Node chaining
# ---------------------------------------------------------------------------

class TestNodeChaining:
    """Tests for Node.then() and Node.split()."""

    def test_then_returns_new_node(self):
        p = Pipeline(seed=0)
        n1 = p.input(_make_dataset())
        n2 = n1.then(_make_si_transform())
        assert isinstance(n2, Node)
        assert n2 is not n1

    def test_then_accepts_probability(self):
        p = Pipeline(seed=0)
        n1 = p.input(_make_dataset())
        n2 = n1.then(_make_si_transform(), p=0.5)
        assert isinstance(n2, Node)

    def test_then_rejects_probability_below_zero(self):
        p = Pipeline(seed=0)
        n1 = p.input(_make_dataset())
        with pytest.raises(ValueError):
            n1.then(_make_si_transform(), p=-0.1)

    def test_then_rejects_probability_above_one(self):
        p = Pipeline(seed=0)
        n1 = p.input(_make_dataset())
        with pytest.raises(ValueError):
            n1.then(_make_si_transform(), p=1.5)

    def test_split_returns_correct_count(self):
        p = Pipeline(seed=0)
        branches = p.input(_make_dataset()).split(0.3, 0.7)
        assert len(branches) == 2

    def test_split_returns_nodes(self):
        p = Pipeline(seed=0)
        branches = p.input(_make_dataset()).split(0.5, 0.3, 0.2)
        assert all(isinstance(b, Node) for b in branches)

    def test_split_requires_at_least_two_branches(self):
        p = Pipeline(seed=0)
        n = p.input(_make_dataset())
        with pytest.raises(ValueError):
            n.split(1.0)

    def test_split_rejects_negative_shares(self):
        p = Pipeline(seed=0)
        n = p.input(_make_dataset())
        with pytest.raises(ValueError):
            n.split(0.5, -0.5)

    def test_split_normalizes_shares(self):
        """Shares that don't sum to 1.0 are normalized internally."""
        p = Pipeline(seed=0)
        n = p.input(_make_dataset())
        # Should not raise — shares are normalized
        branches = n.split(1, 1, 1)
        assert len(branches) == 3

    def test_chaining_after_split(self):
        """Each branch from split can be independently chained."""
        p = Pipeline(seed=0)
        a, b = p.input(_make_dataset()).split(0.5, 0.5)
        a2 = a.then(_make_si_transform())
        b2 = b.then(_make_si_transform())
        assert isinstance(a2, Node)
        assert isinstance(b2, Node)
        assert a2 is not b2


# ---------------------------------------------------------------------------
# Merge (multi-input augmentations)
# ---------------------------------------------------------------------------

class TestMerge:
    """Tests for Pipeline.merge()."""

    def test_merge_returns_node(self):
        p = Pipeline(seed=0)
        ds = _make_dataset()
        inputs = [p.input(ds) for _ in range(4)]
        mosaic = _make_mi_transform(inflation=0.25)
        out = p.merge(inputs, mosaic)
        assert isinstance(out, Node)

    def test_merge_validates_input_count_too_few(self):
        p = Pipeline(seed=0)
        ds = _make_dataset()
        inputs = [p.input(ds) for _ in range(2)]
        mosaic = _make_mi_transform(inflation=0.25)  # expects 4
        with pytest.raises(ValueError):
            p.merge(inputs, mosaic)

    def test_merge_validates_input_count_too_many(self):
        p = Pipeline(seed=0)
        ds = _make_dataset()
        inputs = [p.input(ds) for _ in range(3)]
        mixup = _make_mi_transform(inflation=0.5)  # expects 2
        with pytest.raises(ValueError):
            p.merge(inputs, mixup)

    def test_merge_accepts_correct_count(self):
        p = Pipeline(seed=0)
        ds = _make_dataset()
        inputs = [p.input(ds) for _ in range(2)]
        mixup = _make_mi_transform(inflation=0.5)
        out = p.merge(inputs, mixup)
        assert isinstance(out, Node)

    def test_merge_output_can_be_chained(self):
        p = Pipeline(seed=0)
        ds = _make_dataset()
        inputs = [p.input(ds) for _ in range(2)]
        mixup = _make_mi_transform(inflation=0.5)
        out = p.merge(inputs, mixup)
        chained = out.then(_make_si_transform())
        assert isinstance(chained, Node)


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------

class TestCompile:
    """Tests for Pipeline.compile()."""

    def test_compile_returns_compiled_pipeline(self):
        p = Pipeline(seed=0)
        p.input(_make_dataset())
        compiled = p.compile()
        assert isinstance(compiled, CompiledPipeline)

    def test_compiled_pipeline_is_callable(self):
        p = Pipeline(seed=0)
        p.input(_make_dataset())
        compiled = p.compile()
        assert callable(compiled)

    def test_compile_raises_on_empty_pipeline(self):
        p = Pipeline(seed=0)
        with pytest.raises(ValueError):
            p.compile()

    def test_compile_freezes_pipeline(self):
        """After compile, adding nodes should raise."""
        p = Pipeline(seed=0)
        n = p.input(_make_dataset())
        p.compile()
        with pytest.raises(RuntimeError):
            n.then(_make_si_transform())

    def test_compile_freezes_split(self):
        p = Pipeline(seed=0)
        n = p.input(_make_dataset())
        p.compile()
        with pytest.raises(RuntimeError):
            n.split(0.5, 0.5)

    def test_compile_freezes_input(self):
        p = Pipeline(seed=0)
        p.input(_make_dataset())
        p.compile()
        with pytest.raises(RuntimeError):
            p.input(_make_dataset())

    def test_compile_freezes_merge(self):
        p = Pipeline(seed=0)
        ds = _make_dataset()
        p.input(ds)
        p.compile()
        with pytest.raises(RuntimeError):
            p.merge(
                [p.input(ds), p.input(ds)],
                _make_mi_transform(0.5),
            )


# ---------------------------------------------------------------------------
# CompiledPipeline
# ---------------------------------------------------------------------------

class TestCompiledPipeline:
    """Tests for CompiledPipeline interface."""

    def test_seed_property(self):
        p = Pipeline(seed=99)
        p.input(_make_dataset())
        compiled = p.compile()
        assert compiled.seed == 99

    def test_reset_does_not_raise(self):
        p = Pipeline(seed=42)
        p.input(_make_dataset())
        compiled = p.compile()
        compiled.reset()

    def test_stream_returns_iterator(self):
        p = Pipeline(seed=0)
        p.input(_make_dataset())
        compiled = p.compile()
        stream = compiled.stream(n=5)
        assert hasattr(stream, '__iter__')
        assert hasattr(stream, '__next__')


# ---------------------------------------------------------------------------
# Complex pipeline construction (integration-style)
# ---------------------------------------------------------------------------

class TestExecutionProbability:
    """Tests for ext_exe_prob and int_exe_prob computation.

    ext_exe_prob (external execution probability):
        The probability that a given output node is *selected* as the
        execution path.  Determined by branch shares along the path.

    int_exe_prob (internal execution probability):
        Per-node coin flip.  When the path is selected and data reaches
        this node, the transform fires with probability ``p``.

    effective_prob:
        The overall probability that a specific augmentation actually
        runs = ext_exe_prob * int_exe_prob.
    """

    def test_single_input_has_ext_prob_one(self):
        """A lone input node has ext_exe_prob=1.0."""
        p = Pipeline(seed=0)
        inp = p.input(_make_dataset())
        p.compile()
        assert p.get_ext_exe_prob(inp.node_id) == pytest.approx(1.0)

    def test_split_shares_become_ext_probs(self):
        """Split shares propagate as ext_exe_prob to outputs."""
        p = Pipeline(seed=0)
        a, b = p.input(_make_dataset()).split(0.7, 0.3)
        out_a = a.then(_make_si_transform())
        out_b = b.then(_make_si_transform())
        p.compile()
        assert p.get_ext_exe_prob(out_a.node_id) == pytest.approx(0.7)
        assert p.get_ext_exe_prob(out_b.node_id) == pytest.approx(0.3)

    def test_chained_transforms_inherit_ext_prob(self):
        """Transforms chained after a split inherit the branch share."""
        p = Pipeline(seed=0)
        a, b = p.input(_make_dataset()).split(0.6, 0.4)
        a2 = a.then(_make_si_transform())
        a3 = a2.then(_make_si_transform())
        b.then(_make_si_transform())
        p.compile()
        # Both a2 and a3 are on the 0.6 branch
        assert p.get_ext_exe_prob(a2.node_id) == pytest.approx(0.6)
        assert p.get_ext_exe_prob(a3.node_id) == pytest.approx(0.6)

    def test_nested_splits_multiply(self):
        """Nested splits multiply their shares."""
        p = Pipeline(seed=0)
        a, b = p.input(_make_dataset()).split(0.8, 0.2)
        a1, a2 = a.split(0.5, 0.5)
        out_a1 = a1.then(_make_si_transform())
        out_a2 = a2.then(_make_si_transform())
        out_b = b.then(_make_si_transform())
        p.compile()
        # a1 path: 0.8 * 0.5 = 0.4
        assert p.get_ext_exe_prob(out_a1.node_id) == pytest.approx(0.4)
        # a2 path: 0.8 * 0.5 = 0.4
        assert p.get_ext_exe_prob(out_a2.node_id) == pytest.approx(0.4)
        # b path: 0.2
        assert p.get_ext_exe_prob(out_b.node_id) == pytest.approx(0.2)

    def test_int_exe_prob_stored_on_transform_nodes(self):
        """The p parameter on then() is stored as int_exe_prob."""
        p = Pipeline(seed=0)
        n = p.input(_make_dataset()).then(_make_si_transform(), p=0.4)
        p.compile()
        assert p.get_int_exe_prob(n.node_id) == pytest.approx(0.4)

    def test_effective_prob_combines_ext_and_int(self):
        """effective_prob = ext_exe_prob * int_exe_prob."""
        p = Pipeline(seed=0)
        a, b = p.input(_make_dataset()).split(0.7, 0.3)
        # Mosaic on the 0.7 branch with p=0.4
        out_a = a.then(_make_si_transform(), p=0.4)
        out_b = b.then(_make_si_transform())
        p.compile()
        # effective = 0.7 * 0.4 = 0.28
        assert p.get_effective_prob(out_a.node_id) == pytest.approx(
            0.28
        )
        # effective = 0.3 * 1.0 = 0.3
        assert p.get_effective_prob(out_b.node_id) == pytest.approx(
            0.3
        )

    def test_merge_ext_prob_sums_predecessors(self):
        """A merge node's ext_exe_prob is the sum of its inputs."""
        p = Pipeline(seed=0)
        ds = _make_dataset()
        # Two independent input nodes, each with ext_exe_prob=1.0
        a = p.input(ds)
        b = p.input(ds)
        mixup = _make_mi_transform(inflation=0.5)
        merged = p.merge([a, b], mixup)
        p.compile()
        # Merge receives data from both inputs (sum = 1.0 + 1.0 = 2.0)
        # but since it's a merge consuming both, ext_exe_prob = 1.0
        # (both paths always flow into it)
        assert p.get_ext_exe_prob(merged.node_id) == pytest.approx(
            1.0
        )

    def test_multiple_independent_outputs_ext_probs(self):
        """Two independent paths each have ext_exe_prob=1.0."""
        p = Pipeline(seed=0)
        ds = _make_dataset()
        out_a = p.input(ds).then(_make_si_transform())
        out_b = p.input(ds).then(_make_si_transform())
        p.compile()
        assert p.get_ext_exe_prob(out_a.node_id) == pytest.approx(
            1.0
        )
        assert p.get_ext_exe_prob(out_b.node_id) == pytest.approx(
            1.0
        )


# ---------------------------------------------------------------------------
# Complex pipeline construction (integration-style)
# ---------------------------------------------------------------------------

class TestComplexConstruction:
    """Tests for more complex pipeline DAG shapes."""

    def test_branching_pipeline(self):
        """Input -> split -> two augmentation chains."""
        p = Pipeline(seed=0)
        inp = p.input(_make_dataset())
        a, b = inp.split(0.6, 0.4)
        a.then(_make_si_transform())
        b.then(_make_si_transform()).then(_make_si_transform())
        compiled = p.compile()
        assert isinstance(compiled, CompiledPipeline)

    def test_multi_dataset_pipeline(self):
        """Two datasets feeding into separate chains."""
        p = Pipeline(seed=0)
        ds_a = _make_dataset()
        ds_b = _make_dataset()
        p.input(ds_a).then(_make_si_transform())
        p.input(ds_b).then(_make_si_transform())
        compiled = p.compile()
        assert isinstance(compiled, CompiledPipeline)

    def test_mosaic_pipeline(self):
        """Four inputs merged via mosaic then chained."""
        p = Pipeline(seed=0)
        ds = _make_dataset()
        inputs = [p.input(ds) for _ in range(4)]
        mosaic = _make_mi_transform(inflation=0.25)
        out = p.merge(inputs, mosaic)
        out.then(_make_si_transform())
        compiled = p.compile()
        assert isinstance(compiled, CompiledPipeline)

    def test_branch_then_merge(self):
        """Input splits, each branch augmented, then merged."""
        p = Pipeline(seed=0)
        ds = _make_dataset()
        a, b = p.input(ds).split(0.5, 0.5)
        a_out = a.then(_make_si_transform())
        b_out = b.then(_make_si_transform())
        mixup = _make_mi_transform(inflation=0.5)
        merged = p.merge([a_out, b_out], mixup)
        merged.then(_make_si_transform())
        compiled = p.compile()
        assert isinstance(compiled, CompiledPipeline)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

class TestVisualization:
    """Tests for Pipeline.__str__() tree rendering."""

    def test_simple_chain(self):
        p = Pipeline(seed=0)
        p.input(_make_dataset("COCO")).then(
            _make_si_transform("Resize")
        )
        p.compile()
        text = str(p)
        assert "COCO" in text
        assert "Resize" in text
        assert "[Output]" in text

    def test_split_shows_branches_and_shares(self):
        p = Pipeline(seed=0)
        a, b = p.input(_make_dataset("VOC")).split(0.7, 0.3)
        a.then(_make_si_transform("Resize"))
        b.then(_make_si_transform("Rotate"))
        p.compile()
        text = str(p)
        assert "0.7" in text
        assert "0.3" in text
        assert "Resize" in text
        assert "Rotate" in text

    def test_shows_int_exe_prob_when_not_one(self):
        p = Pipeline(seed=0)
        p.input(_make_dataset("DS")).then(
            _make_si_transform("Mosaic"), p=0.4,
        )
        p.compile()
        text = str(p)
        assert "p=0.4" in text

    def test_hides_int_exe_prob_when_one(self):
        p = Pipeline(seed=0)
        p.input(_make_dataset("DS")).then(
            _make_si_transform("Resize"),
        )
        p.compile()
        text = str(p)
        assert "p=" not in text

    def test_shows_ext_exe_prob_on_outputs(self):
        p = Pipeline(seed=0)
        a, b = p.input(_make_dataset("DS")).split(0.6, 0.4)
        a.then(_make_si_transform("Resize"))
        b.then(_make_si_transform("Rotate"))
        p.compile()
        text = str(p)
        assert "ext=0.6" in text
        assert "ext=0.4" in text

    def test_merge_shown(self):
        p = Pipeline(seed=0)
        ds = _make_dataset("DS")
        inputs = [p.input(ds) for _ in range(2)]
        mixup = _make_mi_transform(0.5, "MixUp")
        out = p.merge(inputs, mixup)
        out.then(_make_si_transform("Resize"))
        p.compile()
        text = str(p)
        assert "MixUp" in text

    def test_works_before_compile(self):
        """Visualization works without ext_exe_prob (pre-compile)."""
        p = Pipeline(seed=0)
        p.input(_make_dataset("DS")).then(
            _make_si_transform("Resize"),
        )
        text = str(p)
        assert "DS" in text
        assert "Resize" in text


# ---------------------------------------------------------------------------
# Probabilistic path-selection tests
# ---------------------------------------------------------------------------

# Tolerance for statistical tests.  With N=5000 runs most
# proportions should land within ±0.04 of the expected value.
N_RUNS = 5000
ABS_TOL = 0.04


def _count_traces(compiled, n=N_RUNS):
    """Run ``trace()`` *n* times, return list of trace dicts."""
    return [compiled.trace() for _ in range(n)]


def _node_freq(traces, node_id):
    """Fraction of traces that visited *node_id*."""
    return sum(
        1 for t in traces if node_id in t["visited"]
    ) / len(traces)


def _fired_freq(traces, node_id):
    """Fraction of traces where *node_id* fired (not bypassed)."""
    return sum(
        1 for t in traces if node_id in t["fired"]
    ) / len(traces)


def _output_freq(traces, node_id):
    """Fraction of traces that ended at *node_id*."""
    return sum(
        1 for t in traces if t["output"] == node_id
    ) / len(traces)


def _input_freq(traces, node_id):
    """Fraction of traces that loaded from *node_id*."""
    return sum(
        1 for t in traces if node_id in t["inputs"]
    ) / len(traces)


class TestProbabilisticPathSelection:
    """Run trace() many times and verify frequencies match probs."""

    def test_single_path_always_selected(self):
        """A pipeline with one path selects it 100% of the time."""
        p = Pipeline(seed=7)
        out = p.input(_make_dataset()).then(_make_si_transform())
        compiled = p.compile()
        traces = _count_traces(compiled)
        assert _output_freq(traces, out.node_id) == pytest.approx(
            1.0, abs=0.001
        )

    def test_split_frequencies_match_shares(self):
        """split(0.7, 0.3) should route ~70% / ~30%."""
        p = Pipeline(seed=42)
        a, b = p.input(_make_dataset()).split(0.7, 0.3)
        out_a = a.then(_make_si_transform("A"))
        out_b = b.then(_make_si_transform("B"))
        compiled = p.compile()
        traces = _count_traces(compiled)
        assert _output_freq(traces, out_a.node_id) == pytest.approx(
            0.7, abs=ABS_TOL
        )
        assert _output_freq(traces, out_b.node_id) == pytest.approx(
            0.3, abs=ABS_TOL
        )

    def test_three_way_split(self):
        """split(0.5, 0.3, 0.2) should distribute accordingly."""
        p = Pipeline(seed=99)
        a, b, c = p.input(_make_dataset()).split(0.5, 0.3, 0.2)
        out_a = a.then(_make_si_transform("A"))
        out_b = b.then(_make_si_transform("B"))
        out_c = c.then(_make_si_transform("C"))
        compiled = p.compile()
        traces = _count_traces(compiled)
        assert _output_freq(traces, out_a.node_id) == pytest.approx(
            0.5, abs=ABS_TOL
        )
        assert _output_freq(traces, out_b.node_id) == pytest.approx(
            0.3, abs=ABS_TOL
        )
        assert _output_freq(traces, out_c.node_id) == pytest.approx(
            0.2, abs=ABS_TOL
        )

    def test_int_exe_prob_fires_at_expected_rate(self):
        """A node with p=0.4 should fire ~40% when visited."""
        p = Pipeline(seed=21)
        aug = p.input(_make_dataset()).then(
            _make_si_transform("Aug"), p=0.4,
        )
        compiled = p.compile()
        traces = _count_traces(compiled)
        # Node is always visited (single path), fires 40%.
        assert _node_freq(traces, aug.node_id) == pytest.approx(
            1.0, abs=0.001
        )
        assert _fired_freq(traces, aug.node_id) == pytest.approx(
            0.4, abs=ABS_TOL
        )

    def test_effective_prob_split_and_int(self):
        """Combine split share with int_exe_prob.

        Branch A (share=0.6) -> Aug (p=0.5)
        effective fire rate = 0.6 * 0.5 = 0.3
        """
        p = Pipeline(seed=55)
        a, b = p.input(_make_dataset()).split(0.6, 0.4)
        aug_a = a.then(_make_si_transform("AugA"), p=0.5)
        b.then(_make_si_transform("B"))
        compiled = p.compile()
        traces = _count_traces(compiled)
        # aug_a is visited ~60% of the time (branch share)
        assert _node_freq(traces, aug_a.node_id) == pytest.approx(
            0.6, abs=ABS_TOL
        )
        # Of those visits, it fires 50% -> overall ~30%
        assert _fired_freq(traces, aug_a.node_id) == pytest.approx(
            0.3, abs=ABS_TOL
        )

    def test_nested_split_probabilities(self):
        """Nested splits multiply: 0.8 * 0.5 = 0.4 for each."""
        p = Pipeline(seed=13)
        a, b = p.input(_make_dataset()).split(0.8, 0.2)
        a1, a2 = a.split(0.5, 0.5)
        out_a1 = a1.then(_make_si_transform("A1"))
        out_a2 = a2.then(_make_si_transform("A2"))
        out_b = b.then(_make_si_transform("B"))
        compiled = p.compile()
        traces = _count_traces(compiled)
        assert _output_freq(traces, out_a1.node_id) == pytest.approx(
            0.4, abs=ABS_TOL
        )
        assert _output_freq(traces, out_a2.node_id) == pytest.approx(
            0.4, abs=ABS_TOL
        )
        assert _output_freq(traces, out_b.node_id) == pytest.approx(
            0.2, abs=ABS_TOL
        )

    def test_multiple_independent_paths(self):
        """Two independent input paths are selected 50/50."""
        p = Pipeline(seed=77)
        out_a = p.input(
            _make_dataset("DS_A")
        ).then(_make_si_transform("A"))
        out_b = p.input(
            _make_dataset("DS_B")
        ).then(_make_si_transform("B"))
        compiled = p.compile()
        traces = _count_traces(compiled)
        # Two independent outputs, each ext_exe_prob=1.0
        # should be selected ~50/50.
        assert _output_freq(traces, out_a.node_id) == pytest.approx(
            0.5, abs=ABS_TOL
        )
        assert _output_freq(traces, out_b.node_id) == pytest.approx(
            0.5, abs=ABS_TOL
        )

    def test_input_dataset_selection(self):
        """Traces record which input nodes were used."""
        p = Pipeline(seed=33)
        ds_a = _make_dataset("A")
        ds_b = _make_dataset("B")
        inp_a = p.input(ds_a)
        inp_b = p.input(ds_b)
        inp_a.then(_make_si_transform())
        inp_b.then(_make_si_transform())
        compiled = p.compile()
        traces = _count_traces(compiled)
        assert _input_freq(traces, inp_a.node_id) == pytest.approx(
            0.5, abs=ABS_TOL
        )
        assert _input_freq(traces, inp_b.node_id) == pytest.approx(
            0.5, abs=ABS_TOL
        )

    def test_chain_of_int_exe_probs(self):
        """Two nodes in series: p=0.6 then p=0.5.

        Both are always visited (single path).
        First fires 60%, second fires 50%, independently.
        """
        p = Pipeline(seed=88)
        inp = p.input(_make_dataset())
        aug1 = inp.then(_make_si_transform("Aug1"), p=0.6)
        aug2 = aug1.then(_make_si_transform("Aug2"), p=0.5)
        compiled = p.compile()
        traces = _count_traces(compiled)
        assert _fired_freq(traces, aug1.node_id) == pytest.approx(
            0.6, abs=ABS_TOL
        )
        assert _fired_freq(traces, aug2.node_id) == pytest.approx(
            0.5, abs=ABS_TOL
        )

    def test_merge_inputs_all_visited(self):
        """All inputs feeding a merge are visited every time."""
        p = Pipeline(seed=44)
        ds = _make_dataset()
        inp_a = p.input(ds)
        inp_b = p.input(ds)
        mixup = _make_mi_transform(0.5, "MixUp")
        p.merge([inp_a, inp_b], mixup)
        compiled = p.compile()
        traces = _count_traces(compiled)
        assert _input_freq(traces, inp_a.node_id) == pytest.approx(
            1.0, abs=0.001
        )
        assert _input_freq(traces, inp_b.node_id) == pytest.approx(
            1.0, abs=0.001
        )

    def test_deterministic_with_same_seed(self):
        """Same seed produces identical trace sequences."""
        def build():
            p = Pipeline(seed=42)
            a, b = p.input(_make_dataset()).split(0.6, 0.4)
            a.then(_make_si_transform(), p=0.5)
            b.then(_make_si_transform())
            return p.compile()

        c1 = build()
        c2 = build()
        traces_1 = [c1.trace() for _ in range(100)]
        traces_2 = [c2.trace() for _ in range(100)]
        for t1, t2 in zip(traces_1, traces_2):
            assert t1["output"] == t2["output"]
            assert t1["fired"] == t2["fired"]
