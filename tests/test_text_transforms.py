"""Tests for text augmentation transforms."""
import numpy as np
import pytest

from daugx.core.data.data_package import DataPackage
from daugx.core.data.components.text import Text
from daugx.core.data.components.label import Label
from daugx.core.augmentation.base import Transform


SYNONYMS = {
    "good": ["great", "fine"],
    "bad": ["poor", "terrible"],
    "big": ["large", "huge"],
    "happy": ["glad", "joyful"],
}


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def text_package():
    """DataPackage with text and a label."""
    return DataPackage({
        "text": Text("the good big happy cat"),
        "label": Label(1, "positive"),
    })


@pytest.fixture
def no_synonym_package():
    """DataPackage with text that has no synonym matches."""
    return DataPackage({
        "text": Text("xyz uvw abc"),
    })


# -------------------------------------------------------------------
# SynonymReplace
# -------------------------------------------------------------------

class TestSynonymReplace:
    def test_is_transform(self):
        from daugx.core.augmentation.text.synonym_replace import (
            SynonymReplace,
        )
        assert issubclass(SynonymReplace, Transform)

    def test_replaces_words(self, text_package, rng):
        from daugx.core.augmentation.text.synonym_replace import (
            SynonymReplace,
        )
        t = SynonymReplace(n=2, synonyms=SYNONYMS)
        result = t.apply(text_package, rng)
        orig_words = text_package["text"].words
        new_words = result["text"].words
        assert len(new_words) == len(orig_words)
        changed = sum(
            a != b for a, b in zip(orig_words, new_words)
        )
        assert changed == 2

    def test_replaced_word_is_synonym(
        self, text_package, rng,
    ):
        from daugx.core.augmentation.text.synonym_replace import (
            SynonymReplace,
        )
        t = SynonymReplace(n=1, synonyms=SYNONYMS)
        result = t.apply(text_package, rng)
        new_words = result["text"].words
        all_syns = set()
        for v in SYNONYMS.values():
            all_syns.update(v)
        orig_words = text_package["text"].words
        for ow, nw in zip(orig_words, new_words):
            if ow != nw:
                assert nw in all_syns

    def test_no_match_returns_unchanged(
        self, no_synonym_package, rng,
    ):
        from daugx.core.augmentation.text.synonym_replace import (
            SynonymReplace,
        )
        t = SynonymReplace(n=1, synonyms=SYNONYMS)
        result = t.apply(no_synonym_package, rng)
        assert result["text"].text == "xyz uvw abc"

    def test_label_preserved(self, text_package, rng):
        from daugx.core.augmentation.text.synonym_replace import (
            SynonymReplace,
        )
        t = SynonymReplace(n=1, synonyms=SYNONYMS)
        result = t.apply(text_package, rng)
        assert result["label"].name == "positive"

    def test_seeded_reproducibility(self, text_package):
        from daugx.core.augmentation.text.synonym_replace import (
            SynonymReplace,
        )
        t = SynonymReplace(n=2, synonyms=SYNONYMS)
        r1 = t.apply(
            text_package, np.random.default_rng(7),
        )
        r2 = t.apply(
            text_package, np.random.default_rng(7),
        )
        assert r1["text"].text == r2["text"].text

    def test_returns_datapackage(self, text_package, rng):
        from daugx.core.augmentation.text.synonym_replace import (
            SynonymReplace,
        )
        t = SynonymReplace(n=1, synonyms=SYNONYMS)
        result = t.apply(text_package, rng)
        assert isinstance(result, DataPackage)


# -------------------------------------------------------------------
# RandomInsertion
# -------------------------------------------------------------------

class TestRandomInsertion:
    def test_is_transform(self):
        from daugx.core.augmentation.text.random_insertion import (
            RandomInsertion,
        )
        assert issubclass(RandomInsertion, Transform)

    def test_word_count_increases(
        self, text_package, rng,
    ):
        from daugx.core.augmentation.text.random_insertion import (
            RandomInsertion,
        )
        t = RandomInsertion(n=2, synonyms=SYNONYMS)
        result = t.apply(text_package, rng)
        orig_len = len(text_package["text"].words)
        new_len = len(result["text"].words)
        assert new_len == orig_len + 2

    def test_inserted_word_is_synonym(
        self, text_package, rng,
    ):
        from daugx.core.augmentation.text.random_insertion import (
            RandomInsertion,
        )
        all_syns = set()
        for v in SYNONYMS.values():
            all_syns.update(v)
        t = RandomInsertion(n=1, synonyms=SYNONYMS)
        result = t.apply(text_package, rng)
        orig_words = set(text_package["text"].words)
        new_words = result["text"].words
        inserted = [
            w for w in new_words if w not in orig_words
        ]
        assert len(inserted) >= 1
        assert inserted[0] in all_syns

    def test_no_match_returns_unchanged(
        self, no_synonym_package, rng,
    ):
        from daugx.core.augmentation.text.random_insertion import (
            RandomInsertion,
        )
        t = RandomInsertion(n=1, synonyms=SYNONYMS)
        result = t.apply(no_synonym_package, rng)
        assert result["text"].text == "xyz uvw abc"

    def test_seeded_reproducibility(self, text_package):
        from daugx.core.augmentation.text.random_insertion import (
            RandomInsertion,
        )
        t = RandomInsertion(n=2, synonyms=SYNONYMS)
        r1 = t.apply(
            text_package, np.random.default_rng(7),
        )
        r2 = t.apply(
            text_package, np.random.default_rng(7),
        )
        assert r1["text"].text == r2["text"].text


# -------------------------------------------------------------------
# RandomDeletion
# -------------------------------------------------------------------

class TestRandomDeletion:
    def test_is_transform(self):
        from daugx.core.augmentation.text.random_deletion import (
            RandomDeletion,
        )
        assert issubclass(RandomDeletion, Transform)

    def test_word_count_decreases(
        self, text_package, rng,
    ):
        from daugx.core.augmentation.text.random_deletion import (
            RandomDeletion,
        )
        t = RandomDeletion(p=0.5)
        result = t.apply(text_package, rng)
        orig_len = len(text_package["text"].words)
        new_len = len(result["text"].words)
        assert new_len < orig_len

    def test_at_least_one_word_remains(
        self, text_package, rng,
    ):
        from daugx.core.augmentation.text.random_deletion import (
            RandomDeletion,
        )
        t = RandomDeletion(p=1.0)
        result = t.apply(text_package, rng)
        assert len(result["text"].words) >= 1

    def test_p_zero_no_change(self, text_package, rng):
        from daugx.core.augmentation.text.random_deletion import (
            RandomDeletion,
        )
        t = RandomDeletion(p=0.0)
        result = t.apply(text_package, rng)
        assert result["text"].text == text_package[
            "text"
        ].text

    def test_label_preserved(self, text_package, rng):
        from daugx.core.augmentation.text.random_deletion import (
            RandomDeletion,
        )
        t = RandomDeletion(p=0.3)
        result = t.apply(text_package, rng)
        assert result["label"].name == "positive"

    def test_seeded_reproducibility(self, text_package):
        from daugx.core.augmentation.text.random_deletion import (
            RandomDeletion,
        )
        t = RandomDeletion(p=0.5)
        r1 = t.apply(
            text_package, np.random.default_rng(7),
        )
        r2 = t.apply(
            text_package, np.random.default_rng(7),
        )
        assert r1["text"].text == r2["text"].text
