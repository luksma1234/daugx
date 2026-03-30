"""Tests for text augmentation transforms."""
import numpy as np
import pytest

from daugx.core.augmentation.base import Transform
from daugx.core.data.components.label import ImageCategory
from daugx.core.data.components.text import Text
from daugx.core.data.data_package import DataPackage


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
    return DataPackage(
        Text("the good big happy cat"),
        ImageCategory(1, "positive"),
    )


@pytest.fixture
def no_synonym_package():
    """DataPackage with text that has no synonym matches."""
    return DataPackage(Text("xyz uvw abc"))


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
        orig_words = text_package.get(Text).words
        new_words = result.get(Text).words
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
        new_words = result.get(Text).words
        all_syns = set()
        for v in SYNONYMS.values():
            all_syns.update(v)
        orig_words = text_package.get(Text).words
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
        assert result.get(Text).text == "xyz uvw abc"

    def test_label_preserved(self, text_package, rng):
        from daugx.core.augmentation.text.synonym_replace import (
            SynonymReplace,
        )
        t = SynonymReplace(n=1, synonyms=SYNONYMS)
        result = t.apply(text_package, rng)
        assert result.get(ImageCategory).class_name == "positive"

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
        assert r1.get(Text).text == r2.get(Text).text

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
        orig_len = len(text_package.get(Text).words)
        new_len = len(result.get(Text).words)
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
        orig_words = set(text_package.get(Text).words)
        new_words = result.get(Text).words
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
        assert result.get(Text).text == "xyz uvw abc"

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
        assert r1.get(Text).text == r2.get(Text).text


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
        orig_len = len(text_package.get(Text).words)
        new_len = len(result.get(Text).words)
        assert new_len < orig_len

    def test_at_least_one_word_remains(
        self, text_package, rng,
    ):
        from daugx.core.augmentation.text.random_deletion import (
            RandomDeletion,
        )
        t = RandomDeletion(p=1.0)
        result = t.apply(text_package, rng)
        assert len(result.get(Text).words) >= 1

    def test_p_zero_no_change(self, text_package, rng):
        from daugx.core.augmentation.text.random_deletion import (
            RandomDeletion,
        )
        t = RandomDeletion(p=0.0)
        result = t.apply(text_package, rng)
        assert (
            result.get(Text).text
            == text_package.get(Text).text
        )

    def test_label_preserved(self, text_package, rng):
        from daugx.core.augmentation.text.random_deletion import (
            RandomDeletion,
        )
        t = RandomDeletion(p=0.3)
        result = t.apply(text_package, rng)
        assert result.get(ImageCategory).class_name == "positive"

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
        assert r1.get(Text).text == r2.get(Text).text
