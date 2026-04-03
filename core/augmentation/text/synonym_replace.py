"""Synonym replacement augmentation (EDA).

Reference: Wei & Zou, "EDA: Easy Data Augmentation
Techniques for Boosting Performance on Text Classification
Tasks", 2019.
"""
from typing import Dict, List, Optional, Tuple, Type

import numpy as np

from daugx.core.augmentation.base import Transform
from daugx.core.augmentation.text._synonyms import (
    DEFAULT_SYNONYMS,
)
from daugx.core.data.component import Component
from daugx.core.data.components.text import Text


class SynonymReplace(Transform):
    """Replace *n* random words with synonyms.

    Words are matched case-insensitively against a synonym
    mapping.  If fewer than *n* words have synonyms, all
    replaceable words are replaced.

    Args:
        n: Number of words to replace.
        synonyms: Custom ``{word: [synonyms]}`` mapping.
            Falls back to a small built-in thesaurus.
    """

    operates_on: Tuple[Type[Component], ...] = (Text,)

    def __init__(
        self,
        n: int = 1,
        synonyms: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.n = n
        self.synonyms = synonyms or DEFAULT_SYNONYMS

    def _key(self) -> tuple:
        return (type(self).__name__, self.n)

    def _apply(
        self,
        component: Component,
        rng: Optional[np.random.Generator] = None,
    ) -> Component:
        """Replace words in *component* with synonyms.

        Args:
            component: Text component to transform.
            rng: Required random number generator.

        Returns:
            New Text with replaced words, or the original
            component if rng is None or no words are
            replaceable.
        """
        text_comp = component  # type: Text
        if rng is None:
            return component
        words = text_comp.words
        if not words:
            return component

        replaceable = [
            i for i, w in enumerate(words)
            if w.lower() in self.synonyms
        ]
        if not replaceable:
            return component

        n = min(self.n, len(replaceable))
        chosen = rng.choice(
            replaceable, size=n, replace=False,
        )
        new_words = list(words)
        for idx in chosen:
            key = words[idx].lower()
            syns = self.synonyms[key]
            new_words[idx] = str(rng.choice(syns))

        return Text(
            " ".join(new_words),
            text_comp.language,
            text_comp.metadata,
            name=text_comp.name,
        )
