"""Random insertion augmentation (EDA).

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


class RandomInsertion(Transform):
    """Insert *n* synonym words at random positions.

    For each insertion: pick a random word that has
    synonyms, choose one of its synonyms, and insert it
    at a random position in the sentence.

    Args:
        n: Number of insertions.
        synonyms: Custom ``{word: [synonyms]}`` mapping.
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
        """Insert synonym words at random positions.

        Args:
            component: Text component to transform.
            rng: Required random number generator.

        Returns:
            New Text with inserted words, or the original
            component if rng is None or no words have
            synonyms.
        """
        text_comp = component  # type: Text
        if rng is None:
            return component
        words = list(text_comp.words)
        if not words:
            return component

        replaceable = [
            i for i, w in enumerate(words)
            if w.lower() in self.synonyms
        ]
        if not replaceable:
            return component

        for _ in range(self.n):
            idx = int(rng.choice(replaceable))
            key = words[idx].lower()
            syns = self.synonyms[key]
            syn = str(rng.choice(syns))
            pos = int(rng.integers(0, len(words) + 1))
            words.insert(pos, syn)

        return Text(
            " ".join(words),
            text_comp.language,
            text_comp.metadata,
            name=text_comp.name,
        )
