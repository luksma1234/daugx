"""Random deletion augmentation (EDA).

Reference: Wei & Zou, "EDA: Easy Data Augmentation
Techniques for Boosting Performance on Text Classification
Tasks", 2019.
"""
from typing import Optional, Tuple, Type

import numpy as np

from daugx.core.augmentation.base import Transform
from daugx.core.data.component import Component
from daugx.core.data.components.text import Text


class RandomDeletion(Transform):
    """Drop each word with probability *p*.

    At least one word is always preserved to avoid
    producing empty text.

    Args:
        p: Probability of deleting each word.
    """

    operates_on: Tuple[Type[Component], ...] = (Text,)

    def __init__(self, p: float = 0.1) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(
                f"p must be in [0, 1], got {p}"
            )
        self.p = p

    def _key(self) -> tuple:
        return (type(self).__name__, self.p)

    def _apply(
        self,
        component: Component,
        rng: Optional[np.random.Generator] = None,
    ) -> Component:
        """Delete words with probability p.

        Args:
            component: Text component to transform.
            rng: Required random number generator.

        Returns:
            New Text with deleted words, or the original
            component if rng is None or p is 0.
        """
        text_comp = component  # type: Text
        if rng is None or self.p == 0:
            return component
        words = text_comp.words
        if not words:
            return component

        # Keep words where random draw exceeds p
        kept = [w for w in words if rng.random() > self.p]

        # Ensure at least one word remains
        if not kept:
            kept = [words[int(rng.integers(len(words)))]]

        return Text(
            " ".join(kept),
            text_comp.language,
            text_comp.metadata,
            name=text_comp.name,
        )
