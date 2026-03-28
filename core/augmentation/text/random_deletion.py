"""Random deletion augmentation (EDA).

Reference: Wei & Zou, "EDA: Easy Data Augmentation
Techniques for Boosting Performance on Text Classification
Tasks", 2019.
"""
from typing import Optional

import numpy as np

from daugx.core.augmentation.base import Transform
from daugx.core.data.components.text import Text
from daugx.core.data.data_package import DataPackage


class RandomDeletion(Transform):
    """Drop each word with probability *p*.

    At least one word is always preserved to avoid
    producing empty text.

    Args:
        p: Probability of deleting each word.
        text_key: Key in the DataPackage holding the
            ``Text`` component.
    """

    def __init__(
        self,
        p: float = 0.1,
        text_key: str = "text",
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(
                f"p must be in [0, 1], got {p}"
            )
        self.p = p
        self.text_key = text_key

    def _key(self) -> tuple:
        return (type(self).__name__, self.p, self.text_key)

    def apply(
        self,
        package: DataPackage,
        rng: Optional[np.random.Generator] = None,
    ) -> DataPackage:
        """Delete words with probability p.

        Args:
            package: Input data package.
            rng: Required random number generator.

        Returns:
            New DataPackage with modified text.
        """
        text_comp = package[self.text_key]
        words = text_comp.words
        if not words or rng is None or self.p == 0.0:
            return package

        # Keep words where random draw exceeds p
        kept = [w for w in words if rng.random() > self.p]

        # Ensure at least one word remains
        if not kept:
            kept = [words[int(rng.integers(len(words)))]]

        new_text = Text(
            " ".join(kept),
            text_comp.language,
            text_comp.metadata,
        )
        return package.replace(
            **{self.text_key: new_text},
        )
