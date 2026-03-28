"""Random insertion augmentation (EDA).

Reference: Wei & Zou, "EDA: Easy Data Augmentation
Techniques for Boosting Performance on Text Classification
Tasks", 2019.
"""
from typing import Dict, List, Optional

import numpy as np

from daugx.core.augmentation.base import Transform
from daugx.core.augmentation.text._synonyms import (
    DEFAULT_SYNONYMS,
)
from daugx.core.data.components.text import Text
from daugx.core.data.data_package import DataPackage


class RandomInsertion(Transform):
    """Insert *n* synonym words at random positions.

    For each insertion: pick a random word that has
    synonyms, choose one of its synonyms, and insert it
    at a random position in the sentence.

    Args:
        n: Number of insertions.
        synonyms: Custom ``{word: [synonyms]}`` mapping.
        text_key: Key in the DataPackage holding the
            ``Text`` component.
    """

    def __init__(
        self,
        n: int = 1,
        synonyms: Optional[Dict[str, List[str]]] = None,
        text_key: str = "text",
    ) -> None:
        self.n = n
        self.synonyms = synonyms or DEFAULT_SYNONYMS
        self.text_key = text_key

    def _key(self) -> tuple:
        return (
            type(self).__name__,
            self.n,
            self.text_key,
        )

    def apply(
        self,
        package: DataPackage,
        rng: Optional[np.random.Generator] = None,
    ) -> DataPackage:
        """Insert synonym words at random positions.

        Args:
            package: Input data package.
            rng: Required random number generator.

        Returns:
            New DataPackage with modified text.
        """
        text_comp = package[self.text_key]
        words = list(text_comp.words)
        if not words or rng is None:
            return package

        replaceable = [
            i for i, w in enumerate(words)
            if w.lower() in self.synonyms
        ]
        if not replaceable:
            return package

        for _ in range(self.n):
            idx = int(rng.choice(replaceable))
            key = words[idx].lower()
            syns = self.synonyms[key]
            syn = str(rng.choice(syns))
            pos = int(rng.integers(0, len(words) + 1))
            words.insert(pos, syn)

        new_text = Text(
            " ".join(words),
            text_comp.language,
            text_comp.metadata,
        )
        return package.replace(
            **{self.text_key: new_text},
        )
