"""Synonym replacement augmentation (EDA).

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

    def __init__(
        self,
        n: int = 1,
        synonyms: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.n = n
        self.synonyms = synonyms or DEFAULT_SYNONYMS

    def _key(self) -> tuple:
        return (type(self).__name__, self.n)

    def apply(
        self,
        package: DataPackage,
        rng: Optional[np.random.Generator] = None,
    ) -> DataPackage:
        """Replace words with synonyms.

        Args:
            package: Input data package.
            rng: Required random number generator.

        Returns:
            New DataPackage with modified text.
        """
        text_comp = package.get(Text)
        if text_comp is None or rng is None:
            return package
        words = text_comp.words
        if not words:
            return package

        replaceable = [
            i for i, w in enumerate(words)
            if w.lower() in self.synonyms
        ]
        if not replaceable:
            return package

        n = min(self.n, len(replaceable))
        chosen = rng.choice(
            replaceable, size=n, replace=False,
        )
        new_words = list(words)
        for idx in chosen:
            key = words[idx].lower()
            syns = self.synonyms[key]
            new_words[idx] = str(rng.choice(syns))

        new_text = Text(
            " ".join(new_words),
            text_comp.language,
            text_comp.metadata,
            name=text_comp.name,
        )
        return package.replacing(text_comp, new_text)
