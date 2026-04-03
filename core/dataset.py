"""Dataset — collection of samples.

A Dataset performs zero filesystem I/O at construction.
All actual data loading is deferred to materialization.
"""
from __future__ import annotations

from typing import Iterator, List, Optional

from daugx.core.data.sample import Sample


class Dataset:
    """Collection of samples.

    Args:
        samples: Preloaded samples.
        name: Human-readable name.  Falls back to
            ``"Dataset"``.
    """

    def __init__(
        self,
        samples: List[Sample],
        name: Optional[str] = None,
    ) -> None:
        self._samples = samples
        self._name = name or "Dataset"

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(self._samples)

    def __bool__(self) -> bool:
        return True

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)

    def __getitem__(self, index: int) -> Sample:
        return self._samples[index]
