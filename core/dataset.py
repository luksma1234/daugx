"""Dataset — collection of samples conforming to a schema.

A Dataset performs zero filesystem I/O at construction.
All actual data loading is deferred to materialization.
"""
from __future__ import annotations

from typing import List, Optional

from daugx.core.data.sample import Sample
from daugx.core.data.schema import Schema
from daugx.errors import SchemaValidationError


class Dataset:
    """Collection of samples conforming to a schema.

    Args:
        schema: Structural definition all samples must
            match.
        samples: Preloaded samples.  Validated against
            *schema* at construction.
        name: Human-readable name.  Falls back to
            ``"Dataset"``.

    Raises:
        SchemaValidationError: If any sample violates
            the schema.
    """

    def __init__(
        self,
        schema: Schema,
        samples: List[Sample],
        name: Optional[str] = None,
    ) -> None:
        self._schema = schema
        self._samples = samples
        self._name = name or "Dataset"
        self._validate_all()

    @property
    def schema(self) -> Schema:
        return self._schema

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(self._samples)

    def __bool__(self) -> bool:
        return True

    def __getitem__(self, index: int) -> Sample:
        return self._samples[index]

    def _validate_all(self) -> None:
        """Validate every sample against the schema."""
        for i, sample in enumerate(self._samples):
            try:
                self._schema.validate(sample)
            except SchemaValidationError as exc:
                raise SchemaValidationError(
                    f"Sample {i}: {exc}"
                ) from exc
