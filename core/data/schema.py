"""Schema — structural definition of a sample."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Type

from daugx.core.data.component import Component
from daugx.errors import SchemaValidationError


class Schema:
    """Structural definition of a sample.

    Describes what component types a valid sample must
    contain and under which keys.

    A key mapped to a bare ``Type[Component]`` means
    exactly one instance (singular).  A key mapped to a
    ``list[Type[Component]]`` means a variable-length list
    where each element is a tuple of those types (plural).

    Args:
        definition: Mapping of string keys to component
            types or lists of component types.

    Example::

        schema = Schema({
            "image": Image,
            "objects": [BoundingBox, Label],
        })
    """

    def __init__(
        self,
        definition: Dict[
            str,
            Type[Component] | List[Type[Component]],
        ],
    ) -> None:
        self._definition = definition

    @property
    def definition(
        self,
    ) -> Dict[
        str, Type[Component] | List[Type[Component]]
    ]:
        return self._definition

    @property
    def keys(self) -> Tuple[str, ...]:
        return tuple(self._definition.keys())

    @property
    def singular_keys(self) -> Tuple[str, ...]:
        """Keys expecting a single component."""
        return tuple(
            k for k, v in self._definition.items()
            if not isinstance(v, list)
        )

    @property
    def plural_keys(self) -> Tuple[str, ...]:
        """Keys expecting a list of component tuples."""
        return tuple(
            k for k, v in self._definition.items()
            if isinstance(v, list)
        )

    def validate(self, sample: Any) -> None:
        """Validate a sample against this schema.

        Args:
            sample: A :class:`Sample` instance.

        Raises:
            SchemaValidationError: If the sample does not
                conform to this schema.
        """
        for key, spec in self._definition.items():
            if key not in sample:
                raise SchemaValidationError(
                    f"Missing key: '{key}'"
                )
            value = sample[key]
            if isinstance(spec, list):
                self._validate_plural(key, spec, value)
            else:
                self._validate_singular(key, spec, value)

    def _validate_singular(
        self,
        key: str,
        expected_type: Type[Component],
        value: Any,
    ) -> None:
        if not isinstance(value, expected_type):
            raise SchemaValidationError(
                f"Key '{key}': expected "
                f"{expected_type.__name__}, "
                f"got {type(value).__name__}"
            )

    def _validate_plural(
        self,
        key: str,
        expected_types: List[Type[Component]],
        value: Any,
    ) -> None:
        if not isinstance(value, list):
            raise SchemaValidationError(
                f"Key '{key}': expected list, "
                f"got {type(value).__name__}"
            )
        for i, element in enumerate(value):
            if not isinstance(element, tuple):
                raise SchemaValidationError(
                    f"Key '{key}[{i}]': expected tuple, "
                    f"got {type(element).__name__}"
                )
            if len(element) != len(expected_types):
                raise SchemaValidationError(
                    f"Key '{key}[{i}]': expected tuple "
                    f"of length {len(expected_types)}, "
                    f"got {len(element)}"
                )
            for j, (comp, exp_type) in enumerate(
                zip(element, expected_types)
            ):
                if not isinstance(comp, exp_type):
                    raise SchemaValidationError(
                        f"Key '{key}[{i}][{j}]': "
                        f"expected "
                        f"{exp_type.__name__}, "
                        f"got {type(comp).__name__}"
                    )
