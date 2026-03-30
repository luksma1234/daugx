"""Text component."""
from typing import Any, Dict, List, Optional

from daugx.core.data.component import (
    Component,
    ComponentState,
)


class Text(Component):
    """Plain text component.  Always materialized.

    Args:
        text: The text content.
        language: ISO 639-1 language code.
        metadata: Optional arbitrary metadata dict.
    """

    def __init__(
        self,
        text: str,
        language: str = "en",
        metadata: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> None:
        self._text = text
        self._language = language
        self._metadata = dict(metadata) if metadata else {}
        self._component_name = name

    @property
    def text(self) -> str:
        """The raw text content."""
        return self._text

    @property
    def language(self) -> str:
        """ISO 639-1 language code."""
        return self._language

    @property
    def metadata(self) -> Dict[str, Any]:
        """Copy of the metadata dict."""
        return dict(self._metadata)

    @property
    def words(self) -> List[str]:
        """Text split on whitespace."""
        return self._text.split()

    @property
    def state(self) -> ComponentState:
        return ComponentState.MATERIALIZED

    @property
    def is_materialized(self) -> bool:
        return True

    def materialize(self) -> None:
        """No-op for Text."""
        pass
