"""InvalidComponentError — raised for invalid component
configurations in a Sample."""


class InvalidComponentError(Exception):
    """Raised when a Sample contains orphaned or ambiguous
    annotation components.

    Orphaned: annotation's ``target`` references a component
    name that does not exist in the sample.

    Ambiguous: annotation with ``target=None`` when multiple
    components of the matched type exist (e.g. two Images).
    """
