class InvalidActionError(RuntimeError):
    """Raised when attempting an action that by itself is invalid, but that can
    optionally be forced by removing conflicting edges."""
